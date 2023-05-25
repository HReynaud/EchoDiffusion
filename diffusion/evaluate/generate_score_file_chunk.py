import os
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torch
from einops import rearrange
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pandas as pd

from imagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer, Imagen
from diffusion.dataset import EchoVideo

class LVEFRegressor:
    def __init__(self, exp_path, device) -> None:
        from af_lvef_regression.model import EFRegressor
        exp_root = os.path.join("/", *exp_path.split('/')[:-2])
        config = OmegaConf.load(os.path.join(exp_root, "config.yaml"))
        weights = os.path.join(exp_path)
        self.model = EFRegressor.load_from_checkpoint(weights, config=config)
        self.model.eval()
        self.model.freeze()
        self.model.to(device)
        self.device = device
    
    def __call__(self, x):
        with torch.no_grad():
            return self.model(x.to(self.device)).detach().cpu().squeeze()

def timestamp():
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[{t}]:"

if __name__ == "__main__":
    # Get config and args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to merged folder.")
    parser.add_argument("--reg", type=str, required=True, help="Path to trained regression model .ckpt file.")
    parser.add_argument("--bs", type=int, default="-1")
    parser.add_argument("--cond_scale", type=float, default=5., help="Evaluation batch size.")
    parser.add_argument("--max_samples", type=int, default=99999, help="Max number of samples to generate.")
    parser.add_argument("--num_noise", type=int, default=1, help="Number of repeptition .")
    parser.add_argument("--diffusion_steps", type=int, default=-1, help="Number of diffusion steps.")
    parser.add_argument("--save_videos", action="store_true", help="Save videos.", default=False)
    parser.add_argument("--rand_ef", action="store_true", help="Randomize EF input.", default=False)
    parser.add_argument("--chunks", type=int, default=1, help="Number of dataset splits for multi-gpu parallel computation.")
    parser.add_argument("--chunk", type=int, default=0, help="Rank of the current split.")
    parser.add_argument("--stop_at", type=float, default=-1, help="stop at a given unet in the cascade.")

    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(args.model, "config.yaml"))
    config = OmegaConf.merge(config, vars(args))

    print(f"Started work on chunk {args.chunk} of {args.chunks} at {datetime.now()}")

    # Overwrite config values with args
    config.dataset.num_frames = int(config.dataset.fps * config.dataset.duration)
    if args.stop_at == -1:
        args.stop_at = None

    # Get current exp name
    exp_name = args.model.split("/")[-1]

    # Load dataset
    val_ds = EchoVideo(config, ["VAL"], chunks=args.chunks, chunk=args.chunk) # evaluate on separate set, as noise is learned
    val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=args.bs)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    unets = []
    for i, (k, v) in enumerate(config.unets.items()):
        unets.append(Unet3D(**v, lowres_cond=(i>0))) # type: ignore

    imagen_klass = ElucidatedImagen if config.imagen.elucidated == True else Imagen
    del config.imagen.elucidated
    imagen = imagen_klass(
        unets = unets,
        **OmegaConf.to_container(config.imagen), # type: ignore
    )

    trainer = ImagenTrainer(
        imagen = imagen,
        **config.trainer
    ).to(device)

    # Load model
    path_to_model = os.path.join(args.model, "merged.pt")
    assert os.path.exists(path_to_model), f"Model {path_to_model} does not exist. Did you merge the checkpoints?"
    additional_data = trainer.load(path_to_model)
    print(f"Loaded model {path_to_model}")
    trainer.eval()

    # Prepare saving folder
    save_folder = os.path.join(args.model, "eval")
    os.makedirs(save_folder, exist_ok = True)
    causal_name = "counterfactual" if args.rand_ef else "factual"
    videos_save_folder = os.path.join(save_folder, f"{causal_name}_videos")
    if args.save_videos:
        os.makedirs(videos_save_folder, exist_ok = True)
    csv_save_folder = os.path.join(save_folder, f"{causal_name}_csvs")
    os.makedirs(csv_save_folder, exist_ok = True)

    # Set up LVEF regressor
    lvef_scorer = LVEFRegressor(
        exp_path=args.reg,
        device=device
    )
    lvef_target = []
    lvef_gen = []
    lvef_ref = []
    
    # Set up SSIM scorer
    ssim_scorer = StructuralSimilarityIndexMeasure(data_range=1.0) # higher better, [0-1]
    ssim_gen = []

    # Set up LPIPS scorer
    lpips_scorer = LearnedPerceptualImagePatchSimilarity(net_type='vgg') # lower better, [0-1]
    lpips_gen = []

    filenames = []

    # Set up counterfactual evaluation
    target_ef = None
    if args.rand_ef: # use random EFs but keep them with different noise samples for counterfactual evaluation
        print("Using random embeddings for counterfactual evaluation.")
        target_ef = val_ds[0][1][None,...].repeat(len(val_ds), 1, 1) # type: ignore
        target_ef = torch.rand_like(target_ef)*0.7+0.15

    print("Loaded all scorers.")
    count = 0

    with torch.no_grad():
        for noise_idx in range(args.num_noise):
            print(f"{timestamp()} Starting noise {noise_idx+1}/{args.num_noise} for chunk {args.chunk}")
            for i, batch in enumerate(val_dl):
                if i == len(val_dl)//2:
                    print(f"{timestamp()} Chunk {args.chunk} is halfway through noise {noise_idx+1}/{args.num_noise}")

                ref_videos = batch[0]

                B, C, T, H, W = ref_videos.shape
                fps = config.dataset.fps
                sidx = i * B
                eidx = sidx + B
                embeddings = batch[1] if not args.rand_ef else target_ef[sidx:eidx] # type: ignore

                cimage = batch[2]
                fname = batch[3]
                sample_kwargs = {
                    "text_embeds": embeddings,
                    "cond_scale": args.cond_scale,
                    "cond_images": cimage,
                    "stop_at_unet_number": args.stop_at,
                }

                # Generate videos
                gen_videos = trainer.sample(
                    batch_size=args.bs,
                    video_frames=config.dataset.num_frames,
                    **sample_kwargs,
                    use_tqdm=False,
                ).detach().cpu() # type: ignore

                # upscale to native resolution
                if gen_videos.shape[-3:] != (T, H, W):
                    gen_videos = torch.nn.functional.interpolate(gen_videos, size=(T, H, W), mode='trilinear', align_corners=False)

                ref_video = ref_videos.cpu()
                gen_video = gen_videos.cpu()

                filenames.extend([fname[k].split('.')[0] for k in range(B)])

                ef = lvef_scorer(ref_video)
                lvef_ref.extend(ef.flatten().numpy())

                ef = lvef_scorer(gen_video)
                lvef_gen.extend(ef.flatten().numpy())

                lvef_target.extend(sample_kwargs['text_embeds'].flatten().numpy()*100.0)

                ssim_gen.extend([ssim_scorer(p, t).item() for p, t in zip(gen_video, ref_video)]) # type: ignore

                normalised_gen_video = rearrange(gen_video * 2. - 1. , 'b c t h w -> b t c h w')
                normalised_ref_video = rearrange(ref_video * 2. - 1. , 'b c t h w -> b t c h w')
                lpips_gen.extend([lpips_scorer(p, t).item() for p, t in zip(normalised_gen_video, normalised_ref_video)]) # type: ignore

                # Save videos
                if args.save_videos and noise_idx==0 :
                    
                    # Convert videos to uint8
                    byte_gen_video = gen_video.multiply(255).byte().permute(0, 2, 3, 4, 1) # B x C x T x H x W -> B x T x H x W x C
                    byte_ref_video = ref_video.multiply(255).byte().permute(0, 2, 3, 4, 1) # B x C x T x H x W -> B x T x H x W x C

                    for k in range(B):
                        all_byte_videos = torch.cat([
                            byte_ref_video[k],
                            byte_gen_video[k],
                        ], dim = 2) # concat on width
                        
                        all_byte_videos = [Image.fromarray(i) for i in all_byte_videos.numpy()]
                        suffix = f"_{noise_idx}" if args.rand_ef else ""
                        all_byte_videos[0].save(os.path.join(videos_save_folder, f"{fname[k].split('.')[0]}{suffix}.gif"), save_all=True, append_images=all_byte_videos[1:], duration=1000/fps, loop=0) # type: ignore

            print()
            print(f"{timestamp()} Finished {noise_idx+1} of {args.num_noise} passes on chunk {args.chunk}.")

    scores = list(zip(
        filenames,
        lvef_target,
        lvef_gen,
        lvef_ref,
        ssim_gen,
        lpips_gen,
    ))

    pd.DataFrame(
        scores, 
        columns=["FileName", "GT EF", "Gen EF", "Ref EF", "SSIM", "LPIPS"]
        ).to_csv(os.path.join(csv_save_folder, f"scores_{causal_name}_{args.chunk}.csv"), index=False)
    print()
    print(f"{timestamp()} {args.chunk} Done.")

