import os
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch
import pandas as pd

from imagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer
from diffusion.dataset import EchoVideo
from utils import savevideo

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
    parser.add_argument("--model", type=str, required=True, help="Path to merged folder")
    parser.add_argument("--bs", type=int, default="-1")
    parser.add_argument("--cond_scale", type=float, default=5., help="Evaluation batch size.")
    parser.add_argument("--chunks", type=int, default=1, help="Number of diffusion steps.")
    parser.add_argument("--chunk", type=int, default=0, help="Number of diffusion steps.")
    parser.add_argument("--stop_at", type=float, default=-1, help="stop a certain UNET")
    parser.add_argument("--ef_list", type=str, help="path to csv containing all requested EFs")

    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(args.model, "config.yaml"))
    config = OmegaConf.merge(config, vars(args))

    print(f"Started work on chunk {args.chunk} of {args.chunks} at {datetime.now()}")

    # Overwrite config values with args
    config.dataset.num_frames = int(config.dataset.fps * config.dataset.duration)
    if os.uname().nodename.endswith("doc.ic.ac.uk"):
        config.dataset.data_path = "/data/hjr119/echo_ds/EchoNet-Dynamic"
        config.checkpoint.path = "/data/hjr119/diffusion_models/imagen_video"
    if args.stop_at == -1:
        args.stop_at = None

    # Get current exp name
    exp_name = args.model.split("/")[-1]

    # Load dataset
    train_ds = EchoVideo(config, ["TRAIN"], chunks=args.chunks, chunk=args.chunk) # evaluate on separate set, as noise is learned
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.bs, pin_memory=True, drop_last=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    unets = []
    for i, (k, v) in enumerate(config.unets.items()):
        unets.append(Unet3D(**v, lowres_cond=(i>0))) # type: ignore

    imagen = ElucidatedImagen(
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
    save_folder = os.path.join(args.model, "balancing_samples")
    os.makedirs(save_folder, exist_ok = True)
    videos_save_folder = os.path.join(save_folder, f"videos")
    os.makedirs(videos_save_folder, exist_ok = True)
    csv_save_folder = os.path.join(save_folder, f"csvs")
    os.makedirs(csv_save_folder, exist_ok = True)

    # Set up LVEF regressor
    root_ef = "/home/atuin/b143dc/b143dc18/merged_models/ef_regression/" 
    if os.uname().nodename.endswith("doc.ic.ac.uk"):
        root_ef = "/data/hjr119/diffusion_models/ef_regression/"
    lvef_scorer = LVEFRegressor(
        exp_path=os.path.join(root_ef, "ef_reg_20230128_150523_ef_112_32_2/checkpoints/epoch=28.ckpt"),
        device=device
    )

    # Set up counterfactual evaluation
    ef_csv = pd.read_csv(args.ef_list).to_numpy().flatten()/100.0
    target_ef = torch.tensor(ef_csv).float().to(device)
    ef_start = int(len(ef_csv) * args.chunk / args.chunks)
    ef_end = int(len(ef_csv) * (args.chunk+1) / args.chunks)
    target_ef = target_ef[ef_start:ef_end].unsqueeze(1).unsqueeze(2)

    # track data
    cond_image_source = []
    targetted_ef = []
    predicted_ef = []
    unique_ids = []

    counter = ef_start
    generate = True

    try:
        with torch.no_grad():
            while generate:
                for i, batch in enumerate(train_dl):

                    # Print progress every 10%
                    if counter % int((ef_end-ef_start)/10+1) == 0:
                        print(f"Progress on chunk {args.chunk}: {(counter-ef_start)/(ef_end-ef_start)*100}%")

                    ref_videos = batch[0]

                    B, C, T, H, W = ref_videos.shape
                    fps = config.dataset.fps
                    sidx = i * B
                    eidx = sidx + B
                    embeddings = target_ef[sidx:eidx] if eidx < len(target_ef) else \
                        torch.cat([target_ef[sidx:], torch.rand_like(target_ef)[:(eidx - len(target_ef))]], dim=0)

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
                    # gen_videos = torch.rand((B, 3, T, H, W))

                    # upscale to native resolution
                    if gen_videos.shape[-3:] != (T, H, W):
                        gen_videos = torch.nn.functional.interpolate(gen_videos, size=(T, H, W), mode='trilinear', align_corners=False)

                    gen_video = gen_videos.cpu()

                    tmp_cond_image_source = [fname[k].split('.')[0] for k in range(B)]
                    tmp_targetted_ef = embeddings.cpu().numpy().flatten()*100.
                    tmp_predicted_ef = lvef_scorer(gen_video).cpu().numpy().flatten()

                    # Convert videos to uint8
                    byte_gen_video = gen_video.multiply(255).byte().permute(0, 1, 2, 3, 4).numpy() # B x C x T x H x W -> B x T x H x W x C

                    for k in range(B):
                        vname = f"{str(counter).zfill(6)}.avi"

                        unique_ids.append(vname)
                        cond_image_source.append(tmp_cond_image_source[k])
                        targetted_ef.append(tmp_targetted_ef[k].item())
                        predicted_ef.append(tmp_predicted_ef[k].item())

                        vpath = os.path.join(videos_save_folder, vname)
                        savevideo(vpath, byte_gen_video[k], fps=fps)
                        counter += 1
                        if counter > ef_end:
                            generate = False
                            break
    except:
        pass

    pd.DataFrame(
        list(zip(cond_image_source, unique_ids, targetted_ef, predicted_ef)), 
        columns=["CondImageFileName", "ID", "TargetEF", "EstimatedEF"]
        ).to_csv(os.path.join(csv_save_folder, f"report_{args.chunk}.csv"), index=False)
    print()
    print(f"{timestamp()} {args.chunk} Done.")

