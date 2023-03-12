import os
from omegaconf import OmegaConf
import argparse
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torch

from imagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer, Imagen
from diffusion.dataset import EchoVideo


if __name__ == "__main__":
    # Get config and args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to merged folder")
    parser.add_argument("--cond_scale", type=float, default=5., help="Evaluation batch size.")
    parser.add_argument("--diffusion_steps", type=int, default=-1, help="Number of diffusion steps.")
    parser.add_argument("--save_videos", action="store_true", help="Save videos.", default=False)
    parser.add_argument("--rand_ef", action="store_true", help="Randomize EF input.", default=False)
    parser.add_argument("--stop_at", type=float, default=-1, help="stop a certain UNET")

    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(args.model, "config.yaml"))
    config = OmegaConf.merge(config, vars(args))
    bs = 1

    # Overwrite config values with args
    config.dataset.num_frames = int(config.dataset.fps * config.dataset.duration)
    if args.stop_at == -1:
        args.stop_at = None

    # Get current exp name
    exp_name = args.model.split("/")[-1]

    # Load dataset
    val_ds = EchoVideo(config, ["VAL"]) # evaluate on separate set, as noise is learned
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=True, num_workers=bs)

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
    save_folder = os.path.join(args.model, "samples")
    os.makedirs(save_folder, exist_ok = True)
    suffix = "counterfactual" if args.rand_ef else "factual"

    with torch.no_grad():
        batch = next(iter(val_dl))
        ref_videos = batch[0]

        B, C, T, H, W = ref_videos.shape
        fps = config.dataset.fps
        embeddings = batch[1] if not args.rand_ef else torch.rand_like(batch[1])*0.7+0.15 # type: ignore

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
            batch_size=bs,
            video_frames=config.dataset.num_frames,
            **sample_kwargs,
            use_tqdm=True,
        ).detach().cpu() # type: ignore

        # upscale to native resolution
        if gen_videos.shape[-3:] != (T, H, W):
            gen_videos = torch.nn.functional.interpolate(gen_videos, size=(T, H, W), mode='trilinear', align_corners=False)

        ref_video = ref_videos.cpu()
        gen_video = gen_videos.cpu()

        # Convert videos to uint8
        byte_gen_video = gen_video.multiply(255).byte().permute(0, 2, 3, 4, 1) # B x C x T x H x W -> B x T x H x W x C
        byte_ref_video = ref_video.multiply(255).byte().permute(0, 2, 3, 4, 1) # B x C x T x H x W -> B x T x H x W x C

        byte_videos = torch.cat([
                byte_ref_video[0],
                byte_gen_video[0],
        ], dim = 2) # concat on width
            
        byte_videos = [Image.fromarray(i) for i in byte_videos.numpy()]
        save_path = os.path.join(save_folder, f"{fname[0].split('.')[0]}_{suffix}.gif")
        byte_videos[0].save(save_path, save_all=True, append_images=byte_videos[1:], duration=1000/fps, loop=0) # type: ignore

    print(f"Generated videos saved to {save_path} with LVEF {embeddings[0].item():.2f}")