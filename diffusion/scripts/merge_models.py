import os
from omegaconf import OmegaConf
import argparse
import torch

def merge_ckpt(checkpoints, out_path, ckpt_index=-1):

    config_ref = OmegaConf.load(f"{checkpoints[0]}/config.yaml")
    for ckpt in checkpoints:
        config = OmegaConf.load(f"{ckpt}/config.yaml")
        if config != config_ref:
            print("Configs are not the same!")
    
    loaded_ckpts = []
    for ckpt in checkpoints:
        fname = sorted(os.listdir(f"{ckpt}/models/"))[ckpt_index] # ascending order
        loaded_ckpts.append(torch.load(f"{ckpt}/models/{fname}"))
        print(f"Loaded {fname} for {ckpt}")


    merged = {}

    # model
    merged['model'] = {}
    for stage, ckpt in enumerate(loaded_ckpts):
        for k, v in ckpt['model'].items():
            if k.startswith(f"unets.{stage}"):
                merged['model'][k] = v

    # version
    for ckpt in loaded_ckpts:
        if ckpt['version'] != loaded_ckpts[0]['version']:
            print("Versions are not the same!")
    merged['version'] = loaded_ckpts[0]['version']

    # steps
    merged['steps'] = loaded_ckpts[0]['steps']
    for stage, ckpt in enumerate(loaded_ckpts):
        merged['steps'][stage] = ckpt['steps'][stage]
    
    # time_elapsed
    merged['time_elapsed'] = sum([ ckpt['time_elapsed'] for ckpt in loaded_ckpts ])

    # scaler
    for stage, ckpt in enumerate(loaded_ckpts):
        merged[f'scaler{stage}'] = ckpt[f'scaler{stage}']
    
    # optim
    for stage, ckpt in enumerate(loaded_ckpts):
        merged[f'optim{stage}'] = ckpt[f'optim{stage}']

    # ema
    merged['ema'] = {}
    for stage, ckpt in enumerate(loaded_ckpts):
        for k, v in ckpt['ema'].items():
            if k.startswith(f"{stage}."):
                merged['ema'][k] = v

    if out_path == "-1":
        full_name = checkpoints[0].split("/")[-1]
        full_name = full_name[:-2] + "merged"
        out_path = os.path.join("/",*checkpoints[0].split("/")[:-1], full_name)
    
    os.makedirs(out_path, exist_ok=True)
    OmegaConf.save(config_ref, f"{out_path}/config.yaml")
    torch.save(merged, f"{out_path}/merged.pt")

    print(f"Saved merged model to {out_path}")
    return out_path


if __name__ == "__main__":
    # Get config and args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoints", nargs='+', required=True, help="Path to all checkpoint folders.")
    parser.add_argument("-o", type=str, default="-1")
    parser.add_argument("-d", type=int, default=-1, help="force a checkpoint index.")
    args = parser.parse_args()
    print(args.checkpoints)
    print(args.o)

    # check is names of checpoints match and order them is ascending order
    checkpoints = sorted(args.checkpoints, key=lambda x: int(x.split("/")[-1][-1])) # sort on stage number
    ref_name = checkpoints[0].split("/")[-1]
    for ckpt in checkpoints:
        name = ckpt.split("/")[-1] # format is name_uX with name the name of the experiment and X the unet stage
        assert name[:-1].endswith('stage'), "Checkpoint name is not in the correct format!"
        assert name[:-2] == ref_name[:-2], "Experiments names do not match!"
    
    print("Merging with the following order:")
    for i, ckpt in enumerate(checkpoints):
        print(f"Stage {i+1}: {ckpt}")

    merge_ckpt(checkpoints, args.o, args.d)