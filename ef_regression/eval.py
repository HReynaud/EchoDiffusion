import os
from omegaconf import OmegaConf
import argparse
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

from ef_regression.dataset import EchoVideoEF, EchoVideoEFextended #type: ignore
from ef_regression.model import EFRegressor #type: ignore

if __name__ == "__main__":

    torch.hub.set_dir(".cache")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(args.model, "config.yaml"))
    seed_everything(config.seed)
    
    ckpt_name = sorted(os.listdir(os.path.join(args.model, "checkpoints")))[-2]
    ckpt_path = os.path.join(args.model, "checkpoints", ckpt_name)
    print(f"Loading model from {ckpt_path}.")

    model = EFRegressor.load_from_checkpoint(ckpt_path, config=config)
    model = model.cuda()
    model.eval()

    val_dss = {}
    # Balanced set:
    val_dss['bin10'] = EchoVideoEFextended(config, splits=["VAL", "TEST"], balanced_bins=10, minef=0, maxef=100)
    # Original validation set
    val_dss['val'] = EchoVideoEF(config, splits=["VAL"], fix_samples=True)

    trgt_efs = []
    pred_efs = []

    with torch.no_grad():
        for k, val_ds in val_dss.items():
            val_dl   = DataLoader(val_ds, batch_size=config.dataloader.batch_size, shuffle=False, num_workers=min(os.cpu_count(),config.dataloader.num_workers), drop_last=False)
            trgt_efs = []
            pred_efs = []
            for batch in tqdm(val_dl):
                frams = batch[0].cuda()
                lvefs = batch[1]
                
                preds = model(frams)
        
                trgt_efs.extend(lvefs.cpu().view(-1).numpy())
                pred_efs.extend(preds.cpu().view(-1).numpy())

            # Compute R2 score
            r2 = r2_score(trgt_efs, pred_efs)

            # Compute MAE
            mae = np.mean(np.abs(np.array(trgt_efs) - np.array(pred_efs)))

            # Compute RMSE
            rmse = np.sqrt(np.mean(np.square(np.array(trgt_efs) - np.array(pred_efs))))

            print(f"{ckpt_path.split('/')[-3]} ({k}): R2: {r2:<0.2f}, MAE: {mae:<0.2f}, RMSE: {rmse:<0.2f}")

            print(
                f"{ckpt_path.split('/')[-3]} ({k}): R2: {r2:<0.2f}, MAE: {mae:<0.2f}, RMSE: {rmse:<0.2f}", 
                file=open(os.path.join(args.model, "eval.txt"), "a")
                )
            print()
    print("Done.")
