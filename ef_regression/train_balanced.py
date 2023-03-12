import os, shutil
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger


from ef_regression.dataset import EchoVideoEF, EchoVideoEFextended #type: ignore
from ef_regression.model import EFRegressor #type: ignore


if __name__ == "__main__":
    torch.hub.set_dir(".cache")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    run_name = "ef_reg_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + os.path.basename(args.config).split(".")[0]

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)

    model = EFRegressor(config)

    os.makedirs(os.path.join(config.checkpoint.path, run_name))
    shutil.copy(args.config, os.path.join(config.checkpoint.path, run_name, "config.yaml"))

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.checkpoint.path, run_name, "checkpoints"),
        filename='{epoch}',
        save_top_k=3,
        monitor='val/loss',
        mode='min',
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    logger = WandbLogger(
        name=run_name,
        project=config.wandb.project,
        config=OmegaConf.to_container(config, resolve=True),
        entity=config.wandb.entity,
    )


    train_ds = EchoVideoEFextended(config, splits=["TRAIN"], balanced_bins=config.elem_per_bin, minef=config.min_ef, maxef=config.max_ef)
    val_ds = EchoVideoEF(config, splits=["VAL"])

    train_dl = DataLoader(train_ds, batch_size=config.dataloader.batch_size, shuffle=True, num_workers=config.dataloader.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=config.dataloader.batch_size, shuffle=False, num_workers=config.dataloader.num_workers, pin_memory=True)


    trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        accelerator="gpu",
        precision=16,
        strategy="ddp",
    )

    trainer.fit(model, train_dl, val_dl)

    print("Done")