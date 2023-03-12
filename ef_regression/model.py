import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
import wandb # type: ignore
import sklearn.metrics
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights



class RegressionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        self.model.fc.bias.data[0] = 55.6

    def forward(self, x):
        return self.model(x)


class EFRegressor(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = RegressionModel(config)
        self.lossf = nn.MSELoss()

        self.idx_labl = {"train": [], "val": []}
        self.idx_pred = {"train": [], "val": []}
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.config.trainer.lr) # type: ignore
        return opt
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")
    
    def shared_epoch_end(self, outputs, name):
        if self.local_rank == 0:
            data = [[x, y] for (x, y) in zip(self.idx_labl[name], self.idx_pred[name])]
            table = wandb.Table(data=data, columns=["Label", "Prediction"])
            self.logger.experiment.log( # type: ignore
                {
                    f"{name}/scatter": wandb.plot.scatter( # type: ignore
                        table, "Label", "Prediction", title="Label vs Prediction"
                    ),
                    f"{name}/R2": sklearn.metrics.r2_score(
                        self.idx_labl[name], self.idx_pred[name]
                    ),
                    f"{name}/MAE": sklearn.metrics.mean_absolute_error(
                        self.idx_labl[name], self.idx_pred[name]
                    ),
                    f"{name}/MSE": sklearn.metrics.mean_squared_error(
                        self.idx_labl[name], self.idx_pred[name]
                    ),
                }
            )
            self.idx_labl[name] = []
            self.idx_pred[name] = []
        return None

    def shared_step(self, batch, batch_idx, name):
        videos, lvefs  = batch

        pred_lvef = self.model(videos.float()).view(-1)

        loss = self.lossf(pred_lvef, lvefs.half())

        self.idx_labl[name].extend(lvefs.cpu().detach().tolist())
        self.idx_pred[name].extend(pred_lvef.cpu().detach().tolist())

        self.log(f"{name}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss