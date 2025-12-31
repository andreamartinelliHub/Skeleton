import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
import pytorch_lightning as pl
from lightning.pytorch.callbacks import Callback
from tqdm import trange

from src import utils
logger = utils.get_logger(__name__)

class Jack_Model(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 use_bias: bool):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=use_bias)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
def get_optimizer(model, optim_name, lr):
    optimizer = eval(f'torch.optim.{optim_name}')
    return optimizer(model.parameters(), lr)

def get_loss_fn(loss_name):
    loss_fn = eval(f'nn.{loss_name}')
    return loss_fn

# train.py should answer one question only:
# “Given a model and data, how do we train it?”
def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device | str = "cpu",
    use_tqdm = True
) -> list[torch.Tensor]:
    
    model.to(device)
    model.train()

    weights_evolution = []

    logger.info('Start Train')
    iterator = trange(epochs) if use_tqdm else range(epochs)
    for epoch in iterator:
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if use_tqdm:
            iterator.set_description(f"Epoch {epoch+1:02d}/{epochs} | loss={avg_loss:.6f}") # type: ignore
        else:
            logger.info(f"Epoch {epoch+1:02d}/{epochs} | loss={avg_loss:.6f}")

        # inspect model.linear weights during epochs
        weights = model.linear.weight.detach().cpu().clone() # type: ignore
        weights_evolution.append(weights)

    return weights_evolution

@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str = "cpu",
) -> Dict[str, float]:
    model.to(device)
    model.eval()

    total_loss = 0.0

    for x, y in eval_loader:
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        total_loss += loss.item()

    avg_loss = total_loss / len(eval_loader)

    logger.info(f"[EVAL] loss={avg_loss:.6f}")

    return {"loss": avg_loss}

# ---------------------------
# Lightning module wrapper
# ---------------------------
class JackModule(pl.LightningModule):
    def __init__(self, backbone, conf):
        super().__init__()
        self.conf = conf
        self.net = backbone(conf.data.array_dim, conf.model.use_bias)
        # to enable intermediate input-ouput sizes
        self.example_input_array = torch.Tensor(10, conf.data.array_dim)
        # in this case train loss == test_loss
        self.loss_fn = get_loss_fn(conf.model.loss_name) 

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # if neededm, adjust shape of x or y here
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim_name = self.conf.model.optim_name
        optimizer = eval(f'torch.optim.{optim_name}')
        return optimizer(self.parameters(), lr=self.conf.optim[optim_name].lr)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y) # change loss if needed
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y) # change loss if needed
        self.log("test_loss", loss, on_step=False, on_epoch=True)
    

# ---------------------------
# Lightning Callbacks: 
# Callbacks allow you to add arbitrary self-contained programs to your training
# ---------------------------
class WeightsSaver(Callback):
    def __init__(self, net_name="net"):
        self.net_name = net_name
        self.epoch_weights = {}
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        net = getattr(pl_module, self.net_name)
        weights = net.linear.weight.detach().cpu()  # Shape: [out_features, in_features]
        # breakpoint()
        self.epoch_weights[epoch] = weights.clone().squeeze()
        # logger.info(f"Epoch {epoch}: stored shape {weights.shape}, sample {weights[0][:3]}...")


        