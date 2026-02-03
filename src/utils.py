import numpy as np
import torch
import random
import matplotlib.pyplot as plt
# from optuna.integration import PyTorchLightningPruningCallback
import seaborn as sns
import logging

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # GPU exact reproducibility:
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # slower, but reproducible
    torch.backends.cudnn.benchmark = False

def get_device(preferred: str):
    if preferred=='auto':
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(preferred)

def get_logger(name: str = __name__):
    logger = logging.getLogger(name)

    # Prevent adding multiple handlers if logger already exists
    if not logger.handlers:
        # StreamHandler outputs to console
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] [%(funcName)s()] %(message)s",
            # "%(asctime)s [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d - %(funcName)s()] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)  # Default level, can be changed later
        # Optional: avoid propagating to root logger to prevent duplicate logs
        logger.propagate = False

    return logger


def plot_weights_heatmap(weights_array, figsize = (10,4)):
    plt.figure(figsize = figsize)
    sns.heatmap(
        np.array(weights_array),
        cmap="RdBu",
        annot=True,          # <--- this shows the value in each cell
        fmt=".2f",           # <--- format numbers
        cbar_kws={"label": "Weight value"},
        linewidths=0.5,      # optional: lines between cells
        center=0,            # optional: center colormap at 0
    )
    plt.xlabel("Input dimension")
    plt.ylabel("Epoch")
    plt.title("Linear Layer Weights Over Epochs")
    plt.tight_layout()
    plt.show()

# ---------------------------
# Optuna objective
# # ---------------------------
# def objective(trial, module, ):
#     lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    
#     # 2. Initialize your model with this LR
#     model = MyLightningModule(lr=lr)
    
#     # 3. Setup the Trainer
#     trainer = L.Trainer(
#         max_epochs=5,
#         accelerator="auto",
#         # We add a pruning callback to stop bad trials early
#         callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")]
#     )
    
#     # 4. Train
#     trainer.fit(model, datamodule=dm)
    
#     # 5. Return the metric you want to optimize
#     return trainer.callback_metrics["val_loss"].item()
    # Suggest hyperparameters
    # lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # # weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # # Update conf dynamically
    # conf.optim[conf.model.optim_name].lr = lr
    # # conf.optim[conf.model.optim_name].weight_decay = weight_decay

    # trainer.fit(module, dl)
    # # Lightning logs metrics internally
    # loss = trainer.callback_metrics["train_loss"].item()
    # return loss

# # --- PHASE 1: Optuna Optimization ---
# def objective(trial, module, trainer, dataloader):
#     # 1. Hyperparameters to tune
#     params = {
#         "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
#         "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
#     }
    
#     # 2. Setup (Note: Re-instantiating loaders per trial is safer for batch_size changes)
#     # train_loader, val_loader, _ = get_loaders(params['batch_size'])
    
#     # 4. Pruning Callback (stops bad trials early)
#     # pruner = PyTorchLightningPruningCallback(trial, monitor="val_acc")
    
#     # trainer = pl.Trainer(
#     #     max_epochs=3, # Small number for tuning, but >1 allows pruning to work
#     #     accelerator="auto",
#     #     enable_checkpointing=False,
#     #     logger=False,
#     #     callbacks=[pruner]
#     # )
    
#     trainer.fit(module, dataloader)
    
#     return trainer.callback_metrics["val_acc"].item()