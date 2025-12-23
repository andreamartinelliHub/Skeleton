import numpy as np
import torch
import random
import matplotlib.pyplot as plt
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
        weights_array,
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