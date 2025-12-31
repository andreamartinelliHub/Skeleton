from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple

from src import utils

logger = utils.get_logger(__name__)

class Jack_Dataset(Dataset):
    def __init__(self, 
                 n_samples: int = 1_000, 
                 shape: int = 5):
        self.x = torch.randn(n_samples, shape)
        self.y = self.x[:, 1].unsqueeze(1)  # target = second component
        logger.info(f"Created DS with {n_samples} of shape {shape}")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

#   Dataset-specific configuration
# The dataset can store things like:
    # normalization stats
    # transforms
    # cached tensors
    # file paths
    # random seeds

    def get_dataloader(self, batch_size, shuffle, num_workers):
        dataloader = DataLoader(
            self,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,   # start here
            # pin_memory=True
        )
        logger.info(f"Num of batches: {len(dataloader)}")
        return dataloader