from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple

import logging
logger = logging.getLogger(__name__)

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
    
def get_dataloaders(conf):
    train_ds = Jack_Dataset(
        n_samples=conf.data.train_samples, 
        shape=conf.data.array_dim)
    train_dl = train_ds.get_dataloader(conf.data.batch_size, conf.data.shuffle, conf.data.num_workers)
    
    val_ds = Jack_Dataset(
        n_samples=conf.data.val_samples, 
        shape=conf.data.array_dim)
    val_dl = val_ds.get_dataloader(conf.data.batch_size, conf.data.shuffle, conf.data.num_workers)

    test_ds = Jack_Dataset(
        n_samples=conf.data.test_samples, 
        shape=conf.data.array_dim)
    test_dl = test_ds.get_dataloader(conf.data.batch_size, conf.data.shuffle, conf.data.num_workers)

    return train_dl, val_dl, test_dl