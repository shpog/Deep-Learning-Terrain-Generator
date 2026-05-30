"""
Dataset and DataLoader definitions for the Terrain Generator.

This module provides the PyTorch Dataset interface to stream training patches
directly from the massive memory-mapped .npy file on the hard drive, ensuring
local RAM is not overwhelmed during training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import PROCESSED_DATA_DIR, RESOLUTION, TILE_SIZE

class TerrainDataset(Dataset):
    """
    A PyTorch Dataset that randomly crops tiles from a global terrain map.

    Because the dataset consists of a single massive continuous image rather 
    than thousands of separate files, this dataset ignores standard indexing 
    and instead returns a random geographical crop upon every fetch.

    Attributes:
        epoch_size (int): The number of patches that define one full training epoch.
        filepath (pathlib.Path): The path to the processed .npy data file.
        data (numpy.memmap): The memory-mapped global terrain array.
        channels (int): Number of data channels (e.g., 3).
        height (int): Total pixel height of the global map.
        width (int): Total pixel width of the global map.
    """
    def __init__(self, epoch_size=10000):
        """
        Initializes the Dataset and opens the memory-mapped file.

        Args:
            epoch_size (int, optional): Since we randomly crop from a continuous map, 
                "length" is arbitrary. This defines how many patches constitute 
                one full pass. Defaults to 10000.
        """
        self.epoch_size = epoch_size
        self.filepath = PROCESSED_DATA_DIR / f"worldclim_{RESOLUTION}_full.npy"
        
        temp_data = np.load(self.filepath, mmap_mode='r')
        self.channels, self.height, self.width = temp_data.shape
        del temp_data 
        
        self.data = None

    def __len__(self):
        """
        Returns the arbitrary size of the epoch.

        Returns:
            int: The predefined epoch size.
        """
        return self.epoch_size

    def __getitem__(self, idx):
        """
        Fetches a single random patch of size [TILE_SIZE, TILE_SIZE].

        This method actively filters out patches that consist entirely of ocean 
        (where elevation max is 0.0) to prevent the neural network from suffering 
        mode collapse on empty water tiles.

        Args:
            idx (int): The index requested by the PyTorch DataLoader. 
                (Note: This is ignored in favor of random spatial cropping).

        Returns:
            torch.Tensor: A float32 tensor of shape [Channels, TILE_SIZE, TILE_SIZE].
        """
        if self.data is None:
            self.data = np.load(self.filepath, mmap_mode='r')

        MIN_DELTA = 0.05

        while True:
            y = np.random.randint(0, self.height - TILE_SIZE)
            x = np.random.randint(0, self.width - TILE_SIZE)
            
            patch_numpy = self.data[:, y:y+TILE_SIZE, x:x+TILE_SIZE]
            elevation = patch_numpy[0]
            
            delta = np.max(elevation) - np.min(elevation)

            if np.mean(elevation) > 0.01 and delta > MIN_DELTA: 
                break
                
        patch_tensor = torch.from_numpy(np.array(patch_numpy).copy()).float()
        return patch_tensor

def get_dataloader(batch_size=32, num_workers=0):
    """
    Creates the PyTorch DataLoader for the TerrainDataset.

    Args:
        batch_size (int, optional): Number of patches per batch. Defaults to 32.
        num_workers (int, optional): Subprocesses to use for data loading. 
            On multi-GPU clusters, higher values (e.g., 4 or 8) speed up fetching. 
            Defaults to 0.

    Returns:
        torch.utils.data.DataLoader: The configured PyTorch DataLoader.
    """
    dataset = TerrainDataset()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader