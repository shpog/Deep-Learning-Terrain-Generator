import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import PROCESSED_DATA_DIR, RESOLUTION, TILE_SIZE

class TerrainDataset(Dataset):
    def __init__(self, epoch_size=10000):
        """
        Args:
            epoch_size (int): Since we are randomly cropping from a single continuous map, 
                              "length" is arbitrary. This defines how many patches 
                              constitute one full pass (epoch) of training.
        """
        self.epoch_size = epoch_size
        self.filepath = PROCESSED_DATA_DIR / f"worldclim_{RESOLUTION}_full.npy"
        
        self.data = np.load(self.filepath, mmap_mode='r')
        self.channels, self.height, self.width = self.data.shape

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        """
        Fetches a single TILE_SIZExTILE_SIZE patch. 
        PyTorch uses the 'idx' to iterate, but since we are randomly cropping, 
        we actually ignore 'idx' and just pick random coordinates.
        We ignore samples with no land on them
        """
        while True:
            y = np.random.randint(0, self.height - TILE_SIZE)
            x = np.random.randint(0, self.width - TILE_SIZE)
            
            patch_numpy = self.data[:, y:y+TILE_SIZE, x:x+TILE_SIZE]
            
            if np.max(patch_numpy[0]) > 0.0:
                break
                
        patch_tensor = torch.from_numpy(np.array(patch_numpy)).float()
        
        return patch_tensor

def get_dataloader(batch_size=32, num_workers=0):
    """
    Creates the PyTorch DataLoader that will handle batching and multi-processing.
    Note: On Windows, setting num_workers > 0 with memory-mapped numpy arrays 
    can sometimes cause freezing. If training hangs, keep num_workers=0.
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