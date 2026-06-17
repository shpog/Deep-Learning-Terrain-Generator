import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from config import PROCESSED_DATA_DIR, RESOLUTION, TILE_SIZE, EPOCHS


class TerrainDataset(Dataset):
    def __init__(self, epoch_size=10000, current_epoch=0):
        """
        A custom PyTorch Dataset for loading and pairing high-resolution terrain patches
        with their corresponding low-resolution macro guides.

        This dataset loads preprocessed, memory-mapped numpy arrays. It features dynamic
        filtering to reject overly flat or deep-ocean patches, with the rejection
        probability scaling dynamically based on the current training epoch.

        :param epoch_size: The number of patches to sample per epoch. Defaults to 10000.
        :type epoch_size: int, optional
        :param current_epoch: The current training epoch, used to calculate dynamic flatland rejection rates. Defaults to 0.
        :type current_epoch: int, optional
        """
        self.epoch_size = epoch_size
        self.current_epoch = current_epoch
        self.max_epochs = EPOCHS
        self.filepath = PROCESSED_DATA_DIR / f"worldclim_{RESOLUTION}_full.npy"

        temp_data = np.load(self.filepath, mmap_mode="r")
        self.channels, self.height, self.width = temp_data.shape
        del temp_data

        self.data = None

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        """
        Retrieves a random, augmented high-resolution terrain patch and its corresponding macro guide.

        Samples random coordinates to extract a tile, applies dynamic filtering based on
        elevation variance (delta) and mean, and performs random horizontal/vertical flips
        for data augmentation. It then creates a downsampled macro guide using bilinear
        and bicubic interpolation.

        :param idx: The index of the item (unused in implementation as sampling is random).
        :type idx: int
        :return: A tuple containing the high-resolution patch tensor and the downsampled macro guide tensor.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        if self.data is None:
            self.data = np.load(self.filepath, mmap_mode="r")

        MIN_DELTA = 0.45
        MIN_MEAN = -0.2

        flatland_keep_chance = 0.30 - 0.25 * (self.current_epoch / self.max_epochs)

        while True:
            y = np.random.randint(0, self.height - TILE_SIZE)
            x = np.random.randint(0, self.width - TILE_SIZE)

            patch_numpy = self.data[:, y : y + TILE_SIZE, x : x + TILE_SIZE]
            elevation = patch_numpy[0]

            delta = np.max(elevation) - np.min(elevation)
            mean_elev = np.mean(elevation)

            if mean_elev < -0.75:
                continue

            elif delta <= MIN_DELTA or mean_elev <= MIN_MEAN:
                if np.random.rand() < flatland_keep_chance:
                    break

            else:
                break

        if np.random.rand() > 0.5:
            patch_numpy = np.flip(patch_numpy, axis=1)
        if np.random.rand() > 0.5:
            patch_numpy = np.flip(patch_numpy, axis=2)

        patch_tensor = torch.from_numpy(np.array(patch_numpy).copy()).float()

        downscale_factor = 8
        macro_size = TILE_SIZE // downscale_factor

        patch_expanded = patch_tensor.unsqueeze(0)

        downsampled = F.interpolate(
            patch_expanded,
            size=(macro_size, macro_size),
            mode="bilinear",
            align_corners=False,
        )

        macro_tensor = F.interpolate(
            downsampled,
            size=(TILE_SIZE, TILE_SIZE),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        return patch_tensor, macro_tensor


def get_dataloader(batch_size=32, num_workers=0, current_epoch=0):
    """
    Instantiates and returns a PyTorch DataLoader for the TerrainDataset.

    :param batch_size: The number of samples per batch. Defaults to 32.
    :type batch_size: int, optional
    :param num_workers: The number of subprocesses to use for data loading. Defaults to 0.
    :type num_workers: int, optional
    :param current_epoch: The current training epoch to pass to the dataset for dynamic filtering. Defaults to 0.
    :type current_epoch: int, optional
    :return: A configured DataLoader yielding batches of (patch, macro_guide) pairs.
    :rtype: torch.utils.data.DataLoader
    """
    dataset = TerrainDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


class MacroDataset(Dataset):
    def __init__(self, epoch_size=5000):
        """
        A custom PyTorch Dataset for loading low-resolution macro terrain patches.

        This dataset is used for training the unconditional macro generator. It loads
        preprocessed 10m-resolution memory-mapped numpy arrays and extracts random patches,
        applying a basic variance filter to avoid completely flat areas.

        :param epoch_size: The number of patches to sample per epoch. Defaults to 5000.
        :type epoch_size: int, optional
        """
        self.epoch_size = epoch_size
        self.filepath = PROCESSED_DATA_DIR / "worldclim_10m_full.npy"

        temp_data = np.load(self.filepath, mmap_mode="r")
        self.channels, self.height, self.width = temp_data.shape
        del temp_data

        self.data = None

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        """
        Retrieves a random, augmented low-resolution macro terrain patch.

        Samples random coordinates, ensures the patch has at least a minimal elevation
        variance (or accepts it with a 10% random chance), applies random horizontal/vertical
        flips, and returns the converted tensor.

        :param idx: The index of the item (unused as sampling is random).
        :type idx: int
        :return: The augmented macro terrain patch tensor.
        :rtype: torch.Tensor
        """
        if self.data is None:
            self.data = np.load(self.filepath, mmap_mode="r")

        while True:
            y = np.random.randint(0, self.height - TILE_SIZE)
            x = np.random.randint(0, self.width - TILE_SIZE)

            patch_numpy = self.data[:, y : y + TILE_SIZE, x : x + TILE_SIZE]
            elevation = patch_numpy[0]

            delta = np.max(elevation) - np.min(elevation)

            if delta > 0.1 or np.random.rand() < 0.1:
                break

        if np.random.rand() > 0.5:
            patch_numpy = np.flip(patch_numpy, axis=1)
        if np.random.rand() > 0.5:
            patch_numpy = np.flip(patch_numpy, axis=2)

        patch_tensor = torch.from_numpy(np.array(patch_numpy).copy()).float()
        return patch_tensor


def get_macro_dataloader(batch_size=32, num_workers=0):
    """
    Instantiates and returns a PyTorch DataLoader for the MacroDataset.

    :param batch_size: The number of samples per batch. Defaults to 32.
    :type batch_size: int, optional
    :param num_workers: The number of subprocesses to use for data loading. Defaults to 0.
    :type num_workers: int, optional
    :return: A configured DataLoader yielding batches of macro terrain patches.
    :rtype: torch.utils.data.DataLoader
    """
    dataset = MacroDataset()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
