"""
Global Configuration and Hyperparameters.

This module acts as the single source of truth for all file paths, 
dataset resolutions, and neural network hyperparameters used across 
the Terrain Generator project. Modifying values here will automatically 
propagate through the preprocessing, dataset, and training pipelines.
"""

import os
from pathlib import Path

# PATH DEFINITIONS
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# DATA & IMAGE SETTINGS
RESOLUTION = "2.5m"
TILE_SIZE = 256 
IMG_CHANNELS = 3

# MODEL & TRAINING HYPERPARAMETERS
BATCH_SIZE = 8
EPOCHS = 200
LATENT_DIM = 128
LR_GEN = 1e-4
LR_CRITIC = 4e-4
BETA1 = 0.0 # Adam optimizer beta1
BETA2 = 0.9 # Adam optimizer beta2
CONDITION_WIDTH = 64

# WGAN-Specific Settings
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10.0 # Gradient penalty weight