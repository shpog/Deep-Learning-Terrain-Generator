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
RESOLUTION = "10m"
TILE_SIZE = 256 
IMG_CHANNELS = 3

# MODEL & TRAINING HYPERPARAMETERS
BATCH_SIZE = 64
EPOCHS = 200
LATENT_DIM = 128
LEARNING_RATE = 1e-4
BETA1 = 0.0 # Adam optimizer beta1
BETA2 = 0.9 # Adam optimizer beta2

# WGAN-Specific Settings
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10.0 # Gradient penalty weight