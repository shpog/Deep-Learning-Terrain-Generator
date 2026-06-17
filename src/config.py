import os
from pathlib import Path

"""
Configuration settings and hyperparameters for the Terrain Generator pipeline.

This module defines directory paths, dataset parameters, neural network 
hyperparameters, and WGAN-GP settings utilized across the entire project. 
It automatically creates necessary directories if they do not exist.

Paths:
    PROJECT_ROOT (pathlib.Path): Absolute path to the project root directory.
    RAW_DATA_DIR (pathlib.Path): Target directory for raw GeoTIFF datasets.
    PROCESSED_DATA_DIR (pathlib.Path): Target directory for processed numpy arrays.
    MODELS_DIR (pathlib.Path): Target directory for saving model checkpoints and final weights.

Dataset Parameters:
    RESOLUTION (str): Resolution identifier for the WorldClim dataset (e.g., "30s").
    TILE_SIZE (int): Spatial dimension (width and height) of the training patches.
    IMG_CHANNELS (int): Number of image channels (Elevation, Precipitation, Temperature).

Training & Model Hyperparameters:
    BATCH_SIZE (int): Number of samples per training batch.
    EPOCHS (int): Total number of full passes through the training dataset.
    LATENT_DIM (int): Dimensionality of the latent noise vector.
    LR_GEN (float): Learning rate for the Adam optimizer of the generator.
    LR_CRITIC (float): Learning rate for the Adam optimizer of the critic.
    BETA1 (float): Beta1 parameter for the Adam optimizers.
    BETA2 (float): Beta2 parameter for the Adam optimizers.
    CONDITION_WIDTH (int): Spatial dimension of the downscaled macro condition.
    LAMBDA_L1 (float): Weighting factor for the L1 pixel-wise loss penalty.
    LAMBDA_GRAD (int): Weighting factor for the spatial gradient alignment penalty.
    CRITIC_ITERATIONS (int): Number of critic training steps per generator step.
    LAMBDA_GP (float): Weighting factor for the WGAN-GP gradient penalty.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTION = "30s"
TILE_SIZE = 256
IMG_CHANNELS = 3

BATCH_SIZE = 32
EPOCHS = 200
LATENT_DIM = 128
LR_GEN = 1e-4
LR_CRITIC = 4e-4
BETA1 = 0.0
BETA2 = 0.9
CONDITION_WIDTH = 32
LAMBDA_L1 = 0.5
LAMBDA_GRAD = 1

CRITIC_ITERATIONS = 5
LAMBDA_GP = 10.0
