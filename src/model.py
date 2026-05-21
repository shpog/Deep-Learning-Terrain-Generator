"""
Neural Network Architectures for Terrain Generation.

This module defines the PyTorch modules for the Wasserstein GAN with 
Gradient Penalty (WGAN-GP). It includes the Generator (which upsamples 
noise into terrain maps) and the Critic (which scores the realism of maps).
"""

import torch
import torch.nn as nn

class TerrainGenerator(nn.Module):
    """
    The Generator network for the WGAN-GP.

    Takes a latent noise vector and progressively upsamples it through 
    transposed convolutional layers to output a 3-channel image 
    (Elevation, Precipitation, Temperature) normalized between [0, 1].

    Attributes:
        latent_dim (int): The size of the input noise vector.
        model (torch.nn.Sequential): The sequential block of neural network layers.
    """
    def __init__(self, latent_dim=128, img_channels=3):
        """
        Initializes the TerrainGenerator.

        Args:
            latent_dim (int, optional): The dimensionality of the input noise vector. 
                Defaults to 128.
            img_channels (int, optional): The number of output channels (e.g., 3). 
                Defaults to 3.
        """
        super(TerrainGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        self.model = nn.Sequential(
            # Input: [Batch, latent_dim, 1, 1] -> Output: [Batch, 512, 4, 4]
            self._block(latent_dim, 512, 4, 1, 0),
            
            # Input: 4x4 -> Output: 8x8
            self._block(512, 256, 4, 2, 1),
            
            # Input: 8x8 -> Output: 16x16
            self._block(256, 128, 4, 2, 1),
            
            # Input: 16x16 -> Output: 32x32
            self._block(128, 64, 4, 2, 1),
            
            # Input: 32x32 -> Output: 64x64
            self._block(64, 32, 4, 2, 1),
            
            # Input: 64x64 -> Output: 128x128
            self._block(32, 16, 4, 2, 1),
            
            # Final Layer: 128x128 -> 256x256 (3 channels)
            nn.ConvTranspose2d(16, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Forces output to [0.0, 1.0] to match our preprocessed dataset
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Creates a standard Generator building block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (int): Zero-padding added to both sides of the input.

        Returns:
            torch.nn.Sequential: A block of ConvTranspose2d, BatchNorm2d, and ReLU.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Executes the forward pass of the Generator.

        Args:
            x (torch.Tensor): A noise tensor of shape [Batch, latent_dim, 1, 1].

        Returns:
            torch.Tensor: Generated terrain image of shape [Batch, img_channels, 256, 256].
        """
        return self.model(x)


class TerrainCritic(nn.Module):
    """
    The Critic (Discriminator) network for the WGAN-GP.

    Unlike a standard GAN, the Critic outputs an unbounded real number 
    representing the Wasserstein distance, rather than a probability between 0 and 1.
    It uses InstanceNorm2d instead of BatchNorm2d to remain compatible 
    with the gradient penalty.

    Attributes:
        model (torch.nn.Sequential): The sequential block of neural network layers.
    """
    def __init__(self, img_channels=3):
        """
        Initializes the TerrainCritic.

        Args:
            img_channels (int, optional): The number of input channels (e.g., 3). 
                Defaults to 3.
        """
        super(TerrainCritic, self).__init__()
        
        self.model = nn.Sequential(
            # Input: [Batch, 3, 256, 256] -> Output: [Batch, 16, 128, 128]
            nn.Conv2d(img_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: 128x128 -> Output: 64x64
            self._block(16, 32, 4, 2, 1),
            
            # Input: 64x64 -> Output: 32x32
            self._block(32, 64, 4, 2, 1),
            
            # Input: 32x32 -> Output: 16x16
            self._block(64, 128, 4, 2, 1),
            
            # Input: 16x16 -> Output: 8x8
            self._block(128, 256, 4, 2, 1),
            
            # Input: 8x8 -> Output: 4x4
            self._block(256, 512, 4, 2, 1),
            
            # Final Layer: 4x4 -> 1x1 scalar
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0) 
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Creates a standard Critic building block safely using InstanceNorm.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (int): Zero-padding added to both sides of the input.

        Returns:
            torch.nn.Sequential: A block of Conv2d, InstanceNorm2d, and LeakyReLU.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True), # NO BatchNorm here!
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        """
        Executes the forward pass of the Critic.

        Args:
            x (torch.Tensor): Image tensor of shape [Batch, img_channels, 256, 256].

        Returns:
            torch.Tensor: A raw scalar score of shape [Batch, 1, 1, 1].
        """
        return self.model(x)


def initialize_weights(model):
    """
    Initializes the network weights according to a normal distribution.

    Args:
        model (torch.nn.Module): The PyTorch neural network module to initialize.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)