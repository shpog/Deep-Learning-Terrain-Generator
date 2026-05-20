import torch
import torch.nn as nn

class TerrainGenerator(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3):
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
        """Helper to create a standard Generator block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)


class TerrainCritic(nn.Module):
    def __init__(self, img_channels=3):
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
        """Helper to create a Critic block using InstanceNorm to protect the gradient penalty"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True), # NO BatchNorm here!
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.model(x)


def initialize_weights(model):
    """
    Initializes weights according to a normal distribution.
    Helps prevent the GAN from getting stuck early in training.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)