import torch
import torch.nn as nn

class TerrainGenerator(nn.Module):
    """
    Conditional Generator network for the cWGAN-GP.
    
    Acts as an Encoder-Decoder. It encodes a 4-channel input (3-channel condition 
    + 1-channel mask) down to a feature vector, concatenates it with latent noise, 
    and decodes it back to a 256x256 terrain map.
    """
    def __init__(self, latent_dim=128, img_channels=3):
        super(TerrainGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        # ENCODER: Input [Batch, 4, 256, 256] -> Output [Batch, 512, 1, 1]
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels + 1, 32, kernel_size=4, stride=2, padding=1, bias=False), # 128x128
            nn.LeakyReLU(0.2, inplace=True),
            self._conv_block(32, 64, 4, 2, 1),   # 64x64
            self._conv_block(64, 128, 4, 2, 1),  # 32x32
            self._conv_block(128, 256, 4, 2, 1), # 16x16
            self._conv_block(256, 512, 4, 2, 1), # 8x8
            self._conv_block(512, 512, 4, 2, 1), # 4x4
            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=0, bias=False), # 1x1
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # DECODER: Input [Batch, 512 + latent_dim, 1, 1] -> Output [Batch, 3, 256, 256]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512 + latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False), # 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            self._block(512, 256, 3, 1, 1), # 8x8
            self._block(256, 128, 3, 1, 1), # 16x16
            self._block(128, 64, 3, 1, 1),  # 32x32
            self._block(64, 32, 3, 1, 1),   # 64x64
            self._block(32, 16, 3, 1, 1),   # 128x128
            
            # Final Layer: 128x128 -> 256x256
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # Your original upsampling block
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, noise, condition, mask):
        x = torch.cat([condition, mask], dim=1) 
        features = self.encoder(x)
        features = torch.nn.functional.dropout(features, p=0.5, training=self.training)
        bottleneck = torch.cat([features, noise], dim=1)
        return self.decoder(bottleneck)


class TerrainCritic(nn.Module):
    """
    Conditional Critic network for the cWGAN-GP.
    """
    def __init__(self, img_channels=3):
        super(TerrainCritic, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(img_channels + 4, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            self._block(16, 32, 4, 2, 1),
            self._block(32, 64, 4, 2, 1),
            self._block(64, 128, 4, 2, 1),
            self._block(128, 256, 4, 2, 1),
            self._block(256, 512, 4, 2, 1),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0) 
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, image, condition, mask):
        x = torch.cat([image, condition, mask], dim=1)
        return self.model(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)