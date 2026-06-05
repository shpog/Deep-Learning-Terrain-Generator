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
        super().__init__()
        self.latent_dim = latent_dim
        
        self.enc1 = self._conv_block(img_channels + 1, 32)   # -> 128x128
        self.enc2 = self._conv_block(32, 64)                 # -> 64x64
        self.enc3 = self._conv_block(64, 128)                # -> 32x32
        self.enc4 = self._conv_block(128, 256)               # -> 16x16
        self.enc5 = self._conv_block(256, 512)               # -> 8x8
        
        self.bottleneck = self._conv_block(512, 512)         
        
        self.dec0 = self._up_block(512 + latent_dim, 512)    # -> 8x8
        self.dec1 = self._up_block(512 + 512 + latent_dim, 256)           # -> 16x16
        self.dec2 = self._up_block(256 + 256 + latent_dim, 128)           # -> 32x32
        self.dec3 = self._up_block(128 + 128, 64)            # -> 64x64
        self.dec4 = self._up_block(64 + 64, 32)              # -> 128x128
        
        # WARSTWA KOŃCOWA
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32 + 32, img_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, noise, condition, mask):
        x = torch.cat([condition, mask], dim=1) 
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        b = self.bottleneck(e5)
        
        noise_b = noise.expand(-1, -1, b.shape[2], b.shape[3])
        d0 = self.dec0(torch.cat([b, noise_b], dim=1))
        
        noise_e5 = noise.expand(-1, -1, e5.shape[2], e5.shape[3])
        d1 = self.dec1(torch.cat([d0, e5, noise_e5], dim=1))
        
        noise_e4 = noise.expand(-1, -1, e4.shape[2], e4.shape[3])
        d2 = self.dec2(torch.cat([d1, e4, noise_e4], dim=1))
        
        d3 = self.dec3(torch.cat([d2, e3], dim=1))
        d4 = self.dec4(torch.cat([d3, e2], dim=1))
        
        out = self.final(torch.cat([d4, e1], dim=1))
        return out


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