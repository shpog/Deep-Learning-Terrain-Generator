import torch
import torch.nn as nn

class AdaIN(nn.Module):
    """
    Moduł Adaptive Instance Normalization.
    Normalizuje cechy przestrzenne i moduluje je (skalowanie + przesunięcie)
    na podstawie wektora szumu (stylu).
    """
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(in_channels, affine=False)
        
        self.style_scale = nn.Linear(latent_dim, in_channels)
        self.style_shift = nn.Linear(latent_dim, in_channels)

    def forward(self, x, noise):
        noise_flat = noise.view(noise.size(0), -1)
        
        scale = self.style_scale(noise_flat).unsqueeze(2).unsqueeze(3)
        shift = self.style_shift(noise_flat).unsqueeze(2).unsqueeze(3)
        
        return scale * self.instance_norm(x) + shift


class UpBlockAdaIN(nn.Module):
    """
    Klasa pomocnicza zastępująca nn.Sequential w dekoderze.
    Pozwala na przekazanie mapy cech oraz szumu jednocześnie.
    """
    def __init__(self, in_c, out_c, latent_dim, use_dropout=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
        self.adain = AdaIN(out_c, latent_dim)
        self.relu = nn.ReLU(inplace=True)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.3)

    def forward(self, x, noise):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.adain(x, noise)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x

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
        
        self.bottleneck = self._conv_block(512, 512, use_dropout=True)    

        self.dec0 = self._up_block(512, 512, use_dropout=True)                # -> 8x8  
        self.dec1 = self._up_block(512 + 512, 256, use_dropout=True)          # -> 16x16
        self.dec2 = self._up_block(256 + 256, 128)                            # -> 32x32
        self.dec3 = self._up_block(128 + 128, 64)                             # -> 64x64
        self.dec4 = self._up_block(64 + 64, 32)                               # -> 128x128
        
        # WARSTWA KOŃCOWA
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32 + 32, img_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )

    def _conv_block(self, in_c, out_c, use_dropout=False):
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.3))
        return nn.Sequential(*layers)

    def _up_block(self, in_c, out_c, use_dropout=False):
        return UpBlockAdaIN(in_c, out_c, self.latent_dim, use_dropout)

    def forward(self, noise, condition, mask):
        x = torch.cat([condition, mask], dim=1) 
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        b = self.bottleneck(e5)
        
        d0 = self.dec0(b, noise)
        d1 = self.dec1(torch.cat([d0, e5], dim=1), noise)
        d2 = self.dec2(torch.cat([d1, e4], dim=1), noise)
        d3 = self.dec3(torch.cat([d2, e3], dim=1), noise)
        d4 = self.dec4(torch.cat([d3, e2], dim=1), noise)
        
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
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)