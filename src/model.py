import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self, in_channels, latent_dim):
        """
        Adaptive Instance Normalization (AdaIN) module.

        Applies instance normalization to the input tensor and scales/shifts it based on
        style parameters linearly derived from a latent noise vector.

        :param in_channels: Number of channels in the input feature map.
        :type in_channels: int
        :param latent_dim: Dimension of the latent noise vector used for styling.
        :type latent_dim: int
        """
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(in_channels, affine=False)

        self.style_scale = nn.Linear(latent_dim, in_channels)
        self.style_shift = nn.Linear(latent_dim, in_channels)

    def forward(self, x, noise):
        """
        Forward pass for the AdaIN layer.

        :param x: The input feature map tensor.
        :type x: torch.Tensor
        :param noise: The latent noise tensor used to generate scale and shift parameters.
        :type noise: torch.Tensor
        :return: The styled and normalized feature map.
        :rtype: torch.Tensor
        """
        noise_flat = noise.view(noise.size(0), -1)

        scale = self.style_scale(noise_flat).unsqueeze(2).unsqueeze(3)
        shift = self.style_shift(noise_flat).unsqueeze(2).unsqueeze(3)

        return scale * self.instance_norm(x) + shift


class UpBlockAdaIN(nn.Module):
    def __init__(self, in_c, out_c, latent_dim, use_dropout=False):
        """
        Upsampling block incorporating Adaptive Instance Normalization (AdaIN).

        Performs bilinear upsampling, followed by a convolution, AdaIN conditioning,
        and a ReLU activation. Optionally applies dropout for regularization.

        :param in_c: Number of input channels.
        :type in_c: int
        :param out_c: Number of output channels.
        :type out_c: int
        :param latent_dim: Dimension of the latent noise vector used in the AdaIN layer.
        :type latent_dim: int
        :param use_dropout: If True, applies dropout with a probability of 0.3. Defaults to False.
        :type use_dropout: bool, optional
        """
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv = nn.Conv2d(
            in_c,
            out_c,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            bias=False,
        )
        self.adain = AdaIN(out_c, latent_dim)
        self.relu = nn.ReLU(inplace=True)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.3)

    def forward(self, x, noise):
        """
        Forward pass for the AdaIN-equipped upsampling block.

        Upsamples the input feature map, applies a convolution, and then conditions
        the features using the latent noise via Adaptive Instance Normalization.
        Follows up with a ReLU activation and optional dropout.

        :param x: The input feature map tensor from the previous layer.
        :type x: torch.Tensor
        :param noise: The latent noise tensor used for style conditioning.
        :type noise: torch.Tensor
        :return: The upsampled and styled feature map.
        :rtype: torch.Tensor
        """
        x = self.upsample(x)
        x = self.conv(x)
        x = self.adain(x, noise)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class TerrainGenerator(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3):
        """
        Conditional Generator network for detailed terrain generation.

        Utilizes a U-Net-like architecture with encoder and decoder blocks.
        The encoder processes a lower-resolution "macro guide," while the decoder
        uses AdaIN to condition the high-resolution generation on a latent noise vector.

        :param latent_dim: Dimension of the latent noise vector. Defaults to 128.
        :type latent_dim: int, optional
        :param img_channels: Number of output image channels (e.g., elevation, precipitation, temperature). Defaults to 3.
        :type img_channels: int, optional
        """
        super().__init__()
        self.latent_dim = latent_dim

        self.enc1 = self._conv_block(img_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)
        self.enc5 = self._conv_block(256, 512)

        self.bottleneck = self._conv_block(512, 512, use_dropout=True)

        self.dec0 = self._up_block(512, 512, use_dropout=True)
        self.dec1 = self._up_block(512 + 512, 256, use_dropout=True)
        self.dec2 = self._up_block(256 + 256, 128)
        self.dec3 = self._up_block(128 + 128, 64)
        self.dec4 = self._up_block(64 + 64, 32)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                32 + 32,
                img_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.Tanh(),
        )

    def _conv_block(self, in_c, out_c, use_dropout=False):
        """
        Creates a convolutional encoder block consisting of a Conv2d layer, Instance
        Normalization, and a LeakyReLU activation. Optionally includes dropout.

        :param in_c: Number of input channels.
        :type in_c: int
        :param out_c: Number of output channels.
        :type out_c: int
        :param use_dropout: If True, appends a Dropout layer with a 0.3 probability. Defaults to False.
        :type use_dropout: bool, optional
        :return: A sequential block of encoding layers.
        :rtype: torch.nn.Sequential
        """
        layers = [
            nn.Conv2d(
                in_c,
                out_c,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.3))
        return nn.Sequential(*layers)

    def _up_block(self, in_c, out_c, use_dropout=False):
        """
        Creates a customized upsampling block utilizing the UpBlockAdaIN module.

        :param in_c: Number of input channels.
        :type in_c: int
        :param out_c: Number of output channels.
        :type out_c: int
        :param use_dropout: If True, applies dropout within the AdaIN block. Defaults to False.
        :type use_dropout: bool, optional
        :return: An initialized UpBlockAdaIN module configured with the generator's latent dimension.
        :rtype: UpBlockAdaIN
        """
        return UpBlockAdaIN(in_c, out_c, self.latent_dim, use_dropout)

    def forward(self, noise, macro_guide):
        """
        Forward pass of the TerrainGenerator.

        :param noise: Latent noise tensor, typically of shape (Batch, latent_dim, 1, 1).
        :type noise: torch.Tensor
        :param macro_guide: Low-resolution macro terrain tensor used to condition the spatial structure.
        :type macro_guide: torch.Tensor
        :return: Generated high-resolution terrain tensor normalized to [-1, 1].
        :rtype: torch.Tensor
        """
        e1 = self.enc1(macro_guide)
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
    def __init__(self, img_channels=3):
        """
        Critic (Discriminator) network for the conditional Wasserstein GAN.

        Evaluates the realism of a high-resolution terrain patch conditioned strictly
        on its corresponding macro guide by concatenating them along the channel dimension.

        :param img_channels: Number of image channels for the terrain patches. Defaults to 3.
        :type img_channels: int, optional
        """
        super(TerrainCritic, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                img_channels * 2,
                16,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(16, 32, 4, 2, 1),
            self._block(32, 64, 4, 2, 1),
            self._block(64, 128, 4, 2, 1),
            self._block(128, 256, 4, 2, 1),
            self._block(256, 512, 4, 2, 1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Constructs a fundamental building block for the critic network, consisting of a
        Conv2d layer, Instance Normalization, and a LeakyReLU activation.

        :param in_channels: Number of input channels.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param kernel_size: Size of the convolving kernel.
        :type kernel_size: int
        :param stride: Stride of the convolution.
        :type stride: int
        :param padding: Zero-padding added to both sides of the input.
        :type padding: int
        :return: A sequential block for feature extraction and downsampling.
        :rtype: torch.nn.Sequential
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                padding_mode="reflect",
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, image, macro_guide):
        """
        Forward pass of the TerrainCritic.

        :param image: The high-resolution terrain patch (real or fake).
        :type image: torch.Tensor
        :param macro_guide: The corresponding low-resolution macro guide.
        :type macro_guide: torch.Tensor
        :return: A 1D tensor representing the critic's validity score for the input patch.
        :rtype: torch.Tensor
        """
        x = torch.cat([image, macro_guide], dim=1)
        return self.model(x)


def initialize_weights(model):
    """
    Initializes the weights of a neural network module.

    Applies a normal distribution initialization to Convolutional and Linear layers
    (mean=0.0, std=0.02), and sets biases to zero. BatchNorm and InstanceNorm layers
    are initialized with a mean of 1.0 and standard deviation of 0.02.

    :param model: The PyTorch neural network module to initialize.
    :type model: torch.nn.Module
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)


class MacroGenerator(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3):
        """
        Unconditional Generator network for macro-level continent generation.

        Maps a latent noise vector to a coarse, low-resolution terrain map (macro guide)
        using a series of transposed convolutions, batch normalization, and upsampling blocks.

        :param latent_dim: Dimension of the latent noise vector. Defaults to 128.
        :type latent_dim: int, optional
        :param img_channels: Number of output image channels. Defaults to 3.
        :type img_channels: int, optional
        """
        super().__init__()

        self.start = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.block1 = self._upsample_block(512, 256)
        self.block2 = self._upsample_block(256, 128)
        self.block3 = self._upsample_block(128, 64)
        self.block4 = self._upsample_block(64, 32)
        self.block5 = self._upsample_block(32, 16)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def _upsample_block(self, in_c, out_c):
        """
        Creates an upsampling block for the macro generator, consisting of bilinear
        upsampling, a Conv2d layer, Batch Normalization, and a ReLU activation.

        :param in_c: Number of input channels.
        :type in_c: int
        :param out_c: Number of output channels.
        :type out_c: int
        :return: A sequential upsampling block.
        :rtype: torch.nn.Sequential
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, noise):
        """
        Forward pass of the MacroGenerator.

        :param noise: Latent noise tensor, typically of shape (Batch, latent_dim, 1, 1).
        :type noise: torch.Tensor
        :return: Generated low-resolution macro terrain map normalized to [-1, 1].
        :rtype: torch.Tensor
        """
        x = self.start(noise)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.final(x)


class MacroCritic(nn.Module):
    def __init__(self, img_channels=3):
        """
        Critic (Discriminator) network for the unconditional macro Wasserstein GAN.

        Evaluates the realism of the generated low-resolution, macro-level terrain maps.

        :param img_channels: Number of input image channels. Defaults to 3.
        :type img_channels: int, optional
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(16, 32, 4, 2, 1),
            self._block(32, 64, 4, 2, 1),
            self._block(64, 128, 4, 2, 1),
            self._block(128, 256, 4, 2, 1),
            self._block(256, 512, 4, 2, 1),
            nn.Conv2d(512, 1, 4, 1, 0),
        )

    def _block(self, in_c, out_c, kernel, stride, padding):
        """
        Creates a discriminator block for the macro critic, consisting of a Conv2d layer,
        Instance Normalization, and a LeakyReLU activation.

        :param in_c: Number of input channels.
        :type in_c: int
        :param out_c: Number of output channels.
        :type out_c: int
        :param kernel: Size of the convolving kernel.
        :type kernel: int
        :param stride: Stride of the convolution.
        :type stride: int
        :param padding: Zero-padding added to both sides of the input.
        :type padding: int
        :return: A sequential discriminator block.
        :rtype: torch.nn.Sequential
        """
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, img):
        """
        Forward pass of the MacroCritic.

        :param img: The low-resolution macro terrain map (real or fake).
        :type img: torch.Tensor
        :return: A 1D tensor representing the critic's validity score for the input map.
        :rtype: torch.Tensor
        """
        return self.net(img).view(-1)
