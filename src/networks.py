import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_skip: bool = False) -> None:
        super().__init__()
        self.use_shortcut = use_skip or (in_channels != out_channels)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        if self.use_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)  # adjust channels
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv_block(x)
        return (x + shortcut) / 1.414

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        return self.model(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class Unet(nn.Module):
    def __init__(self, in_channels, n_feat=256):
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, use_skip=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 3 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 3 * n_feat)
        self.timeembed2 = EmbedFC(1, 2 * n_feat)
        self.timeembed3 = EmbedFC(1, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(3 * n_feat, 3 * n_feat, 8, 8),
            nn.GroupNorm(8, 3 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(6 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t):
        # x is (noisy) image, t is timestep, 
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_vec(down3) #converts channels to vector with average pooling
        
        # embed context, time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 3, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 2, 1, 1)
        temb3 = self.timeembed3(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1 + temb1, down3)  # add and multiply embeddings
        up3 = self.up2(up2 + temb2, down2)
        up4 = self.up3(up3 + temb3, down1)
        out = self.out(torch.cat((up4, x), 1))
        return out


class Discriminator_Micro(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator_Micro, self).__init__()

        def discriminator_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def discriminator_block_even(in_filters, out_filters, kernel_size=3, stride=1, padding=1, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            *discriminator_block_even(512, 512),
            *discriminator_block_even(512, 64),
            *discriminator_block_even(64, 32)
        )

        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.linear_1 = nn.Linear(2592, 1024)
        self.linear_2 = nn.Linear(1024, 256)
        self.linear_3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.leaky(self.linear_1(x))
        x = self.leaky(self.linear_2(x))
        return self.sigmoid(self.linear_3(x))


class Discriminator_WGAN_Micro(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator_WGAN_Micro, self).__init__()

        def discriminator_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def discriminator_block_even(in_filters, out_filters, kernel_size=3, stride=1, padding=1, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            *discriminator_block_even(512, 512),
            *discriminator_block_even(512, 64),
            *discriminator_block_even(64, 32)
        )

        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.linear_1 = nn.Linear(2592, 1024)
        self.linear_2 = nn.Linear(1024, 256)
        self.linear_3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.leaky(self.linear_1(x))
        x = self.leaky(self.linear_2(x))
        return self.linear_3(x)

class Generator(nn.Module):
    def __init__(self, out_channels=3, latent_dim=128, base_dim=16, steps=5, activation='sigmoid'):
        """
        Unified Generator that combines the latent projection and upsampling parts.

        Args:
            out_channels (int): Number of output channels (default 3, e.g. for RGB images).
            latent_dim (int): Dimensionality of the latent input vector (default 128).
            base_dim (int): Base number of channels before upsampling (default 16).
            steps (int): Number of upsampling blocks (default 5). 
                         The initial number of channels will be base_dim * (2**steps).
            activation (str): Final activation function to use. 'sigmoid' applies a Sigmoid,
                              'relu' applies a ReLU, or any other value applies no activation.
        """
        super(Generator, self).__init__()
        # Compute the initial feature map depth after the latent projection.
        self.hidden_dim = base_dim * (2 ** steps)  # e.g. 16 * 32 = 512 for base_dim=16, steps=5

        # Latent block: projects latent vector into a flattened feature map.
        self.latent_block = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.hidden_dim * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder / Upsampling block: gradually upsample the seed feature map.
        layers = []
        dim = self.hidden_dim
        for _ in range(steps):
            layers.append(nn.ConvTranspose2d(dim, dim // 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(dim // 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            dim //= 2
        # Final output layer with reflection padding to preserve border information.
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(dim, out_channels, kernel_size=7))
        self.decoder_blocks = nn.Sequential(*layers)

        # Optional activation function on output.
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None

    def forward(self, x):
        """
        Forward pass:
          1. x is expected to be of shape [batch, latent_dim].
          2. The latent block reshapes it into a feature map of shape [batch, hidden_dim, 4, 4].
          3. The decoder blocks upsample it to produce the final output.
        """
        x = self.latent_block(x)
        x = rearrange(x, "b (c h w) -> b c h w", c=self.hidden_dim, h=4, w=4)
        x = self.decoder_blocks(x)
        if self.activation is not None:
            x = self.activation(x)
        return x



class JF_Net(nn.Module):
	def __init__(self):
		super(JF_Net, self).__init__()
		#in_channel, out_channel, kernel_size, stride, padding=0 (default)
		self.conv1 = nn.Conv2d(1, 16, 9, 1) 
		self.bn1 = nn.BatchNorm2d(16)

		self.conv2 = nn.Conv2d(16, 32, 9, 1)
		self.bn2 = nn.BatchNorm2d(32)

		self.fc1 = nn.Linear(21632, 1024)
		self.fc4 = nn.Linear(1024, 2)

	def forward(self, x):

		h = F.relu(self.conv1(x))
		h = F.max_pool2d(h, kernel_size=2)
		h = self.bn1(h)

		h = F.relu(self.conv2(h))
		h = F.max_pool2d(h, kernel_size=2)
		h = self.bn2(h)

		h = torch.flatten(h, start_dim=1) #flatten input of [bs, c, w, h], so from dim=1
		h = F.relu(self.fc1(h))
		h = F.relu(self.fc4(h))
		h = h.squeeze() #squeezing to reduce dims from (64,1) to (64,) to match target
		output = h
		return output


if __name__ == '__main__':
    # Test Unet
    # Using a smaller base feature (n_feat) for quicker debugging.
    unet_net = Unet(in_channels=1, n_feat=32)
    dummy_image = torch.randn(2, 1, 128, 128)  # image input
    dummy_t = torch.randn(2, 1)              # time step input
    unet_out = unet_net(dummy_image, dummy_t)
    print("Unet output shape:", unet_out.shape)

    # Test Discriminator_Micro
    # Note: for the discriminator, we use an image size of 128x128 so that the flattening size matches 2592.
    disc_micro = Discriminator_Micro(in_channels=1)
    dummy_disc = torch.randn(2, 1, 128, 128)
    disc_micro_out = disc_micro(dummy_disc)
    print("Discriminator_Micro output shape:", disc_micro_out.shape)

    # Test Discriminator_WGAN_Micro
    disc_wgan = Discriminator_WGAN_Micro(in_channels=1)
    dummy_wgan = torch.randn(2, 1, 128, 128)
    disc_wgan_out = disc_wgan(dummy_wgan)
    print("Discriminator_WGAN_Micro output shape:", disc_wgan_out.shape)

    # Test Generator (Unified)
    # The unified Generator projects a latent vector (of size 128 by default) and upsamples it.
    gen = Generator(out_channels=3, latent_dim=128, base_dim=16, steps=5, activation='sigmoid')
    dummy_latent = torch.randn(2, 128)
    gen_out = gen(dummy_latent)
    print("Generator output shape:", gen_out.shape)

    # Test JF_Net
    jf_net = JF_Net()
    dummy_jf = torch.randn(2, 1, 128, 128)
    jf_out = jf_net(dummy_jf)
    print("JF_Net output shape:", jf_out.shape)
