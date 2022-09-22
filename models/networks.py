import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce



class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2



class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*[ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)])

    def forward(self, x):
        return self.model(x)



class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x



class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)



class ContextUnet_2lvls(nn.Module):
    def __init__(self, in_channels, n_feat = 256):
        super(ContextUnet_2lvls, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        # self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t):
        # x is (noisy) image, c is context label, t is timestep, 

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2) #converts channels to vector with average pooling
        
        # embed context, time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out



class Unet_2lvls(nn.Module):
    def __init__(self, in_channels, n_feat = 256):
        super(Unet_2lvls, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x):
        # x is (noisy) image, c is context label, t is timestep, 
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2) #converts channels to vector with average pooling

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1, down2)
        up3 = self.up2(up2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out



class ContextUnet_3lvls(nn.Module):
    def __init__(self, in_channels, n_feat = 256):
        super(ContextUnet_3lvls, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 3 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 3*n_feat)
        self.timeembed2 = EmbedFC(1, 2*n_feat)
        self.timeembed3 = EmbedFC(1, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(3 * n_feat, 3 * n_feat, 8, 8), # otherwise just have 2*n_feat
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
        # x is (noisy) image, c is context label, t is timestep, 
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_vec(down3) #converts channels to vector with average pooling
        
        # embed context, time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 3, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 2, 1, 1)
        temb3 = self.timeembed3(t).view(-1, self.n_feat    , 1, 1)
        

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1+ temb1, down3)  # add and multiply embeddings
        up3 = self.up2(up2+ temb2, down2)
        up4 = self.up3(up3+ temb3, down1)
        out = self.out(torch.cat((up4, x), 1))
        return out



class Unet_3lvls(nn.Module):
    def __init__(self, in_channels, n_feat = 256):
        super(Unet_3lvls, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 3 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(3 * n_feat, 3 * n_feat, 8, 8), # otherwise just have 2*n_feat
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

    def forward(self, x):
        # x is (noisy) image 
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_vec(down3) #converts channels to vector with average pooling

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1, down3)
        up3 = self.up2(up2, down2)
        up4 = self.up3(up3, down1)
        out = self.out(torch.cat((up4, x), 1))
        return out



class Discriminator_TO(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator_TO, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def discriminator_block_even(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            *discriminator_block_even(512, 128),
            *discriminator_block_even(128, 64),
            *discriminator_block_even(64, 16)
        )

        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.linear_1 = nn.Linear(400, 50)
        self.linear_2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.leaky(self.linear_1(x))
        return self.sigmoid(self.linear_2(x))



class Discriminator_Micro(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator_Micro, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def discriminator_block_even(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, padding=1)]
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



class Discriminator_WGAN_TO(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator_WGAN_TO, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def discriminator_block_even(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            *discriminator_block_even(512, 128),
            *discriminator_block_even(128, 64),
            *discriminator_block_even(64, 16)
        )

        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.linear_1 = nn.Linear(400, 50)
        self.linear_2 = nn.Linear(50, 1)


    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.leaky(self.linear_1(x))
        return self.linear_2(x)



class Discriminator_WGAN_Micro(nn.Module):

    def __init__(self, in_channels=1):
        super(Discriminator_WGAN_Micro, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def discriminator_block_even(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, padding=1)]
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



class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=3, encoder_type='convolutional'):
        super(Encoder, self).__init__()
        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim *= 2

        self.model_blocks = nn.Sequential(*layers, nn.Tanh())

    def forward(self, x):
        x = self.model_blocks(x)
        return x



class Decoder(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_upsample=3, encoder_type='convolutional', activation='sigmoid'):
        super(Decoder, self).__init__()

        layers = []
        dim = dim * 2 ** n_upsample


        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7)]

        self.model_blocks = nn.Sequential(*layers)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.model_blocks(x))
        return x



class AE(nn.Module):
    """docstring for AE"""
    def __init__(self, in_channels, out_channels, dims=64, n_downsample=3):
        super(AE, self).__init__()
        self.encoder = Encoder(in_channels, dim=dims, n_downsample=n_downsample, encoder_type='regular')
        self.decoder = Decoder(out_channels, dim=dims, n_upsample=n_downsample, activation='sigmoid')

    def forward(self, x):
        code = self.encoder(x)
        out = self.decoder(code)
        return out



class Generator_Normal(nn.Module):
    def __init__(self, out_channels=1, dim=16, steps=5):
        super(Generator_Normal, self).__init__()

        decoder_layers = []
        dim = dim * 2 ** steps

        # Upsampling
        for _ in range(steps):
            decoder_layers += [
                nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim = dim // 2

        # Output layer
        decoder_layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7)]

        self.decoder_blocks = nn.Sequential(*decoder_layers)

        self.latent_block = nn.Sequential(nn.Linear(128,1024), nn.LeakyReLU(), 
                                          nn.Linear(1024,8192), nn.LeakyReLU())

    def forward(self, x):
        x = self.latent_block(x)
        x = rearrange(x, 'b (c h w) -> b c h w', c=512, w=4, h=4)
        return self.decoder_blocks(x)



class JF_Net(nn.Module):
	def __init__(self):
		super(JF_Net, self).__init__()
		#in_channel, out_channel, kernel_size, stride, padding=0 (default)
		self.conv1 = nn.Conv2d(1, 16, 9, 1) 
		self.bn1 = nn.BatchNorm2d(16)

		self.conv2 = nn.Conv2d(16, 32, 9, 1)
		self.bn2 = nn.BatchNorm2d(32)

		self.dropout1 = nn.Dropout2d(0.3)

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
		h = self.dropout1(h)
		h = F.relu(self.fc4(h))
		h = h.squeeze() #squeezing to reduce dims from (64,1) to (64,) to match target
		output = h
		return output