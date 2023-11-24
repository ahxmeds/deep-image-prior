#%%
import torch 
import torch.nn as nn 
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from torchsummary import summary 
# %%
class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolution, self).__init__()
        self.double_conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv_block(x)

class DownsamplingOperation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingOperation, self).__init__()
        self.downsampling_op = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        return self.downsampling_op(x)

class UpsamplingOperation(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels):
        super(UpsamplingOperation, self).__init__()

        self.upsampling_op = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.upsampling_op(x)
    

class UNetCustom(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetCustom, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = DoubleConvolution(self.in_channels, 16)
        self.downsample1 = DownsamplingOperation(16, 16)
        self.encoder2 = DoubleConvolution(16, 32)
        self.downsample2 = DownsamplingOperation(32, 32)
        self.encoder3 = DoubleConvolution(32, 64)
        self.downsample3 = DownsamplingOperation(64, 64)
        self.bottleneck = DoubleConvolution(64, 128)
        self.upsample3 = UpsamplingOperation(2, 128, 64)
        self.decoder3 = DoubleConvolution(64, 64)
        self.upsample2 = UpsamplingOperation(2, 64, 32)
        self.decoder2 = DoubleConvolution(32, 32)
        self.upsample1 = UpsamplingOperation(2, 32, 16)
        self.decoder1 = DoubleConvolution(16, self.out_channels)

    # def forward(self, x):
    #     x1 = self.encoder1(x)
    #     x2 = self.downsample1(x1)
    #     x3 = self.encoder2(x2)
    #     x4 = self.downsample2(x3)
    #     x5 = self.encoder3(x4)
    #     x6 = self.downsample3(x5)
    #     x7 = self.bottleneck(x6)
    #     x8 = self.upsample3(x7) + x5
    #     x9 = self.decoder3(x8)
    #     x10 = self.upsample2(x9) + x3
    #     x11 = self.decoder2(x10)
    #     x12 = self.upsample1(x11) + x1 
    #     out = self.decoder1(x12)
    #     return out
    def forward(self, x):
        x1 = self.encoder1(x)
        x = self.downsample1(x1)
        x3 = self.encoder2(x)
        x = self.downsample2(x3)
        x5 = self.encoder3(x)
        x = self.downsample3(x5)
        x = self.bottleneck(x)
        x = self.upsample3(x) + x5
        x = self.decoder3(x)
        x = self.upsample2(x) + x3
        x = self.decoder2(x)
        x = self.upsample1(x) + x1 
        x = self.decoder1(x)
        return x
    

# %%
device = torch.device('cuda:0')
#%%
unet_custom = UNetCustom(1,1).to(device)
unet_custom.train()
#%%
input = torch.rand((1, 1, 256, 256, 256)).to(device)
from torch.cuda.amp import autocast
with autocast():
    with autocast():
        output = unet_custom(input)
    print(output.shape)
# torch.cuda.empty_cache()
# %%
unet_monai = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        ).to(device)
# from torch.cuda.amp import autocast
# with autocast():
input = torch.rand((1, 1, 256, 256, 256)).to(device)
output = unet_monai(input)
#%%
with autocast():
    with autocast():
        input = torch.rand((1, 1, 256, 256, 256)).to(device)
        output = unet_monai(input)
# print(output.shape)
# torch.cuda.empty_cache()
# %%
