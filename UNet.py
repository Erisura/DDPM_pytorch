import torch
import torch.nn as nn


# (conv+ReLU)*2 with padding
class ConvBLock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels) -> None:
        super().__init__()
        self.conv_ReLU = nn.ModuleList([])
        self.conv_ReLU.append(nn.ModuleList([
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU()
        ]))
        self.conv_ReLU.append(nn.ModuleList([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        ]))
    def forward(self, x):
        for conv,act in self.conv_ReLU:
            x = conv(x)
            x = act(x)
        return x
    
class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        """define the left part of UNet"""
        self.left_conv_1 = ConvBLock(3, 64, 64)
        self.downsample_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = ConvBLock(64,128,128)
        self.downsample_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = ConvBLock(128,256,256)
        self.downsample_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_4 = ConvBLock(256,512,512)
        self.downsample_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_5 = ConvBLock(512,1024,1024)

        """define the right part of UNet"""
        # 左右序号对应
        self.upsample_4 = nn.ConvTranspose2d(1024,512,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.right_conv_4 = ConvBLock(1024,512,512)

        self.upsample_3 = nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.right_conv_3 = ConvBLock(512,256,256)

        self.upsample_2 = nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.right_conv_2 = ConvBLock(256,128,128)

        self.upsample_1 = nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.right_conv_1 = ConvBLock(128,64,64)

        self.right_conv_0 = nn.Conv2d(64,3,kernel_size=1,padding=0)

    def forward(self, x):
        """encode x"""
        feature_1_l = self.left_conv_1(x)
        feature_1_l_ds = self.downsample_1(feature_1_l)

        feature_2_l = self.left_conv_2(feature_1_l_ds)
        feature_2_l_ds = self.downsample_2(feature_2_l)

        feature_3_l = self.left_conv_3(feature_2_l_ds)
        feature_3_l_ds = self.downsample_3(feature_3_l)

        feature_4_l = self.left_conv_4(feature_3_l_ds)
        feature_4_l_ds = self.downsample_4(feature_4_l)

        feature_5 = self.left_conv_5(feature_4_l_ds)

        """decode x"""
        feature_4_r = self.upsample_4(feature_5)
        feature_4_r_us = self.right_conv_4(torch.cat((feature_4_l, feature_4_r),dim=1))

        feature_3_r = self.upsample_3(feature_4_r_us)
        feature_3_r_us = self.right_conv_3(torch.cat((feature_3_l, feature_3_r),dim=1))

        feature_2_r = self.upsample_2(feature_3_r_us)
        feature_2_r_us = self.right_conv_2(torch.cat((feature_2_l, feature_2_r),dim=1))

        feature_1_r = self.upsample_1(feature_2_r_us)
        feature_1_r_us = self.right_conv_1(torch.cat((feature_1_l, feature_1_r),dim=1))

        feature_0 = self.right_conv_0(feature_1_r_us)

        return feature_0
    
if __name__ == '__main__':
    
    x = torch.randn((5,3,64,64))
    net = UNet()
    x = net(x)
    print(x.shape)
        
