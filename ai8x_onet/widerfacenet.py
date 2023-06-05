###################################################################################################
# WIDER Faces Network
# Lionnus Kesting
# Machine Learning on Microcontrollers
# 2023 - ETH Zurich
###################################################################################################
"""
WIDERFaceNet network description
"""
from signal import pause
import torch
from torch import nn

import ai8x

import matplotlib
import matplotlib.pyplot as plt

## Function to calculate linear layer dimensions
def conv_shape(x, k=1, p=0, s=1, d=1):
    return int((x + 2*p - d*(k - 1) - 1)/s + 1)

class WIDERFaceONet(nn.Module):
    def __init__(self, num_channels=3, num_classes=2, dimensions = (48,48), bias=False, **kwargs):
        super().__init__()
        # +++++++++++++++++++++ layer 0:
        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 32, kernel_size=3, padding=1, bias=bias, **kwargs)
    
        dim_x = conv_shape(dimensions[0], k=3, p=1, s=1, d=1) # Conv2d
        # print("dDimensions after first layer: ", dim_x)

        # +++++++++++++++++++++ layer 1:
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(32,64, kernel_size=3, pool_size=3, pool_stride=2, padding=0, bias=bias, **kwargs)

        dim_x = conv_shape(dim_x, k=3, s=2) # Maxpool
        dim_x = conv_shape(dim_x, k=3, p=0, s=1, d=1) # Conv2d
        # print("dDimensions after second layer: ", dim_x)

        # +++++++++++++++++++++ layer 2:
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(64,64, kernel_size=3, pool_size=3, pool_stride=2, padding=0, bias=bias, **kwargs)

        dim_x = conv_shape(dim_x, k=3, s=2) # Maxpool
        dim_x = conv_shape(dim_x, k=3, p=0, s=1, d=1) # Conv2d
        # print("dDimensions after third layer: ", dim_x)
       
        # +++++++++++++++++++++ layer 3:
        # Conv2d with 3x3 kernel since 2x2 kernel is not supported by ai8x
        self.conv4 = ai8x.FusedMaxPoolConv2d(64,128, kernel_size=3, pool_size=2, pool_stride=2, padding=0, bias=bias, **kwargs)
       
        dim_x = conv_shape(dim_x, k=2, s=2) # Maxpool
        dim_x = conv_shape(dim_x, k=3, p=0, s=1, d=1) # Conv2d
        # print("dDimensions after fourth layer: ", dim_x)

        dim_y=dim_x #change when not square!
        # print("dDimensions for linear layer: ", 256*dim_x*dim_y)
        self.fc1 = ai8x.FusedLinearReLU(128*dim_x*dim_y, 256)

        self.fc2 = ai8x.Linear(256, 5, wide=True)

    def forward(self, x):
        # print("Dimensions before first layer: ", x.shape)
        x = self.conv1(x)
        # print("Dimensions after first layer: ", x.shape)
        x = self.conv2(x)
        # print("Dimensions after second layer: ", x.shape)
        x = self.conv3(x)
        # print("Dimensions after third layer: ", x.shape)
        x = self.conv4(x)
        # print("Dimensions after fourth layer: ", x.shape)
        # x = self.conv5(x)
        # print("Dimensions after fifth layer: ", x.shape)
        # x = self.conv6(x)
        # print("Dimensions after sixth layer: ", x.shape)
        x = x.view(x.size(0), -1)
        # print(self.num_flat_features(x))
        x = self.fc1(x)
      
        x = self.fc2(x)

        return x
    
class WIDERFaceRNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=2, dimensions = (24,24), bias=False, **kwargs):
        super().__init__()
        # +++++++++++++++++++++ layer 0:
        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 28, kernel_size=3, padding=1, bias=bias, **kwargs)
    
        dim_x = conv_shape(dimensions[0], k=3, p=1, s=1, d=1) # Conv2d
        # print("dDimensions after first layer: ", dim_x)

        # +++++++++++++++++++++ layer 1:
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(28, 48, kernel_size=3, pool_size=3, pool_stride=2, padding=0, bias=bias, **kwargs)

        dim_x = conv_shape(dim_x, k=3, s=2) # Maxpool
        dim_x = conv_shape(dim_x, k=3, p=0, s=1, d=1) # Conv2d
        # print("dDimensions after second layer: ", dim_x)

        # +++++++++++++++++++++ layer 2:
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(48, 64, kernel_size=3, pool_size=3, pool_stride=2, padding=0, bias=bias, **kwargs)

        dim_x = conv_shape(dim_x, k=3, s=2) # Maxpool
        dim_x = conv_shape(dim_x, k=3, p=0, s=1, d=1) # Conv2d
        # print("dDimensions after third layer: ", dim_x)

        dim_y=dim_x #change when not square!
        # print("dDimensions for linear layer: ", 256*dim_x*dim_y)
        self.fc1 = ai8x.FusedLinearReLU(128*dim_x*dim_y, 128)

        self.fc2 = ai8x.Linear(128, 5, wide=True)

    def forward(self, x):
        # print("Dimensions before first layer: ", x.shape)
        x = self.conv1(x)
        # print("Dimensions after first layer: ", x.shape)
        x = self.conv2(x)
        # print("Dimensions after second layer: ", x.shape)
        x = self.conv3(x)
        # print("Dimensions after third layer: ", x.shape)
        x = self.conv4(x)
        # print("Dimensions after fourth layer: ", x.shape)
        # x = self.conv5(x)
        # print("Dimensions after fifth layer: ", x.shape)
        # x = self.conv6(x)
        # print("Dimensions after sixth layer: ", x.shape)
        x = x.view(x.size(0), -1)
        # print(self.num_flat_features(x))
        x = self.fc1(x)
      
        x = self.fc2(x)

        return x

def widerfaceonet(pretrained=False, **kwargs):
    """
    Constructs a WIDERFaceONet model.
    """
    assert not pretrained
    return WIDERFaceONet(**kwargs)

def widerfacernet(pretrained=False, **kwargs):
    """
    Constructs a WIDERFaceRNet model.
    """
    assert not pretrained
    return WIDERFaceRNet(**kwargs)

"""
Network description
"""
models = [
    {
        'name': 'widerfaceonet',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'widerfacernet',
        'min_input': 1,
        'dim': 2,
    },
]
