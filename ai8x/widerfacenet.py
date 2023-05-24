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
from torch import nn

import ai8x

import matplotlib
import matplotlib.pyplot as plt

def conv_shape(x, k=1, p=0, s=1, d=1):
    """
    Calculates the size of the layer output based on the input size, kernel size, padding, stride and dilation
    """
    return int((x + 2*p - d*(k - 1) - 1)/s + 1)

class WIDERFaceNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=2, dimensions = (128,128), bias=False, **kwargs):
        super().__init__()
        # Set dimensions for calculation of linear layer width
        dim_x, dim_y = dimensions
 
        self.conv1 = ai8x.FusedMaxPoolConv2dReLU(num_channels, 64, kernel_size=3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        dim_x = conv_shape(x=dim_x, k=3, p=1, s=2, d=1)
        dim_y = conv_shape(x=dim_y, k=3, p=1, s=2, d=1)
        print('L1: Dim_x:',dim_x,'Dim_y:', dim_y) 

        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(64, 32, kernel_size=3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        dim_x = conv_shape(dim_x, k=3, p=1, s=2, d=1)
        dim_y = conv_shape(dim_y, k=3, p=1, s=2, d=1)
        print('L2: Dim_x:',dim_x,'Dim_y:', dim_y) 

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(32, 16, kernel_size=3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        dim_x = conv_shape(dim_x, k=3, p=1, s=2, d=1)
        dim_y = conv_shape(dim_y, k=3, p=1, s=2, d=1)
        print('L3: Dim_x:',dim_x,'Dim_y:', dim_y) 

        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(16, 8, kernel_size=3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        dim_x = conv_shape(dim_x, k=3, p=1, s=2, d=1)
        dim_y = conv_shape(dim_y, k=3, p=1, s=2, d=1)
        print('L4: Dim_x:',dim_x,'Dim_y:', dim_y) 

        self.fc1 = ai8x.FusedLinearReLU(8 *dim_x*dim_y, 32)

        self.fc2 = ai8x.Linear(32, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = self.fc2(x)

        return x

def widerfacenet(pretrained=False, **kwargs):
    """
    Constructs a WIDERFaceNet model.
    """
    assert not pretrained
    return WIDERFaceNet(**kwargs)

"""
Network description
"""
models = [
    {
        'name': 'widerfacenet',
        'min_input': 1,
        'dim': 3,
    }
]
