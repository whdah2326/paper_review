import torch

from torch import nn
from typing import Union
from collections import OrderedDict


shape_dict = dict()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, 
                      padding='same', padding_mode='reflect'),
            nn.InstanceNorm2d(self.out_channels),
            nn.RELU(inplace=True),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, 
                      padding='same', padding_mode='reflect'),
            nn.InstanceNorm2d(self.out_channels)
        )

    def forward(self, x):
        output = self.block(x) + x

        return output
    

class Generator(nn.Module):
    def __init__(self, init_channel:int, kernel_size:int, stride:int, n_blocks:int=6):
        super(Generator, self).__init__()

        self.init_channel=init_channel
        self.kernel_size=kernel_size
        self.stride=stride
        self.n_blocks=n_blocks

        layers = OrderedDict()
        layers['conv_first'] = self._make_block(in_channels=3, out_channels=self.init_channel,
                                                kernel_size=7, stride=1, padding='same')

        for i in range(2):
            ic = self.init_channel*(i+1)
            k = 2*ic
            layers[f'd_{k}'] = self.make_block(in_channels=ic, out_channels=k,
                                               kernel_size=self.kernel_size, stride=self.stride)

        for i in range(self.n_blocks):
            layers[f'R{k}_{i+1}'] = ResidualBlock(k, k)

        for i in range(2):
            k = int(k/2)
            layers[f'u_{k}'] = self._make_block(in_channels=k*2, out_channels=k,
                                                kernel_size=self.kernel_size, stride=self.stride, mode='u')
        
        layers['conv_last'] = nn.Conv2d(in_channels=self.init_channel, out_channels=3,
                                        kernel_size=7, stride=1, padding='same', padding_mode='reflect')
        layers['tanh'] = nn.Tanh()

        self.model = nn.Sequential(
            layers
        )

        def forward(self, x):
            op = self.model(x)
            assert op.shape == x.shape, f"output shape ({op.shape}) must be same with the input size ({x.shape})"
            return op
        
        def _make_block(self, in_channels:int, out_channels:int, kernel_size:int, stride:int,
                        padding:Union[int,str]=1, mode:str='d'):
            block = []
            if mode.lower() == 'd':
                block.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, padding_mode='reflect'))
            elif mode.lower() == 'u':
                block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                stride=stride, padding=padding, output_padding=1))

                block += [nn.InstanceNorm2d(out_channels), nn.ReLU(inplace=True)]

            return nn.Sequential(*block)
        
            
class Discriminator(nn.Module):
    def __init__(self, n_layers:int=4, input_c:int=3, n_filter:int=64, kernel_size:int=4):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential()
        self.kernel_size=kernel_size
        self.n_layers = n_layers
        layers = []

        for i in range(self.n_layers):
            if i == 0:
                ic, oc = input_c, n_filter
                layers.append(self._make_block(ic, oc, kernel_size=self.kernel_size, stride=2,
                                               padding=1, normalize=False))
            else:
                ic = oc
                oc = 2*ic
                stride=2

            