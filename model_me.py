
import torch
import torch.nn as nn
import math
from PIL import Image
import numpy as np
class FSRCNN(nn.Module):
    def __init__(self, d, s, m, n):
        super(FSRCNN, self).__init__()
        self.d = d
        self.s = s
        self.m = m
        self.upscale = n
        self.alpha = 0.25
        self.feat_ext = nn.Conv2d(1, self.d, 5, 1, 2)
        self.act1 = nn.PReLU(self.d)
        self.shrink = nn.Conv2d(self.d, self.s, 1)
        self.act2 = nn.PReLU(self.s)
        self.map = nn.Sequential()
        for i in range(self.m):
            self.map.append(nn.Conv2d(self.s, self.s, 3, 1, 1))
            self.map.append(nn.PReLU(self.s))
        #self.act3 = nn.PReLU(self.s)
        self.expand = nn.Conv2d(self.s, self.d, 1)
        self.act4 = nn.PReLU(self.d)
        self.deconv = nn.ConvTranspose2d(self.d, 1, 9, self.upscale, 4, output_padding=self.upscale-1)
        self._weight_initialization()
    def _weight_initialization(self):
        nn.init.normal_(self.feat_ext.weight.data, mean=0.0, std=math.sqrt(2 / ((1+self.alpha**2)*self.feat_ext.out_channels*self.feat_ext.weight[0][0].numel())))
        nn.init.zeros_(self.feat_ext.bias.data)
        nn.init.normal_(self.shrink.weight.data, mean=0.0, std=math.sqrt(2 / ((1+self.alpha**2)*self.shrink.out_channels*self.shrink.weight[0][0].numel())))
        nn.init.zeros_(self.shrink.bias.data)
        for layer in self.map:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, mean=0.0, std=math.sqrt(2 / ((1+self.alpha**2)*layer.out_channels*layer.weight[0][0].numel())))
                nn.init.zeros_(layer.bias.data)
        nn.init.normal_(self.expand.weight.data, mean=0.0, std=math.sqrt(2 / ((1+self.alpha**2)*self.expand.out_channels*self.expand.weight[0][0].numel())))
        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)
    def forward(self, x):
        x = self.feat_ext(x)
        x = self.act1(x)
        x = self.shrink(x)
        x = self.act2(x)
        x = self.map(x)
        #x = self.act3(x)
        x = self.expand(x)
        x = self.act4(x)
        x = self.deconv(x)
        return x


