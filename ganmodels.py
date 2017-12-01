import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
counterD = 1
counterG = 1

def weightsInitNormal(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class layerLinear(nn.Module):
    def __init__(self, input_size, output_size, squeeze=False):
        super(layerLinear, self).__init__()
        self.Linear = nn.Linear(input_size, output_size)
        self.squeeze = squeeze

    def forward(self, x):
        # global counterD
        y = self.Linear( x ) if not self.squeeze else self.Linear( torch.squeeze(x) )
        # print('Discriminator-L #%s' % counterD, x.size(), '  --->  ', y.size())
        # counterD+=1
        return y

class layerConvolution(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
        super(layerConvolution, self).__init__()
        self.Conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.LRelu = nn.LeakyReLU(0.2)
        self.BatchNorm = nn.BatchNorm2d(output_size)
        self.isActivation = activation
        self.isNormingBatch = batch_norm

    def forward(self, x):
        # global counterD
        conved = self.Conv( x ) if not self.isNormingBatch else self.BatchNorm(self.Conv( x ))
        # print('Discriminator #%s' % counterD, x.size(), '  --->  ', conved.size())
        # counterD+=1
        return conved if not self.isActivation else self.LRelu( conved )

class layerDeconvolution(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=True, activation=True):
        super(layerDeconvolution, self).__init__()
        self.ConvTranspose = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.BatchNorm = nn.BatchNorm2d(output_size)
        self.isNormingBatch = batch_norm
        self.isActivation = activation

    def forward(self, x):
        # global counterG
        deconved = self.ConvTranspose( x ) if not self.isNormingBatch else self.BatchNorm(self.ConvTranspose( x ))
        # print('Generator #%s' % counterG, x.size(), '  --->  ', deconved.size())
        # counterG+=1
        return deconved if not self.isActivation else F.relu( deconved )



class modelGenerator(nn.Module):
    def __init__(self, noise_size, output_size, d=32):
        super(modelGenerator, self).__init__()
        self.net = nn.Sequential(
            layerDeconvolution(noise_size, d * 16, kernel_size=4, stride=1, padding=0),
            layerDeconvolution(d * 16, d * 8),
            layerDeconvolution(d * 8, d * 4),
            layerDeconvolution(d * 4, d * 2),
            layerDeconvolution(d * 2, d, stride=4, padding=0),
            layerDeconvolution(d, output_size, batch_norm=True, activation=False),
            nn.Tanh()
        )
    def weight_init(self, mean, std):
        for m in self._modules:
            weightsInitNormal(self._modules[m], mean, std)
    def forward(self, x):
        return self.net( x )

class modelDiscriminator(nn.Module):
    def __init__(self, input_size, output_size, d=32):
        super(modelDiscriminator, self).__init__()
        self.net = nn.Sequential(
            layerConvolution(input_size, d, batch_norm=False),
            layerConvolution(d, d * 2),
            layerConvolution(d * 2, d * 4),
            layerConvolution(d * 4, d * 8),
            layerConvolution(d * 8, d * 16),
            layerConvolution(d * 16, output_size, activation=False, batch_norm=False, kernel_size=8, stride=1, padding=0),
            nn.Sigmoid()
        )
    def weight_init(self, mean, std):
        for m in self._modules:
            weightsInitNormal(self._modules[m], mean, std)
    def forward(self, x):
        return self.net( x )
