import torch
import torch.nn as nn
import numpy as np
from kornia.filters import filter2d

from .custom_layers import EqualizedConv2d, EqualizedLinear,\
    NormalizationLayer, Upscale2d

def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

class AdaIN(nn.Module):
    def __init__(self, dimIn, dimOut, epsilon=1e-8):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
        self.styleModulator = nn.Linear(dimIn, 2*dimOut)
        self.dimOut = dimOut

        with torch.no_grad():
            self.styleModulator.weight *= 0.25
            self.styleModulator.bias.data.fill_(0)
        
    def forward(self, x, y):
        # x: N x C x W x H
        batchSize, nChannel, width, height = x.size()
        tmpX = x.view(batchSize, nChannel, -1)
        mux = tmpX.mean(dim=2).view(batchSize, nChannel, 1, 1)
        varx = torch.clamp((tmpX*tmpX).mean(dim=2).view(batchSize, nChannel, 1, 1) - mux*mux, min=0)
        varx = torch.rsqrt(varx + self.epsilon)
        x = (x - mux) * varx

        # Adapt style
        styleY = self.styleModulator(y)
        yA = styleY[:, : self.dimOut].view(batchSize, self.dimOut, 1, 1)
        yB = styleY[:, self.dimOut:].view(batchSize, self.dimOut, 1, 1)

        return yA * x + yB


class NoiseMultiplier(nn.Module):
    def __init__(self):
        super(NoiseMultiplier, self).__init__()
        self.module = nn.Conv2d(1, 1, 1, bias=False)
        self.module.weight.data.fill_(0)

    def forward(self, x):

        return self.module(x)

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)

class ToRGB(nn.Module):
    def __init__(self, in_channel, actvn=False):
        super().__init__()
        self.conv = EqualizedConv2d(in_channel,
                                    3,
                                    1,
                                    equalized=True,
                                    initBiasToZero=True)
        self.actvn = actvn

    def forward(self, x, skip=None):
        out = self.conv(x)

        if self.actvn:
            out = torch.sigmoid(out)

        return out
        
class MappingLayer(nn.Module):
    def __init__(self, dimIn, dimLatent, nLayers, leakyReluLeak=0.2):
        super(MappingLayer, self).__init__()
        self.FC = nn.ModuleList()

        inDim = dimIn
        for i in range(nLayers):
            self.FC.append(EqualizedLinear(inDim, dimLatent, lrMul=0.01, equalized=True, initBiasToZero=True))
            inDim = dimLatent

        self.activation = torch.nn.LeakyReLU(leakyReluLeak)

    def forward(self, x):
        for layer in self.FC:
            x = self.activation(layer(x))

        return x

class GNet(nn.Module):
    def __init__(self,
                dimInput: int=256,
                dimHidden: int=256,
                dimMapping: int=256,
                nMappingLayers: int=4,
                leakyReluLeak=0.2,
                generationActivation=None,
                phiTruncation=0.5,
                gamma_avg=0.99):

        super(GNet, self).__init__()
        self.epoch = 0
        self.step = 0
        
        self.dimMapping = dimMapping
        self.scaleLayers = nn.ModuleList()
        self.noiseModulators = nn.ModuleList()
        self.noramlizationLayer = NormalizationLayer()

        self.adain00 = AdaIN(dimMapping, dimInput)
        self.noiseMod00 = NoiseMultiplier()
        self.adain01 = AdaIN(dimMapping, dimHidden)
        self.noiseMod01 = NoiseMultiplier()


        self.conv0 = EqualizedConv2d(dimInput, dimHidden, 3, equalized=True,
                                     initBiasToZero=True, padding=1)
      
    
        self.activation = torch.nn.LeakyReLU(leakyReluLeak)

        self.depthScales = [dimHidden]
        
        self.toRGBLayers = nn.ModuleList()

        self.toRGBLayers.append(ToRGB(dimHidden, actvn=False))

        self.blur = Blur()
        self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
                self.blur)

        self.depthlist = [256, 256, 128, 64]

        for depth in self.depthlist:
            self.addScale(depth)
        

    def set_device(self, device):
        self.device = device
        
    def addScale(self, dimNewScale):

        lastDim = self.depthScales[-1]
        self.scaleLayers.append(nn.ModuleList())
        self.scaleLayers[-1].append(EqualizedConv2d(lastDim,
                                                    dimNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=True,
                                                    initBiasToZero=True))

        self.scaleLayers[-1].append(AdaIN(self.dimMapping, dimNewScale))
        self.scaleLayers[-1].append(EqualizedConv2d(dimNewScale,
                                                    dimNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=True,
                                                    initBiasToZero=True))
        self.scaleLayers[-1].append(AdaIN(self.dimMapping, dimNewScale))

        self.toRGBLayers.append(ToRGB(dimNewScale, actvn=False))

        self.noiseModulators.append(nn.ModuleList())
        self.noiseModulators[-1].append(NoiseMultiplier())
        self.noiseModulators[-1].append(NoiseMultiplier())
        self.depthScales.append(dimNewScale)

    def forward(self, mapping, feat, img_size, output_size, alpha):
        batchSize = mapping.size(0)
        feat_size = feat.size(2) 

        feature = feat 
        feature = self.activation(feature)
        feature = self.adain00(feature, mapping)
        feature = self.conv0(feature)  # [batchSize, 256, 64, 64) -> # [batchSize, 256, 64, 64)

        feature = self.activation(feature)        
        feature = self.adain01(feature, mapping)

        num_depth_scale = int(np.log2(output_size)) - int(np.log2(img_size))
       
       # add different scales
        for nLayer, group in enumerate(self.scaleLayers):
            if nLayer + 1 > num_depth_scale:
                break
            
            skip = self.toRGBLayers[nLayer](feature) # calculate the last second feature

            noiseMod = self.noiseModulators[nLayer] # current noise module
            feature = Upscale2d(feature)            # Upsample (nearest neighborhood instead of biniliear)
#             feature = group[0](feature) + noiseMod[0](torch.randn((batchSize, 1,
#                                                       feature.size(2),
#                                                       feature.size(3)), device=mapping.device)) # inject noise
            feature = group[0](feature)
            feature = self.activation(feature)      # activation function
            feature = group[1](feature, mapping)    # adaptive instance normalization
#             feature = group[2](feature) + noiseMod[1](torch.randn((batchSize, 1,
#                                                       feature.size(2),
#                                                       feature.size(3)), device=mapping.device)) # inject noise
            feature = group[2](feature)
            feature = self.activation(feature)      # activation function
            feature = group[3](feature, mapping)    # adaptive instance normalization

        if num_depth_scale == 0:
            rgb = self.toRGBLayers[num_depth_scale](feature) 
        else: # FIXME: add upsample
            rgb = (1 - alpha) * self.upsample(skip) + alpha * self.toRGBLayers[num_depth_scale](feature) 

        return rgb

    def getOutputSize(self):

        side =  2**(2 + len(self.toRGBLayers))
        return (side, side)
