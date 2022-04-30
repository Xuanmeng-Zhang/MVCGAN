import torch
import torch.nn as nn

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


# FIXME: remove here, maybe we can use later
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
                dimFeat: int=256,
                dimInput: int=256,
                dimMapping: int=256,
                dimOutput: int=3,
                nMappingLayers: int=8,
                leakyReluLeak=0.2,
                generationActivation=None,
                phiTruncation=0.5,
                gamma_avg=0.99):

        super(GNet, self).__init__()
        self.dimMapping = dimMapping

        self.scaleLayers = nn.ModuleList()
        self.noiseModulators = nn.ModuleList()
        self.noramlizationLayer = NormalizationLayer()

        self.adain00 = AdaIN(dimMapping, dimInput)
        self.noiseMod00 = NoiseMultiplier()
        self.adain01 = AdaIN(dimMapping, dimInput)
        self.noiseMod01 = NoiseMultiplier()


        self.conv0 = EqualizedConv2d(dimInput, 256, 3, equalized=True,
                                     initBiasToZero=True, padding=1)     
        self.dimOutput = dimOutput
        self.toRGBLayer = EqualizedConv2d(  64,
                                            self.dimOutput,
                                            1,
                                            equalized=True,
                                            initBiasToZero=True)        

        self.activation = torch.nn.LeakyReLU(leakyReluLeak)
        self.depthScales = [dimInput]
        self.depthlist = [128]

        for depth in self.depthlist:
            self.addScale(depth)
        
        self.toRGBLayer = EqualizedConv2d(  self.depthScales[-1],
                                            self.dimOutput,
                                            1,
                                            equalized=True,
                                            initBiasToZero=True)         


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

        self.noiseModulators.append(nn.ModuleList())
        self.noiseModulators[-1].append(NoiseMultiplier())
        self.noiseModulators[-1].append(NoiseMultiplier())
        self.depthScales.append(dimNewScale)

    def forward(self, mapping, feat):
        batchSize = mapping.size(0)

        feat_size = feat.size(2) 

        feature = feat + self.noiseMod00(torch.randn((batchSize, 1, feat_size, feat_size), device=mapping.device)) # [batchSize, 256, 16, 16)
        feature = self.activation(feature)
        feature = self.adain00(feature, mapping)
        feature = self.conv0(feature)  # [batchSize, 256, 16, 16) -> # [batchSize, 256, 16, 16)
        feature = feature + self.noiseMod01(torch.randn((batchSize, 1, feature.size(2), feature.size(3)), device=mapping.device))
        feature = self.activation(feature)        
        feature = self.adain01(feature, mapping)
  
        for nLayer, group in enumerate(self.scaleLayers):
            noiseMod = self.noiseModulators[nLayer] # current noise module
            feature = Upscale2d(feature)            # Upsample (nearest neighborhood instead of biniliear)
            feature = group[0](feature) + noiseMod[0](torch.randn((batchSize, 1,
                                                      feature.size(2),
                                                      feature.size(3)), device=mapping.device)) # inject noise
            feature = self.activation(feature)      # activation function
            feature = group[1](feature, mapping)    # adaptive instance normalization
            feature = group[2](feature) + noiseMod[1](torch.randn((batchSize, 1,
                                                      feature.size(2),
                                                      feature.size(3)), device=mapping.device)) # inject noise
            feature = self.activation(feature)      # activation function
            feature = group[3](feature, mapping)    # adaptive instance normalization

        rgb = self.toRGBLayer(feature)
        rgb = torch.sigmoid(rgb)

        return rgb

    def getOutputSize(self):

        side =  2**(2 + len(self.toRGBLayers))
        return (side, side)
