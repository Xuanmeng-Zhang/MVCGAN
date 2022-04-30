import torch.nn as nn
import torch
from math import log2
from kornia.filters import filter2d
# from im2scene.layers import Blur

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)

class NeuralRenderer(nn.Module):
    ''' Neural renderer class

    Args:
        n_feat (int): number of features
        input_dim (int): input dimension; if not equal to n_feat,
            it is projected to n_feat with a 1x1 convolution
        out_dim (int): output dimension
        final_actvn (bool): whether to apply a final activation (sigmoid)
        min_feat (int): minimum features
        img_size (int): output image size
        use_rgb_skip (bool): whether to use RGB skip connections
        upsample_feat (str): upsampling type for feature upsampling
        upsample_rgb (str): upsampling type for rgb upsampling
        use_norm (bool): whether to use normalization
    '''

    def __init__(
            self, n_feat=128, input_dim=256, out_dim=3, final_actvn=True,
            min_feat=32, input_img_size=64, output_img_size=128, use_rgb_skip=True,
            upsample_feat="nn", upsample_rgb="bilinear", use_norm=False, device=None, **kwargs):
        super().__init__()

        self.device = device
        self.final_actvn = final_actvn
        self.input_dim = input_dim
        self.use_rgb_skip = use_rgb_skip
        self.use_norm = use_norm
        self.blur = Blur()
        n_blocks = int(log2(output_img_size) - log2(input_img_size))

        assert(upsample_feat in ("nn", "bilinear"))
        if upsample_feat == "nn":
            self.upsample_2 = nn.Upsample(scale_factor=2.)
        elif upsample_feat == "bilinear":
            self.upsample_2 = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), self.blur)

        assert(upsample_rgb in ("nn", "bilinear"))
        if upsample_rgb == "nn":
            self.upsample_rgb = nn.Upsample(scale_factor=2.)
        elif upsample_rgb == "bilinear":
            self.upsample_rgb = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), self.blur)

        if n_feat == input_dim:
            # self.conv_in = nn.Sequential()
            self.add_conv_input = False
        else:
            self.add_conv_input = True
            self.conv_in = nn.Conv2d(input_dim, n_feat, 1, 1, 0)

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1)] +
            [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                       max(n_feat // (2 ** (i + 2)), min_feat), 3, 1, 1)
                for i in range(0, n_blocks - 1)]
        )
        if use_rgb_skip:
            self.conv_rgb = nn.ModuleList(
                [nn.Conv2d(input_dim, out_dim, 3, 1, 1)] +
                [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                           out_dim, 3, 1, 1) for i in range(0, n_blocks)]
            )
        else:
            self.conv_rgb = nn.Conv2d(
                max(n_feat // (2 ** (n_blocks)), min_feat), 3, 1, 1)

        if use_norm:
            self.norms = nn.ModuleList([
                nn.InstanceNorm2d(max(n_feat // (2 ** (i + 1)), min_feat))
                for i in range(n_blocks)
            ])
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.add_conv_input:   
            net = self.conv_in(x)
        else:
            net = x

        if self.use_rgb_skip:
            rgb = self.upsample_rgb(self.conv_rgb[0](x))

        for idx, layer in enumerate(self.conv_layers):
            hid = layer(self.upsample_2(net))
            if self.use_norm:
                hid = self.norms[idx](hid)
            net = self.actvn(hid)

            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx + 1](net)
                if idx < len(self.conv_layers) - 1:
                    rgb = self.upsample_rgb(rgb)

        if not self.use_rgb_skip:
            rgb = self.conv_rgb(net)

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)
        return rgb
