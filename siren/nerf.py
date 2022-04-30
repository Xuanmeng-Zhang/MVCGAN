import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

from .harmonic_embedding import HarmonicEmbedding

def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.0, mode='fan_in', nonlinearity='leaky_relu')

def _xavier_init(m):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)

class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength
        
    def forward(self, coordinates):
        return coordinates * self.scale_factor
        
class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(z_dim, map_hidden_dim), 
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(map_hidden_dim, map_output_dim)        
        )

        self.network.apply(kaiming_leaky_init)
        
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        style = self.network(z)
        style = F.leaky_relu(style, negative_slope=0.2)
        style_s = style[..., :style.shape[-1]//2]
        style_b = style[..., style.shape[-1]//2:]

        return style_s, style_b

class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
            modify  image -> point cloud
            for image: [B, C, H, W]
            for input point coordinate: [B, H * W * num_step, C] -> [B, H * W, num_step, C]
            perform intance norm 
        """
        super(InstanceNorm, self).__init__()
        self.num_steps = 12
        self.epsilon = epsilon

    def forward(self, x):
        batch_size, _, hidden_dim = x.shape
        # print(x.shape)
        x = x.view(batch_size, -1, self.num_steps, hidden_dim)
        x   = x - torch.mean(x, (1), True)
        x = x.view(batch_size, -1, hidden_dim)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (1), True) + self.epsilon)
        return x * tmp

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
            modify  image -> point cloud
            image: [B, C, H, W]
            input point coordinate: [B, H * W * num_step, C]
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=2, keepdim=True) + self.epsilon)

        return x * tmp1
        
        
class StyleBlock(nn.Module):
    def __init__(
        self,
        input_dim, 
        hidden_dim,
        with_activation = True,
        use_pixel_norm = False,
        use_instance_norm = False):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        # self.layer.apply(_xavier_init)

        self.with_activation = with_activation

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None

    def forward(self, x, style_s, style_b):
        x = self.layer(x)
                    
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)

        if self.instance_norm is not None:
            x = self.instance_norm(x)
        
        if self.with_activation:
            x = F.leaky_relu(x, negative_slope=0.1)

        style_s = style_s.unsqueeze(1).expand_as(x)
        style_b = style_b.unsqueeze(1).expand_as(x)

        x = style_s * x + style_b
        
        return x
        
class StyleNeRF(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(
        self, 
        z_dim: int = 100, 
        n_harmonic_functions_xyz: int = 10,
        n_harmonic_functions_dir: int = 4,
        hidden_dim_xyz: int = 256,
        hidden_dim_dir: int = 256,
        device=None):
        super().__init__()
        self.device = device
        self.z_dim = z_dim

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        self.hidden_dim_xyz = hidden_dim_xyz
        self.hidden_dim_dir = hidden_dim_dir

        self.mlp_xyz_s0 = nn.ModuleList([
            StyleBlock(embedding_dim_xyz, hidden_dim_xyz),
            StyleBlock(hidden_dim_xyz, hidden_dim_xyz),
            StyleBlock(hidden_dim_xyz, hidden_dim_xyz), 
        ])

        self.mlp_xyz_s1 = nn.ModuleList([
            StyleBlock(hidden_dim_xyz + embedding_dim_xyz, hidden_dim_xyz), 
            StyleBlock(hidden_dim_xyz, hidden_dim_xyz), 
            StyleBlock(hidden_dim_xyz, hidden_dim_xyz), 
        ])

        self.density_layer = nn.Linear(hidden_dim_xyz, 1)
        self.density_layer.apply(_xavier_init)

        self.mlp_xyz_s2 = StyleBlock(hidden_dim_xyz, hidden_dim_xyz, with_activation=False)

        self.mlp_xyz_dir = StyleBlock(hidden_dim_xyz + embedding_dim_dir, hidden_dim_dir)

        self.color_layer = nn.Linear(hidden_dim_dir, 3)

        self.mapping_network = CustomMappingNetwork(
            z_dim, hidden_dim_xyz, 
            (len(self.mlp_xyz_s0) + len(self.mlp_xyz_s1) + 1) * 2 * hidden_dim_xyz + hidden_dim_dir * 2)
        
        self.gridwarper = UniformBoxWarp(0.24) 
        # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.
        

    def forward(self, input, z, ray_directions, **kwargs):
        style_s, style_b = self.mapping_network(z)
        return self.forward_with_style(input, style_s, style_b, ray_directions, **kwargs)
        
        
    def forward_with_style(self, input, style_s, style_b, ray_directions, **kwargs):
        input = self.gridwarper(input)
        x = input.contiguous()   # x / input [B, H * W * num_steps(64 * 64 * 12=49152), 3]   

        # stage 0
        # add poistion encoding
        position_encoding = self.harmonic_embedding_xyz(x)
        for index, layer in enumerate(self.mlp_xyz_s0):
            if index == 0:
                start = 0
            else:
                start += self.hidden_dim_xyz
                
            end = start +  self.hidden_dim_xyz
            
            if index == 0:
                x = layer(position_encoding, style_s[..., start:end], style_b[..., start:end]) 
            else:   
                x = layer(x, style_s[..., start:end], style_b[..., start:end]) 
        
        # stage 1
        x = torch.cat([x, position_encoding], dim=2)  # (H, W, N_sample, D+pos_in_dims)
        for index, layer in enumerate(self.mlp_xyz_s1):
            start += self.hidden_dim_xyz
            end = start +  self.hidden_dim_xyz
            x = layer(x, style_s[..., start:end], style_b[..., start:end]) 
        
        # output density
        sigma = self.density_layer(x)  # (H, W, N_sample, 1)
        
        start += self.hidden_dim_xyz
        end = start +  self.hidden_dim_xyz
        x = self.mlp_xyz_s2(x, style_s[..., start:end], style_b[..., start:end])

        direction_encoding = self.harmonic_embedding_dir(ray_directions.contiguous())  # B, 64*64*12, 27
        x = torch.cat([x, direction_encoding], dim=-1)  # (B, H * W * N_sample, hidden_dim_xyz + dir_enc_dime 256)

        start += self.hidden_dim_xyz
        end = start +  self.hidden_dim_xyz        
        x = self.mlp_xyz_dir(x, style_s[..., start:end], style_s[..., start:end])
            
        return x, sigma


