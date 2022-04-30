import random
import torch.nn as nn
import torch
import torch.nn.functional as F
import time
import curriculums
from torch.cuda.amp import autocast

from .volumetric_rendering import *
from .refinegan  import * 

class ImplicitGenerator3d(nn.Module):
    def __init__(self, siren, z_dim, stereo_auxiliary, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.siren = siren(output_dim=4, z_dim=self.z_dim, input_dim=3, device=None)
        self.stereo_auxiliary = stereo_auxiliary
        self.decoder = GNet(dimInput=256, dimHidden=256, dimMapping=256)
        self.epoch = 0
        self.step = 0

    def set_device(self, device):
        self.device = device
        self.siren.device = device
        self.generate_avg_frequencies()

    def forward(self, z, alpha, max_mixup_ratio, stereo_auxiliary, img_size, output_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """

        batch_size = z.shape[0]

        # Generate initial camera rays and sample points.
        with torch.no_grad():
            # points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            
            # get_initial_rays_trig
            x, y = torch.meshgrid(torch.linspace(-1, 1, img_size, device=self.device),
                                torch.linspace(1, -1, img_size, device=self.device))
            x = x.T.flatten()
            y = y.T.flatten()
            z_coord = -torch.ones_like(x, device=self.device) / np.tan((2 * math.pi * fov / 360)/2)

            rays_d_cam = torch.stack([x, y, z_coord], -1)
            rays_d_norm = torch.norm(rays_d_cam, dim=-1, keepdim=True)
            rays_d_cam = rays_d_cam / rays_d_norm # (height*width, 3)

            z_vals = torch.linspace(ray_start, ray_end, num_steps, device=self.device).reshape(1, num_steps, 1).repeat(img_size*img_size, 1, 1)
            points_cam = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals

            points_cam = torch.stack(batch_size*[points_cam])
            z_vals = torch.stack(batch_size*[z_vals])
            rays_d_cam = torch.stack(batch_size*[rays_d_cam]).to(self.device)

            primary_points, primary_z_vals, primary_ray_directions, primary_ray_origins, primary_pitch, primary_yaw, primary_cam2world_matrix = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)

            primary_ray_directions_expanded = torch.unsqueeze(primary_ray_directions, -2)
            primary_ray_directions_expanded = primary_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            primary_ray_directions_expanded = primary_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            primary_points = primary_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if stereo_auxiliary:
                auxiliary_points, auxiliary_z_vals, auxiliary_ray_directions, auxiliary_ray_origins, auxiliary_pitch, auxiliary_yaw, auxiliary_cam2world_matrix = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)

                auxiliary_ray_directions_expanded = torch.unsqueeze(auxiliary_ray_directions, -2)
                auxiliary_ray_directions_expanded = auxiliary_ray_directions_expanded.expand(-1, -1, num_steps, -1)
                auxiliary_ray_directions_expanded = auxiliary_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
                auxiliary_points = auxiliary_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            # seem useless
            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        # Model prediction on course points
        rgb_feat_dim = 256
       
        primary_output, primary_rgb_feat, mapping_codes = self.siren(primary_points, z, ray_directions=primary_ray_directions_expanded)
        primary_output = primary_output.reshape(batch_size, img_size * img_size, num_steps, 4)
        primary_rgb_feat = primary_rgb_feat.reshape(batch_size, img_size * img_size, num_steps, rgb_feat_dim)

        # Create images with NeRF
        primary_initial_rgb, primary_rgb_feat_maps, primary_depth, _ = rgb_feat_integration(primary_output, primary_rgb_feat, primary_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        
        primary_rgb_feat_maps = primary_rgb_feat_maps.reshape(batch_size, img_size, img_size, rgb_feat_dim)
        primary_rgb_feat_maps = primary_rgb_feat_maps.permute(0, 3, 1, 2).contiguous()

        primary_initial_rgb = primary_initial_rgb.reshape(batch_size, img_size, img_size, 3)
        primary_initial_rgb = primary_initial_rgb.permute(0, 3, 1, 2).contiguous()

        if stereo_auxiliary:
            auxiliary_output, auxiliary_rgb_feat, _ = self.siren(auxiliary_points, z, ray_directions=auxiliary_ray_directions_expanded)
            auxiliary_output = auxiliary_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            auxiliary_rgb_feat = auxiliary_rgb_feat.reshape(batch_size, img_size * img_size, num_steps, rgb_feat_dim)

            # Create images with NeRF
            auxiliary_initial_rgb, auxiliary_rgb_feat_maps, auxiliary_depth, _ = rgb_feat_integration(auxiliary_output, auxiliary_rgb_feat, auxiliary_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

            auxiliary_rgb_feat_maps = auxiliary_rgb_feat_maps.reshape(batch_size, img_size, img_size, rgb_feat_dim)
            auxiliary_rgb_feat_maps = auxiliary_rgb_feat_maps.permute(0, 3, 1, 2).contiguous()

            auxiliary_initial_rgb = auxiliary_initial_rgb.reshape(batch_size, img_size, img_size, 3)
            auxiliary_initial_rgb = auxiliary_initial_rgb.permute(0, 3, 1, 2).contiguous()

            # back project to the 3d space
            rays_d_cam = rays_d_cam.reshape((batch_size, img_size, img_size, 3))
            primary_points_3d = rays_d_cam * primary_depth.reshape(batch_size, img_size, img_size, 1)

            primary_points_homogeneous = torch.ones((batch_size, img_size, img_size, 4), device=self.device)
            primary_points_homogeneous[:, :, :, :3] = primary_points_3d

            primary_points_project_to_auxiliary = torch.bmm(torch.inverse(auxiliary_cam2world_matrix.float()) @ primary_cam2world_matrix, primary_points_homogeneous.reshape(batch_size, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(batch_size, img_size, img_size, 4)
            primary_grid_in_auxiliary = torch.cat((-primary_points_project_to_auxiliary[..., 0:1] / primary_points_project_to_auxiliary[..., 2:3], primary_points_project_to_auxiliary[..., 1:2] / primary_points_project_to_auxiliary[..., 2:3]), -1) * rays_d_norm.reshape(1, img_size, img_size, 1)

            warp_rgb_feat_maps = F.grid_sample(auxiliary_rgb_feat_maps, primary_grid_in_auxiliary, align_corners=True) 
            warp_rgb = F.grid_sample(auxiliary_initial_rgb, primary_grid_in_auxiliary, align_corners=True) 

            auxiliary_mixup_ratio = torch.rand((batch_size, 1, 1, 1), device=self.device) * max_mixup_ratio
            rgb_feat_maps = (1 - auxiliary_mixup_ratio) * primary_rgb_feat_maps + auxiliary_mixup_ratio * warp_rgb_feat_maps
        else:
            rgb_feat_maps = primary_rgb_feat_maps
            warp_rgb = None
  
        pixels = self.decoder(mapping_codes, rgb_feat_maps, img_size, output_size, alpha)
      
        return pixels, torch.cat([primary_pitch, primary_yaw], -1), primary_initial_rgb, warp_rgb


    def generate_avg_frequencies(self):
        """Calculates average frequencies and phase shifts"""

        z = torch.randn((10000, self.z_dim), device=self.siren.device)

        with torch.no_grad():
            frequencies, phase_shifts, mapping_codes = self.siren.mapping_network(z)

        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
        self.avg_mapping_codes = mapping_codes.mean(0, keepdim=True)

        return self.avg_frequencies, self.avg_phase_shifts, self.avg_mapping_codes


    def staged_forward(self, z, alpha, max_mixup_ratio, stereo_auxiliary, img_size, output_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=0.5, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        Similar to forward but used for inference.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """

        batch_size = z.shape[0]

        self.generate_avg_frequencies()

        with torch.no_grad():

            raw_frequencies, raw_phase_shifts, raw_avg_mapping_codes = self.siren.mapping_network(z)

            truncated_frequencies = self.avg_frequencies + psi * (raw_frequencies - self.avg_frequencies)
            truncated_phase_shifts = self.avg_phase_shifts + psi * (raw_phase_shifts - self.avg_phase_shifts)
            truncated_mapping_codes = self.avg_mapping_codes + psi * (raw_avg_mapping_codes - self.avg_mapping_codes)

            # get_initial_rays_trig
            x, y = torch.meshgrid(torch.linspace(-1, 1, img_size, device=self.device),
                                torch.linspace(1, -1, img_size, device=self.device))
            x = x.T.flatten()
            y = y.T.flatten()
            z_coord = -torch.ones_like(x, device=self.device) / np.tan((2 * math.pi * fov / 360)/2)

            rays_d_cam = torch.stack([x, y, z_coord], -1)
            rays_d_norm = torch.norm(rays_d_cam, dim=-1, keepdim=True)
            rays_d_cam = rays_d_cam / rays_d_norm # (height*width, 3)

            z_vals = torch.linspace(ray_start, ray_end, num_steps, device=self.device).reshape(1, num_steps, 1).repeat(img_size*img_size, 1, 1)
            points_cam = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals

            points_cam = torch.stack(batch_size*[points_cam])
            z_vals = torch.stack(batch_size*[z_vals])
            rays_d_cam = torch.stack(batch_size*[rays_d_cam]).to(self.device)
            
            primary_points, primary_z_vals, primary_ray_directions, primary_ray_origins, primary_pitch, primary_yaw, primary_cam2world_matrix = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)

            primary_ray_directions_expanded = torch.unsqueeze(primary_ray_directions, -2)
            primary_ray_directions_expanded = primary_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            primary_ray_directions_expanded = primary_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            primary_points = primary_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if False:
                #TODO: check this sampling strategy
                auxiliary_points, auxiliary_z_vals, auxiliary_ray_directions, auxiliary_ray_origins, auxiliary_pitch, auxiliary_yaw, auxiliary_cam2world_matrix = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
                auxiliary_ray_directions_expanded = torch.unsqueeze(auxiliary_ray_directions, -2)
                auxiliary_ray_directions_expanded = auxiliary_ray_directions_expanded.expand(-1, -1, num_steps, -1)
                auxiliary_ray_directions_expanded = auxiliary_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
                auxiliary_points = auxiliary_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            rgb_feat_dim = 256
            primary_output, primary_rgb_feat = self.siren.forward_with_frequencies_phase_shifts(primary_points, truncated_frequencies, truncated_phase_shifts, ray_directions=primary_ray_directions_expanded)

            primary_output = primary_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            primary_rgb_feat = primary_rgb_feat.reshape(batch_size, img_size * img_size, num_steps, rgb_feat_dim)

            primary_initial_rgb, primary_rgb_feat_maps, primary_depth, _ = rgb_feat_integration(primary_output, primary_rgb_feat, primary_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

            primary_depth_map = primary_depth.reshape(batch_size, img_size, img_size).contiguous().cpu()

            primary_rgb_feat_maps = primary_rgb_feat_maps.reshape(batch_size, img_size, img_size, rgb_feat_dim)
            primary_rgb_feat_maps = primary_rgb_feat_maps.permute(0, 3, 1, 2).contiguous()

            if False:
                auxiliary_output, auxiliary_rgb_feat = self.siren.forward_with_frequencies_phase_shifts(auxiliary_points, truncated_frequencies, truncated_phase_shifts, ray_directions=auxiliary_ray_directions_expanded)
                
                auxiliary_output = auxiliary_output.reshape(batch_size, img_size * img_size, num_steps, 4)
                auxiliary_rgb_feat = auxiliary_rgb_feat.reshape(batch_size, img_size * img_size, num_steps, rgb_feat_dim)
                # Create images with NeRF
                _, auxiliary_rgb_feat_maps, auxiliary_depth, _ = rgb_feat_integration(auxiliary_output, auxiliary_rgb_feat, auxiliary_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                auxiliary_rgb_feat_maps = auxiliary_rgb_feat_maps.reshape(batch_size, img_size, img_size, rgb_feat_dim)
                auxiliary_rgb_feat_maps = auxiliary_rgb_feat_maps.permute(0, 3, 1, 2).contiguous()

                # back project to the 3d space
                rays_d_cam = rays_d_cam.reshape((batch_size, img_size, img_size, 3))
                primary_points_3d = rays_d_cam * primary_depth.reshape(batch_size, img_size, img_size, 1)

                primary_points_homogeneous = torch.ones((batch_size, img_size, img_size, 4), device=self.device)
                primary_points_homogeneous[:, :, :, :3] = primary_points_3d

                primary_points_project_to_auxiliary = torch.bmm(torch.inverse(auxiliary_cam2world_matrix.float()) @ primary_cam2world_matrix, primary_points_homogeneous.reshape(batch_size, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(batch_size, img_size, img_size, 4)
                primary_grid_in_auxiliary = torch.cat((-primary_points_project_to_auxiliary[..., 0:1] / primary_points_project_to_auxiliary[..., 2:3], primary_points_project_to_auxiliary[..., 1:2] / primary_points_project_to_auxiliary[..., 2:3]), -1) * rays_d_norm.reshape(1, img_size, img_size, 1)

                warp_rgb_feat_maps = F.grid_sample(auxiliary_rgb_feat_maps, primary_grid_in_auxiliary, align_corners=True) 

                auxiliary_mixup_ratio = torch.rand((batch_size, 1, 1, 1), device=self.device) * max_mixup_ratio
                rgb_feat_maps = (1 - auxiliary_mixup_ratio) * primary_rgb_feat_maps + auxiliary_mixup_ratio * warp_rgb_feat_maps
            else:
                rgb_feat_maps = primary_rgb_feat_maps

            pixels = self.decoder(truncated_mapping_codes, rgb_feat_maps, img_size, output_size, alpha)
            pixels = pixels.cpu()

        return pixels, primary_depth_map


    def style_mix_forward(self, z1, z2, alpha, max_mixup_ratio, stereo_auxiliary, img_size, output_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=1, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        style mixing function
        """

        batch_size = z1.shape[0]

        self.generate_avg_frequencies()

        with torch.no_grad():

            z1_raw_frequencies,  z1_raw_phase_shifts,  z1_raw_avg_mapping_codes = self.siren.mapping_network(z1)
            z2_raw_frequencies,  z2_raw_phase_shifts,  z2_raw_avg_mapping_codes = self.siren.mapping_network(z2)

            # use the style from z1 in nerf
            truncated_frequencies = self.avg_frequencies + psi * (z1_raw_frequencies - self.avg_frequencies)
            truncated_phase_shifts = self.avg_phase_shifts + psi * (z1_raw_phase_shifts - self.avg_phase_shifts)
            
            # use the style from z2 in 2d gan
            truncated_mapping_codes = self.avg_mapping_codes + psi * (z2_raw_avg_mapping_codes - self.avg_mapping_codes)

            x, y = torch.meshgrid(torch.linspace(-1, 1, img_size, device=self.device),
                                torch.linspace(1, -1, img_size, device=self.device))
            x = x.T.flatten()
            y = y.T.flatten()
            z_coord = -torch.ones_like(x, device=self.device) / np.tan((2 * math.pi * fov / 360)/2)

            rays_d_cam = torch.stack([x, y, z_coord], -1)
            rays_d_norm = torch.norm(rays_d_cam, dim=-1, keepdim=True)
            rays_d_cam = rays_d_cam / rays_d_norm # (height*width, 3)

            z_vals = torch.linspace(ray_start, ray_end, num_steps, device=self.device).reshape(1, num_steps, 1).repeat(img_size*img_size, 1, 1)
            points_cam = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals

            points_cam = torch.stack(batch_size*[points_cam])
            z_vals = torch.stack(batch_size*[z_vals])
            rays_d_cam = torch.stack(batch_size*[rays_d_cam]).to(self.device)
            
            primary_points, primary_z_vals, primary_ray_directions, primary_ray_origins, primary_pitch, primary_yaw, primary_cam2world_matrix = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)

            primary_ray_directions_expanded = torch.unsqueeze(primary_ray_directions, -2)
            primary_ray_directions_expanded = primary_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            primary_ray_directions_expanded = primary_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            primary_points = primary_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            rgb_feat_dim = 256
            primary_output, primary_rgb_feat = self.siren.forward_with_frequencies_phase_shifts(primary_points, truncated_frequencies, truncated_phase_shifts, ray_directions=primary_ray_directions_expanded)

            primary_output = primary_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            primary_rgb_feat = primary_rgb_feat.reshape(batch_size, img_size * img_size, num_steps, rgb_feat_dim)

            primary_initial_rgb, primary_rgb_feat_maps, primary_depth, _ = rgb_feat_integration(primary_output, primary_rgb_feat, primary_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

            primary_depth_map = primary_depth.reshape(batch_size, img_size, img_size).contiguous().cpu()

            primary_rgb_feat_maps = primary_rgb_feat_maps.reshape(batch_size, img_size, img_size, rgb_feat_dim)
            primary_rgb_feat_maps = primary_rgb_feat_maps.permute(0, 3, 1, 2).contiguous()
                
            rgb_feat_maps = primary_rgb_feat_maps

            pixels = self.decoder(truncated_mapping_codes, rgb_feat_maps, img_size, output_size, alpha)
            pixels = pixels.cpu()

        return pixels, primary_depth_map

    def style_inter_forward(self, z1, z2, inter_ratio, alpha, max_mixup_ratio, stereo_auxiliary, img_size, output_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=1, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        style interpolation function
        """

        batch_size = z1.shape[0]

        self.generate_avg_frequencies()

        with torch.no_grad():

            z1_raw_frequencies,  z1_raw_phase_shifts,  z1_raw_avg_mapping_codes = self.siren.mapping_network(z1)
            z2_raw_frequencies,  z2_raw_phase_shifts,  z2_raw_avg_mapping_codes = self.siren.mapping_network(z2)

            # use the style from z1 
            z1_truncated_frequencies = self.avg_frequencies + psi * (z1_raw_frequencies - self.avg_frequencies)
            z1_truncated_phase_shifts = self.avg_phase_shifts + psi * (z1_raw_phase_shifts - self.avg_phase_shifts)
            z1_truncated_mapping_codes = self.avg_mapping_codes + psi * (z1_raw_avg_mapping_codes - self.avg_mapping_codes)
            
            # use the style from z2 
            z2_truncated_frequencies = self.avg_frequencies + psi * (z2_raw_frequencies - self.avg_frequencies)
            z2_truncated_phase_shifts = self.avg_phase_shifts + psi * (z2_raw_phase_shifts - self.avg_phase_shifts)
            z2_truncated_mapping_codes = self.avg_mapping_codes + psi * (z2_raw_avg_mapping_codes - self.avg_mapping_codes)

            # interpolation between z1 and z2
            truncated_frequencies = z1_truncated_frequencies * (1 - inter_ratio) + z2_truncated_frequencies * inter_ratio
            truncated_phase_shifts = z1_truncated_phase_shifts * (1 - inter_ratio) + z2_truncated_phase_shifts * inter_ratio
            truncated_mapping_codes = z1_truncated_mapping_codes * (1 - inter_ratio) + z2_truncated_mapping_codes * inter_ratio

            # get_initial_rays_trig
            x, y = torch.meshgrid(torch.linspace(-1, 1, img_size, device=self.device),
                                torch.linspace(1, -1, img_size, device=self.device))
            x = x.T.flatten()
            y = y.T.flatten()
            z_coord = -torch.ones_like(x, device=self.device) / np.tan((2 * math.pi * fov / 360)/2)

            rays_d_cam = torch.stack([x, y, z_coord], -1)
            rays_d_norm = torch.norm(rays_d_cam, dim=-1, keepdim=True)
            rays_d_cam = rays_d_cam / rays_d_norm # (height*width, 3)

            z_vals = torch.linspace(ray_start, ray_end, num_steps, device=self.device).reshape(1, num_steps, 1).repeat(img_size*img_size, 1, 1)
            points_cam = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals

            points_cam = torch.stack(batch_size*[points_cam])
            z_vals = torch.stack(batch_size*[z_vals])
            rays_d_cam = torch.stack(batch_size*[rays_d_cam]).to(self.device)
            
            primary_points, primary_z_vals, primary_ray_directions, primary_ray_origins, primary_pitch, primary_yaw, primary_cam2world_matrix = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)

            primary_ray_directions_expanded = torch.unsqueeze(primary_ray_directions, -2)
            primary_ray_directions_expanded = primary_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            primary_ray_directions_expanded = primary_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            primary_points = primary_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            rgb_feat_dim = 256
            primary_output, primary_rgb_feat = self.siren.forward_with_frequencies_phase_shifts(primary_points, truncated_frequencies, truncated_phase_shifts, ray_directions=primary_ray_directions_expanded)

            primary_output = primary_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            primary_rgb_feat = primary_rgb_feat.reshape(batch_size, img_size * img_size, num_steps, rgb_feat_dim)

            primary_initial_rgb, primary_rgb_feat_maps, primary_depth, _ = rgb_feat_integration(primary_output, primary_rgb_feat, primary_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

            primary_depth_map = primary_depth.reshape(batch_size, img_size, img_size).contiguous().cpu()

            primary_rgb_feat_maps = primary_rgb_feat_maps.reshape(batch_size, img_size, img_size, rgb_feat_dim)
            primary_rgb_feat_maps = primary_rgb_feat_maps.permute(0, 3, 1, 2).contiguous()
                
            rgb_feat_maps = primary_rgb_feat_maps

            pixels = self.decoder(truncated_mapping_codes, rgb_feat_maps, img_size, output_size, alpha)
            pixels = pixels.cpu()

        return pixels, primary_depth_map

    def staged_forward_for_3D_shape(self, z, samples, transformed_ray_directions_expanded, sigmas, voxel_resolution, max_batch, max_mixup_ratio, stereo_auxiliary, img_size, output_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=1, lock_view_dependence=False, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        extract 3D shape
        """
        head = 0
        self.generate_avg_frequencies()

        with torch.no_grad():

            raw_frequencies, raw_phase_shifts, raw_avg_mapping_codes = self.siren.mapping_network(z)

            truncated_frequencies = self.avg_frequencies + psi * (raw_frequencies - self.avg_frequencies)
            truncated_phase_shifts = self.avg_phase_shifts + psi * (raw_phase_shifts - self.avg_phase_shifts)

            while head < samples.shape[1]:
                coarse_output, _ = self.siren.forward_with_frequencies_phase_shifts(samples[:, head:head+max_batch],  truncated_frequencies, truncated_phase_shifts, ray_directions=transformed_ray_directions_expanded[:, :samples.shape[1]-head])
                coarse_output = coarse_output.reshape(samples.shape[0], -1, 4)
                sigmas[:, head:head+max_batch] = coarse_output[:, :, -1:]
                head += max_batch
        
        sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
    
        return sigmas

    def staged_forward_pri_and_aux(self, z, alpha, aux_h_mean, aux_v_mean, max_mixup_ratio, stereo_auxiliary, img_size, output_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=1, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        Similar to forward but used for inference and warp.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """

        batch_size = z.shape[0]

        self.generate_avg_frequencies()

        with torch.no_grad():

            raw_frequencies, raw_phase_shifts, raw_avg_mapping_codes = self.siren.mapping_network(z)

            truncated_frequencies = self.avg_frequencies + psi * (raw_frequencies - self.avg_frequencies)
            truncated_phase_shifts = self.avg_phase_shifts + psi * (raw_phase_shifts - self.avg_phase_shifts)
            truncated_mapping_codes = self.avg_mapping_codes + psi * (raw_avg_mapping_codes - self.avg_mapping_codes)
         
            # get_initial_rays_trig
            x, y = torch.meshgrid(torch.linspace(-1, 1, img_size, device=self.device),
                                torch.linspace(1, -1, img_size, device=self.device))
            x = x.T.flatten()
            y = y.T.flatten()
            z_coord = -torch.ones_like(x, device=self.device) / np.tan((2 * math.pi * fov / 360)/2)

            rays_d_cam = torch.stack([x, y, z_coord], -1)
            rays_d_norm = torch.norm(rays_d_cam, dim=-1, keepdim=True)
            rays_d_cam = rays_d_cam / rays_d_norm # (height*width, 3)

            z_vals = torch.linspace(ray_start, ray_end, num_steps, device=self.device).reshape(1, num_steps, 1).repeat(img_size*img_size, 1, 1)
            points_cam = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals

            points_cam = torch.stack(batch_size*[points_cam])
            z_vals = torch.stack(batch_size*[z_vals])
            rays_d_cam = torch.stack(batch_size*[rays_d_cam]).to(self.device)
            
            primary_points, primary_z_vals, primary_ray_directions, primary_ray_origins, primary_pitch, primary_yaw, primary_cam2world_matrix = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)

            primary_ray_directions_expanded = torch.unsqueeze(primary_ray_directions, -2)
            primary_ray_directions_expanded = primary_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            primary_ray_directions_expanded = primary_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            primary_points = primary_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            # auxliary images
            #TODO: check this sampling strategy
            auxiliary_points, auxiliary_z_vals, auxiliary_ray_directions, auxiliary_ray_origins, auxiliary_pitch, auxiliary_yaw, auxiliary_cam2world_matrix = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=aux_h_mean, v_mean=aux_v_mean, device=self.device, mode=sample_dist)
            auxiliary_ray_directions_expanded = torch.unsqueeze(auxiliary_ray_directions, -2)
            auxiliary_ray_directions_expanded = auxiliary_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            auxiliary_ray_directions_expanded = auxiliary_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            auxiliary_points = auxiliary_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            rgb_feat_dim = 256

            # primary output
            primary_output, primary_rgb_feat = self.siren.forward_with_frequencies_phase_shifts(primary_points, truncated_frequencies, truncated_phase_shifts, ray_directions=primary_ray_directions_expanded)

            primary_output = primary_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            primary_rgb_feat = primary_rgb_feat.reshape(batch_size, img_size * img_size, num_steps, rgb_feat_dim)

            _, primary_rgb_feat_maps, primary_depth, _ = rgb_feat_integration(primary_output, primary_rgb_feat, primary_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

            primary_rgb_feat_maps = primary_rgb_feat_maps.reshape(batch_size, img_size, img_size, rgb_feat_dim)
            primary_rgb_feat_maps = primary_rgb_feat_maps.permute(0, 3, 1, 2).contiguous()

            # auxilary output
            auxiliary_output, auxiliary_rgb_feat = self.siren.forward_with_frequencies_phase_shifts(auxiliary_points, truncated_frequencies, truncated_phase_shifts, ray_directions=auxiliary_ray_directions_expanded)
            
            auxiliary_output = auxiliary_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            auxiliary_rgb_feat = auxiliary_rgb_feat.reshape(batch_size, img_size * img_size, num_steps, rgb_feat_dim)

            _, auxiliary_rgb_feat_maps, auxiliary_depth, _ = rgb_feat_integration(auxiliary_output, auxiliary_rgb_feat, auxiliary_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

            auxiliary_rgb_feat_maps = auxiliary_rgb_feat_maps.reshape(batch_size, img_size, img_size, rgb_feat_dim)
            auxiliary_rgb_feat_maps = auxiliary_rgb_feat_maps.permute(0, 3, 1, 2).contiguous()

            # decode to RGB images
            primary_pixels = self.decoder(truncated_mapping_codes, primary_rgb_feat_maps, img_size, output_size, alpha)
            primary_pixels = primary_pixels

            auxiliary_pixels = self.decoder(truncated_mapping_codes, auxiliary_rgb_feat_maps, img_size, output_size, alpha)
            auxiliary_pixels = auxiliary_pixels

        return primary_pixels, auxiliary_pixels

    def staged_forward_reproj(self, z, alpha, aux_h_mean, aux_v_mean, max_mixup_ratio, stereo_auxiliary, img_size, output_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=1, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        Similar to forward but used for reprojection.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """

        batch_size = z.shape[0]

        self.generate_avg_frequencies()

        with torch.no_grad():

            raw_frequencies, raw_phase_shifts, raw_avg_mapping_codes = self.siren.mapping_network(z)

            truncated_frequencies = self.avg_frequencies + psi * (raw_frequencies - self.avg_frequencies)
            truncated_phase_shifts = self.avg_phase_shifts + psi * (raw_phase_shifts - self.avg_phase_shifts)
            truncated_mapping_codes = self.avg_mapping_codes + psi * (raw_avg_mapping_codes - self.avg_mapping_codes)
         
            # get_initial_rays_trig
            x, y = torch.meshgrid(torch.linspace(-1, 1, output_size, device=self.device),
                                torch.linspace(1, -1, output_size, device=self.device))
            x = x.T.flatten()
            y = y.T.flatten()
            z_coord = -torch.ones_like(x, device=self.device) / np.tan((2 * math.pi * fov / 360)/2)

            rays_d_cam = torch.stack([x, y, z_coord], -1)
            rays_d_norm = torch.norm(rays_d_cam, dim=-1, keepdim=True)
            rays_d_cam = rays_d_cam / rays_d_norm # (height*width, 3)

            z_vals = torch.linspace(ray_start, ray_end, num_steps, device=self.device).reshape(1, num_steps, 1).repeat(output_size*output_size, 1, 1)
            points_cam = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals

            points_cam = torch.stack(batch_size*[points_cam])
            z_vals = torch.stack(batch_size*[z_vals])
            rays_d_cam = torch.stack(batch_size*[rays_d_cam]).to(self.device)
            
            primary_points, primary_z_vals, primary_ray_directions, primary_ray_origins, primary_pitch, primary_yaw, primary_cam2world_matrix = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist)
            auxiliary_points, auxiliary_z_vals, auxiliary_ray_directions, auxiliary_ray_origins, auxiliary_pitch, auxiliary_yaw, auxiliary_cam2world_matrix = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=aux_h_mean, v_mean=aux_v_mean, device=self.device, mode=sample_dist)

            primary_ray_directions_expanded = torch.unsqueeze(primary_ray_directions, -2)
            primary_ray_directions_expanded = primary_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            primary_ray_directions_expanded = primary_ray_directions_expanded.reshape(batch_size, output_size*output_size*num_steps, 3)
            primary_points = primary_points.reshape(batch_size, output_size*output_size*num_steps, 3)

            # primary output
            primary_sigmas = self.siren.sigma_forward_with_frequencies_phase_shifts(primary_points, truncated_frequencies, truncated_phase_shifts, ray_directions=primary_ray_directions_expanded)
            primary_sigmas = primary_sigmas.reshape(batch_size, output_size * output_size, num_steps, 1)
            primary_depth = depth_integration(primary_sigmas, primary_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

            # back project to the 3d space
            rays_d_cam = rays_d_cam.reshape((batch_size, output_size, output_size, 3))
            primary_points_3d = rays_d_cam * primary_depth.reshape(batch_size, output_size, output_size, 1)

            primary_points_homogeneous = torch.ones((batch_size, output_size, output_size, 4), device=self.device)
            primary_points_homogeneous[:, :, :, :3] = primary_points_3d

            primary_points_project_to_auxiliary = torch.bmm(torch.inverse(auxiliary_cam2world_matrix.float()) @ primary_cam2world_matrix, primary_points_homogeneous.reshape(batch_size, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(batch_size, output_size, output_size, 4)
            reproject_grid = torch.cat((-primary_points_project_to_auxiliary[..., 0:1] / primary_points_project_to_auxiliary[..., 2:3], primary_points_project_to_auxiliary[..., 1:2] / primary_points_project_to_auxiliary[..., 2:3]), -1) * rays_d_norm.reshape(1, output_size, output_size, 1)

        return reproject_grid 