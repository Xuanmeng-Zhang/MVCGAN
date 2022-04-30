import argparse
import os
import numpy as np
import math

import oss2
import time

from collections import deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from generators import generators
from discriminators import discriminators
from siren import siren
import fid_evaluation

import datasets
import curriculums
from tqdm import tqdm
from datetime import datetime
import copy
from torch_ema import ExponentialMovingAverage


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        
def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def load_images(images, curriculum, device):
    return_images = []
    head = 0
    for stage in curriculum['stages']:
        stage_images = images[head:head + stage['batch_size']]
        stage_images = F.interpolate(stage_images, size=stage['img_size'],  mode='bilinear', align_corners=True)
        return_images.append(stage_images)
        head += stage['batch_size']
    return return_images


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z


def train(rank, world_size, opt):
    torch.manual_seed(0)

    setup(rank, world_size, opt.port)
    device = torch.device(rank)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)
    
    fix_row =  int(5)
    fix_num = int(20)

    fixed_z = z_sampler((fix_num, 256), device='cpu', dist=metadata['z_dist'])

    SIREN = getattr(siren, metadata['model'])

    CHANNELS = 3

    scaler = torch.cuda.amp.GradScaler()

    if opt.load_dir != '':
        generator = torch.load(os.path.join(opt.load_dir, 'generator.pth'), map_location=device)
        discriminator = torch.load(os.path.join(opt.load_dir, 'discriminator.pth'), map_location=device)
        ema = torch.load(os.path.join(opt.load_dir, 'ema.pth'), map_location=device)
        ema2 = torch.load(os.path.join(opt.load_dir, 'ema2.pth'), map_location=device)
    else:
        generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim'], metadata['stereo_auxiliary']).to(device)
        discriminator = getattr(discriminators, metadata['discriminator'])().to(device)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)


    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    generator = generator_ddp.module
    discriminator = discriminator_ddp.module

    ssim =  SSIM().to(device)


    if metadata.get('unique_lr', False):
        mapping_network_param_names = [name for name, _ in generator_ddp.module.siren.mapping_network.named_parameters()]
        mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if n in mapping_network_param_names]
        generator_parameters = [p for n, p in generator_ddp.named_parameters() if n not in mapping_network_param_names]
        optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                        {'params': mapping_network_parameters, 'name': 'mapping_network', 'lr':metadata['gen_lr']*5e-2}],
                                       lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    else:
        optimizer_G = torch.optim.Adam(generator_ddp.parameters(), lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    optimizer_D = torch.optim.Adam(discriminator_ddp.parameters(), lr=metadata['disc_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    if opt.load_dir != '':
        optimizer_G.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_G.pth')))
        optimizer_D.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_D.pth')))
        if not metadata.get('disable_scaler', False):
            scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, 'scaler.pth')))

    generator_losses = []
    discriminator_losses = []

    if opt.set_step != None:
        generator.step = opt.set_step
        discriminator.step = opt.set_step

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    generator.set_device(device)

    # ----------
    #  Training
    # ----------

    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(curriculum))


    torch.manual_seed(rank)
    dataloader = None
    total_progress_bar = tqdm(total = opt.n_epochs, desc = "Total progress", dynamic_ncols=True)
    total_progress_bar.update(discriminator.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)
    for epoch in range (opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)

        # Set learning rates
        for param_group in optimizer_G.param_groups:
            if param_group.get('name', None) == 'mapping_network':
                param_group['lr'] = metadata['gen_lr'] * 5e-2
            else:
                param_group['lr'] = metadata['gen_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = metadata['disc_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']

        if not dataloader or dataloader.batch_size != metadata['batch_size']:
            dataset = datasets.get_dataset_zxm(
                                    metadata['dataset'],
                                        **metadata)

            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=metadata['batch_size'],
                shuffle=False,
                drop_last=True,
                pin_memory=True,
                num_workers=4,
            )
            
            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step)


            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))
        
        # modified 
        sampler.set_epoch(epoch+1)

        for i, (imgs, _) in enumerate(dataloader):
            # opt.model_save_interval default=5000
            if discriminator.step > 0 and discriminator.step % opt.model_save_interval == 0 and rank == 0:
                now = datetime.now()
                now = now.strftime("%d--%H:%M--")
                torch.save(ema, os.path.join(opt.output_dir, '{}_ema.pth'.format(discriminator.step)))
                torch.save(ema2, os.path.join(opt.output_dir, '{}_ema2.pth'.format(discriminator.step)))
                torch.save(generator_ddp.module, os.path.join(opt.output_dir, '{}_generator.pth'.format(discriminator.step)))
                torch.save(discriminator_ddp.module, os.path.join(opt.output_dir, '{}_discriminator.pth'.format(discriminator.step)))
                torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, '{}_optimizer_G.pth'.format(discriminator.step)))
                torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, '{}_optimizer_D.pth'.format(discriminator.step)))
                torch.save(scaler.state_dict(), os.path.join(opt.output_dir, '{}_scaler.pth'.format(discriminator.step)))
                
            metadata = curriculums.extract_metadata(curriculum, discriminator.step)

            if dataloader.batch_size != metadata['batch_size']: break

            if scaler.get_scale() < 1:
                scaler.update(1.)

            generator_ddp.train()
            discriminator_ddp.train()

            alpha = min(1, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))

            real_imgs = imgs.to(device, non_blocking=True)

            metadata['nerf_noise'] = max(0, 1. - discriminator.step/5000.)

            # TRAIN DISCRIMINATOR
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    z = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
                    split_batch_size = z.shape[0] // metadata['batch_split']
                    gen_imgs = []
                    gen_positions = []

                    for split in range(metadata['batch_split']):
                        subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                        g_imgs, g_pos, _, _ = generator_ddp(subset_z, alpha,  **metadata)

                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)

                assert real_imgs.shape == gen_imgs.shape

                real_imgs.requires_grad = True
                r_preds, _, _ = discriminator_ddp(real_imgs, alpha, **metadata)

            if metadata['r1_lambda'] > 0:
                # Gradient penalty
                grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
                inv_scale = 1./scaler.get_scale()
                grad_real = [p * inv_scale for p in grad_real][0]
            with torch.cuda.amp.autocast():
                if metadata['r1_lambda'] > 0:
                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = 0.5 * metadata['r1_lambda'] * grad_penalty
                else:
                    grad_penalty = 0

                g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)
                if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                    latent_penalty = torch.nn.MSELoss()(g_pred_latent, z) * metadata['z_lambda']
                    position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                    identity_penalty = latent_penalty + position_penalty
                else:
                    identity_penalty=0

                d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + grad_penalty + identity_penalty
                discriminator_losses.append(d_loss.item())

            optimizer_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_D)

            # TRAIN GENERATOR
            z = z_sampler((imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])

            split_batch_size = z.shape[0] // metadata['batch_split']
            for split in range(metadata['batch_split']):
                with torch.cuda.amp.autocast():
                    subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                    gen_imgs, gen_positions, gen_init_imgs, gen_warp_imgs= generator_ddp(subset_z, alpha, **metadata)
                    g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)

                    topk_percentage = max(0.99 ** (discriminator.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                    topk_num = math.ceil(topk_percentage * g_preds.shape[0])

                    g_preds = torch.topk(g_preds, topk_num, dim=0).values

                    if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                        latent_penalty = torch.nn.MSELoss()(g_pred_latent, subset_z) * metadata['z_lambda']
                        position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                        identity_penalty = latent_penalty + position_penalty
                    else:
                        identity_penalty = 0
                    
                    if metadata['reproj_lambda'] > 0:
                        pred = (gen_warp_imgs + 1) / 2
                        target = (gen_init_imgs + 1) / 2
                        abs_diff = torch.abs(target - pred)
                        l1_loss = abs_diff.mean(1, True)

                        ssim_loss = ssim(pred, target).mean(1, True)
                        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
                        reprojection_loss = reprojection_loss.mean() * metadata['reproj_lambda']
                    else:
                        reprojection_loss = 0

                    g_loss = torch.nn.functional.softplus(-g_preds).mean() + identity_penalty + reprojection_loss
                    generator_losses.append(g_loss.item())

                scaler.scale(g_loss).backward()
                
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            ema.update(generator_ddp.parameters())
            ema2.update(generator_ddp.parameters())


            if rank == 0:
                interior_step_bar.update(1)
                if i%10 == 0:
                    tqdm.write(f"[Experiment: {opt.output_dir}] [GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Step: {discriminator.step}] [Alpha: {alpha:.2f}] [Img Size: {metadata['output_size']}] [Batch Size: {metadata['batch_size']}] [TopK: {topk_num}] [Scale: {scaler.get_scale()}]")

                if discriminator.step % opt.sample_interval == 0:
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            gen_imgs = []

                            for idx in range(fixed_z.shape[0]):
                                g_imgs, _ = generator_ddp.module.staged_forward(fixed_z[idx:idx+1].to(device), alpha, **copied_metadata)
                                gen_imgs.append(g_imgs)
                            
                            gen_imgs = torch.cat(gen_imgs, axis=0)
                            gen_imgs = ((gen_imgs + 1) / 2).float()
                            gen_imgs = gen_imgs.clamp_(0, 1)

                    save_image(gen_imgs[:fix_num], os.path.join(opt.output_dir, f"{discriminator.step}_fixed.png"), nrow=fix_row, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            gen_imgs = []
                            
                            for idx in range(fixed_z.shape[0]):
                                g_imgs, _ = generator_ddp.module.staged_forward(fixed_z[idx:idx+1].to(device), alpha, **copied_metadata)
                                gen_imgs.append(g_imgs)
                                
                            gen_imgs = torch.cat(gen_imgs, axis=0)
                            gen_imgs = ((gen_imgs + 1) / 2).float()
                            gen_imgs = gen_imgs.clamp_(0, 1)


                    save_image(gen_imgs[:fix_num], os.path.join(opt.output_dir, f"{discriminator.step}_tilted.png"), nrow=fix_row, normalize=True)

                    ema.store(generator_ddp.parameters())
                    ema.copy_to(generator_ddp.parameters())
                    generator_ddp.eval()

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            gen_imgs = []
                            
                            for idx in range(fixed_z.shape[0]):
                                g_imgs, _ = generator_ddp.module.staged_forward(fixed_z[idx:idx+1].to(device),  alpha, **copied_metadata)
                                gen_imgs.append(g_imgs)
                                
                            gen_imgs = torch.cat(gen_imgs, axis=0)
                            gen_imgs = ((gen_imgs + 1) / 2).float()
                            gen_imgs = gen_imgs.clamp_(0, 1)
                            

                    save_image(gen_imgs[:fix_num], os.path.join(opt.output_dir, f"{discriminator.step}_fixed_ema.png"), nrow=fix_row, normalize=True)    

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            gen_imgs = []
                            
                            for idx in range(fixed_z.shape[0]):
                                g_imgs, _ = generator_ddp.module.staged_forward(fixed_z[idx:idx+1].to(device),  alpha, **copied_metadata)
                                gen_imgs.append(g_imgs)
                                
                            gen_imgs = torch.cat(gen_imgs, axis=0)
                            gen_imgs = ((gen_imgs + 1) / 2).float()
                            gen_imgs = gen_imgs.clamp_(0, 1)
                            

                    save_image(gen_imgs[:fix_num], os.path.join(opt.output_dir, f"{discriminator.step}_tilted_ema.png"), nrow=fix_row, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['psi'] = 0.7
                            rand_z = torch.randn_like(fixed_z)
                            gen_imgs = []
                            
                            for idx in range(fixed_z.shape[0]):
                                g_imgs, _ = generator_ddp.module.staged_forward(rand_z[idx:idx+1].to(device), alpha, **copied_metadata)
                                gen_imgs.append(g_imgs)
                                
                            gen_imgs = torch.cat(gen_imgs, axis=0)
                            gen_imgs = ((gen_imgs + 1) / 2).float()
                            gen_imgs = gen_imgs.clamp_(0, 1)
                            
                    save_image(gen_imgs[:fix_num].float(), os.path.join(opt.output_dir, f"{discriminator.step}_random.png"), nrow=fix_row, normalize=True)
              
                    ema.restore(generator_ddp.parameters())

                if discriminator.step % opt.sample_interval == 0:
                    torch.save(ema, os.path.join(opt.output_dir, 'ema.pth'))
                    torch.save(ema2, os.path.join(opt.output_dir, 'ema2.pth'))
                    torch.save(generator_ddp.module, os.path.join(opt.output_dir, 'generator.pth'))
                    torch.save(discriminator_ddp.module, os.path.join(opt.output_dir, 'discriminator.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'optimizer_G.pth'))
                    torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'optimizer_D.pth'))
                    torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'scaler.pth'))
                    torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
                    torch.save(discriminator_losses, os.path.join(opt.output_dir, 'discriminator.losses'))
                        

            if opt.eval_freq > 0 and (discriminator.step + 1) % opt.eval_freq == 0:
                generated_dir = os.path.join(opt.output_dir, 'evaluation/generated')

                if rank == 0:
                    fid_evaluation.setup_evaluation(metadata['dataset'], generated_dir, opt.eval_dir, dataset_path=metadata['dataset_path'], target_size=metadata['output_size'])
                    
                dist.barrier()
                ema.store(generator_ddp.parameters())
                ema.copy_to(generator_ddp.parameters())
                generator_ddp.eval()
                fid_evaluation.output_images(alpha, generator_ddp, metadata, rank, world_size, generated_dir)
                ema.restore(generator_ddp.parameters())
                dist.barrier()
                if rank == 0:
                    fid = fid_evaluation.calculate_fid(metadata['dataset'], generated_dir, opt.eval_dir, target_size=metadata['output_size'])
                    with open(os.path.join(opt.output_dir, f'fid.txt'), 'a') as f:
                        f.write(f'\n{discriminator.step}:{fid}')

                torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        generator.epoch += 1

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--eval_dir', type=str, default='/data/xuanmeng/eval/')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=2000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=2000)

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)

    current_time = time.strftime('%y-%m-%d-%H-%M-%S')
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)





