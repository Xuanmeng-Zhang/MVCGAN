import os
import shutil
import torch
import copy
import argparse

from torchvision.utils import save_image
from pytorch_fid import fid_score
from torch_fidelity import calculate_metrics
from tqdm import tqdm

import datasets
import curriculums


def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in range(num_imgs//batch_size):
        real_imgs, _ = next(dataloader)

        for img in real_imgs:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
            img_counter += 1

def setup_evaluation(dataset_name, generated_dir, eval_dir, dataset_path, target_size=128, num_imgs=8000):
    # Only make real images if they haven't been made yet
    real_dir = os.path.join(eval_dir, 'EvalImages', dataset_name + '_real_images_' + str(target_size))
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        
        dataloader, CHANNELS = datasets.get_dataset(dataset_name, output_size=target_size, dataset_path=dataset_path)
        print('outputting real images...')
        output_real_images(dataloader, num_imgs, real_dir)
        print('...done')

    if generated_dir is not None:
        os.makedirs(generated_dir, exist_ok=True)
    return real_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebAHQ')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--num_imgs', type=int, default=8000)

    opt = parser.parse_args()
    real_images_dir = setup_evaluation(opt.dataset, None, eval_dir=opt.eval_dir, dataset_path=opt.dataset_path, target_size=opt.img_size, num_imgs=opt.num_imgs)
    metrics_dict = calculate_metrics(opt.output_dir, opt.real_image_dir, cuda=True, isc=True, fid=True, kid=True, verbose=False)
    print(metrics_dict)