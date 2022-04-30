import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import random
import math
import pickle
import numpy as np

class FFHQ(Dataset):
    """CelelebA Dataset"""

    def __init__(self, dataset_path, output_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
            [transforms.Resize(576), 
            transforms.CenterCrop(512), 
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5]), 
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.Resize((output_size, output_size), interpolation=0)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0
        
class CelebAHQ(Dataset):
    """CelelebA Dataset"""

    def __init__(self, dataset_path, output_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"

        self.transform = transforms.Compose(
            [transforms.Resize(576), 
            transforms.CenterCrop(512), 
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5]), 
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.Resize((output_size, output_size), interpolation=0)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0
        
class Cat(Dataset):
    """AFHQ Dataset"""

    def __init__(self, dataset_path, output_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
            [transforms.CenterCrop(472), 
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5]), 
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.Resize((output_size, output_size), interpolation=0)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        return X, 0


def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8
    )
    return dataloader, 3

def get_dataset_zxm(name,  **kwargs):
    dataset = globals()[name](**kwargs)
    return dataset

def get_dataset_distributed(name, world_size, rank, batch_size, **kwargs):
    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    return dataloader, 3
