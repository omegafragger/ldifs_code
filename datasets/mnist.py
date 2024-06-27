import os
import torch
import torchvision.datasets as dsets

from datasets.common import get_target_transform
from datasets.templates import get_templates

class MNIST:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 image_text=False):

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if (image_text):
            templates = get_templates('MNIST')
            self.target_transform = get_target_transform(templates, self.classnames)
        else:
            self.target_transform = None

        self.train_dataset = dsets.MNIST(
            root=location,
            download=True,
            train=True,
            transform=preprocess,
            target_transform=self.target_transform
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = dsets.MNIST(
            root=location,
            download=True,
            train=False,
            transform=preprocess,
            target_transform=self.target_transform
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

