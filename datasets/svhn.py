import os
import torch
from torchvision.datasets import SVHN as PyTorchSVHN
import numpy as np

from datasets.common import get_target_transform
from datasets.templates import get_templates


class SVHN:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 image_text=False):

        # to fit with repo conventions for location
        modified_location = os.path.join(location, 'svhn')
        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.templates = get_templates('SVHN')
        self.target_transform = get_target_transform(self.templates, self.classnames) if image_text else None

        self.train_dataset = PyTorchSVHN(
            root=modified_location,
            download=True,
            split='train',
            transform=preprocess,
            target_transform=self.target_transform
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = PyTorchSVHN(
            root=modified_location,
            download=True,
            split='test',
            transform=preprocess,
            target_transform=self.target_transform
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )