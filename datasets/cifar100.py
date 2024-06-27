import os
import torch
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100

from datasets.templates import get_templates
from datasets.common import get_target_transform

class CIFAR100:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 image_text=False):

        if (image_text):
            self.train_dataset = PyTorchCIFAR100(
                root=location, download=True, train=True, transform=preprocess
            )
            self.classnames = self.train_dataset.classes
            templates = get_templates('CIFAR100')
            target_transform = get_target_transform(templates, self.classnames)

        self.train_dataset = PyTorchCIFAR100(
            root=location, download=True, train=True, transform=preprocess,
            target_transform=target_transform if image_text else None
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR100(
            root=location, download=True, train=False, transform=preprocess,
            target_transform=target_transform if image_text else None
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes

