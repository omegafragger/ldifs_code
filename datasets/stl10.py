import os
import torch
import torchvision.datasets as dsets

from datasets.common import get_target_transform
from datasets.templates import get_templates

class STL10:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 image_text=False):

        location = os.path.join(location, 'stl10')

        if (image_text):
            self.train_dataset = dsets.STL10(
                root=location,
                download=True,
                split='train',
                transform=preprocess
            )
            self.classnames = self.train_dataset.classes
            self.templates = get_templates('STL10')
            self.target_transform = get_target_transform(self.templates, self.classnames)
        else:
            self.target_transform = None

        self.train_dataset = dsets.STL10(
            root=location,
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

        self.test_dataset = dsets.STL10(
            root=location,
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

        self.classnames = self.train_dataset.classes