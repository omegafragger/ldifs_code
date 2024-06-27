import os
import torch
import torchvision.datasets as dsets

from datasets.common import get_target_transform
from datasets.templates import get_templates

class SUN397:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 image_text=False):
        # Data loading code
        traindir = os.path.join(location, 'sun397', 'train')
        valdir = os.path.join(location, 'sun397', 'val')

        if (image_text):
            self.train_dataset = dsets.ImageFolder(traindir, transform=preprocess)
            idx_to_class = dict((v, k)
                                for k, v in self.train_dataset.class_to_idx.items())
            self.classnames = [idx_to_class[i][2:].replace('_', ' ') for i in range(len(idx_to_class))]
            self.templates = get_templates('SUN397')
            self.target_transform = get_target_transform(self.templates, self.classnames)
        else:
            self.target_transform = None

        self.train_dataset = dsets.ImageFolder(traindir, transform=preprocess,
                                               target_transform=self.target_transform)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = dsets.ImageFolder(valdir, transform=preprocess,
                                              target_transform=self.target_transform)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i][2:].replace('_', ' ') for i in range(len(idx_to_class))]