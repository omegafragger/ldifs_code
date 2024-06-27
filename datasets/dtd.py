import os
import torch
from torchvision.datasets import DTD as PytorchDTD

from datasets.templates import get_templates
from datasets.common import get_target_transform


class DTD:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 image_text=False):

        # Data loading code
        traindir = os.path.join(location, 'dtd', 'train')
        valdir = os.path.join(location, 'dtd', 'val')
        testdir = os.path.join(location, 'dtd', 'test')

        if (image_text):
            self.train_dataset = PytorchDTD(
                root=traindir, split='train', transform=preprocess, download=True)
            idx_to_class = dict((v, k)
                                for k, v in self.train_dataset.class_to_idx.items())
            self.classnames = [idx_to_class[i].replace(
                '_', ' ') for i in range(len(idx_to_class))]
            templates = get_templates('DTD')
            target_transform = get_target_transform(templates, self.classnames)


        self.train_dataset = PytorchDTD(
            traindir, split='train', transform=preprocess,
            target_transform=target_transform if image_text else None, download=True)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.val_dataset = PytorchDTD(
            valdir, split='val', transform=preprocess,
            target_transform=target_transform if image_text else None, download=True)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.test_dataset = PytorchDTD(
            testdir, split='test', transform=preprocess,
            target_transform=target_transform if image_text else None, download=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]