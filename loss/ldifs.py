import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import functools
from collections import OrderedDict


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)


class LDIFSFinetune(nn.Module):
    def __init__(self, source_model, target_model):
        super(LDIFSFinetune, self).__init__()

        self.source_model = source_model
        self.target_model = target_model

        self.L = self.source_model.N_slices
        for param in self.source_model.parameters():
            param.requires_grad = False


    def forward(self, images):
        outs1, outs2 = self.source_model.get_features(images), self.target_model.get_features(images)
        feats1, feats2, diffs = [], [], []

        for kk in range(self.L):
            feats1.append(normalize_tensor(outs1[kk]))
            feats2.append(normalize_tensor(outs2[kk]))
            diff = ((feats1[kk] - feats2[kk]) ** 2)
            if (len(diff.shape) == 3):
                diff = diff.mean(dim=-1).mean(dim=-1)
            elif (len(diff.shape) == 2):
                diff = diff.mean(dim=-1)
            diffs.append(diff)
        ldifs = torch.stack(diffs, dim=1).mean(dim=-1).mean(dim=0)
        return ldifs



class CE_LDIFS(nn.Module):
    def __init__(self, pretrained_image_encoder, image_classifier, alpha=0.05):
        super(CE_LDIFS, self).__init__()
        self.pretrained_image_encoder = pretrained_image_encoder
        self.image_encoder = image_classifier.image_encoder
        self.classification_head = image_classifier.classification_head
        self.alpha = alpha

        self.ldifs_reg = LDIFSFinetune(self.pretrained_image_encoder, self.image_encoder)

    def forward(self, images, logits, labels):
        ce = F.cross_entropy(logits, labels)
        ldifs_reg = self.ldifs_reg(images)

        loss = ce + (self.alpha * ldifs_reg)
        return loss