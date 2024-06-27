"""
Implementation of the L2-SP method in the paper: arxiv:1802.01483
Code partially taken from: https://github.com/thuml/Transfer-Learning-Library
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
from collections import OrderedDict


class L2Regularization(nn.Module):
    r"""The L2 regularization of parameters :math:`w` can be described as:
    .. math::
        {\Omega} (w) = \Vert w\Vert_2^2 ,
    Args:
        model (torch.nn.Module):  The model to apply L2 penalty.
    Shape:
        - Output: scalar.
    """
    def __init__(self, model: nn.Module):
        super(L2Regularization, self).__init__()
        self.model = model

    def forward(self):
        output = 0.0
        for param in self.model.parameters():
            output += torch.norm(param) ** 2
        return output


class SPRegularization(nn.Module):
    r"""
    The SP (Starting Point) regularization from `Explicit inductive bias for transfer learning with convolutional networks
    (ICML 2018) <https://arxiv.org/abs/1802.01483>`_
    The SP regularization of parameters :math:`w` can be described as:
    .. math::
        {\Omega} (w) =  \Vert w-w_0\Vert_2^2 ,
    where :math:`w_0` is the parameter vector of the model pretrained on the source problem, acting as the starting point (SP) in fine-tuning.
    Args:
        source_model (torch.nn.Module):  The source (starting point) model.
        target_model (torch.nn.Module):  The target (fine-tuning) model.
    Shape:
        - Output: scalar.
    """
    def __init__(self, source_model: nn.Module, target_model: nn.Module):
        super(SPRegularization, self).__init__()
        self.target_model = target_model
        self.source_weight = {}
        for name, param in source_model.named_parameters():
            self.source_weight[name] = param.detach()

    def forward(self):
        output = 0.0
        for name, param in self.target_model.named_parameters():
            output += torch.norm(param - self.source_weight[name]) ** 2
        return output


class L2SP(nn.Module):
    """
    Implementation of the L2SP regularization with CE loss.
    """
    def __init__(self, pretrained_image_encoder, image_classifier, alpha=0.5):
        super(L2SP, self).__init__()
        self.pretrained_image_encoder = pretrained_image_encoder
        self.image_encoder = image_classifier.image_encoder
        self.classification_head = image_classifier.classification_head
        self.alpha = alpha
        # self.beta = beta

        # self.l2_reg = L2Regularization(self.classification_head)
        self.sp_reg = SPRegularization(self.pretrained_image_encoder, self.image_encoder)

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels)
        # l2_reg = self.l2_reg()
        sp_reg = self.sp_reg()

        loss = ce + (self.alpha * sp_reg) # + (self.beta * l2_reg)
        return loss

        