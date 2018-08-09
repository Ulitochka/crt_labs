# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import operator as op
import functools as ft


class TotalLoss(nn.Module):
    def __init__(self, loss_param, num_samples_per_classes, cuda_id):
        super(TotalLoss, self).__init__()
        self.loss_param = loss_param
        self.loss_types = list(loss_param.keys())

    def reduce_sum(self, x, keepdim=True):
        # silly PyTorch, when will you get proper reducing sums/means?
        for a in reversed(range(1, x.dim())):
            x = x.sum(a, keepdim=keepdim)
        return x

    def reduce_mean(self, x, keepdim=True):
        numel = ft.reduce(op.mul, x.size()[1:])
        x = self.reduce_sum(x, keepdim=keepdim)
        return x / numel

    def metric(self, x, y):
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        x_var = torch.var(x)
        y_var = torch.var(y)

        # https://github.com/rwightman/pytorch-nips2017-attack-example/blob/master/attacks/helpers.py
        # mean_cent_prod = self.reduce_mean((logits - logits_mean) * (targets - targets_mean))
        cov = torch.mean((x - x_mean) * (y - y_mean)) # / (x.size()[0] - 1)

        # concordance_loss = 1 - (2 * mean_cent_prod) / (logits_var + targets_var + (logits_mean - targets_mean)**2)
        concordance_loss = 1 - (2 * cov) / (x_var + y_var + (x_mean - y_mean) ** 2)

        return concordance_loss

    def forward(self, logits, targets, emb=None, emb_norm=None, step=None, summary_writer=None):
        total_loss = 0

        if 'MSE' in self.loss_types:
            total_loss += self.loss_param['MSE']['w'] * nn.MSELoss()(logits, targets)
        else:
            # TODO: .......... objective function .. ...... https://arxiv.org/pdf/1704.08619.pdf
            # https://github.com/tzirakis/Multimodal-Emotion-Recognition/blob/master/losses.py

            # https://pytorch.org/docs/stable/torch.html?highlight=torch%20mean#torch.mean

            vx = logits[:, 0]
            ax = logits[:, 1]
            
            vy = targets[:, 0]
            ay = targets[:, 1]
         
            total_loss += (self.metric(vx, vy) + self.metric(ax, ay)) / 2

        return total_loss