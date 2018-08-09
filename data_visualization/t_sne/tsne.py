# -*- coding: utf-8 -*-
import torch
import torch.autograd
from torch import nn


class TSNE(nn.Module):
    def __init__(self, n_points, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super(TSNE, self).__init__()
        # Logit of datapoint-to-topic weight
        self.logits = nn.Embedding(n_points, n_dim)

    def pairwise(self, data):
        n_obs, dim = data.size()
        # print("pairwise dim:", n_obs, dim)
        xk = data.unsqueeze(0).expand(n_obs, n_obs, dim)
        xl = data.unsqueeze(1).expand(n_obs, n_obs, dim)
        # print(xl.size())
        dkl2 = ((xk - xl) ** 2.0).sum(2).squeeze()
        # print(dkl2.size())
        return dkl2

    def forward(self, pij, i, j):
        # TODO: реализуйте вычисление матрицы сходства для точек отображения и расстояние Кульбака-Лейблера
        # pij - значения сходства между точками данных
        # i, j - индексы точек
        
        x = self.logits.weight
        # Compute squared pairwise distances
        dkl2 = self.pairwise(x)
        # Compute partition function
        n_diagonal = dkl2.size()[0]
        part = (1 + dkl2).pow(-1.0).sum() - n_diagonal

        # Compute the numerator
        xi = self.logits(i)
        xj = self.logits(j)
        # print(xj.size())
        num = ((1. + (xi - xj) ** 2.0).sum(1)).pow(-1.0).squeeze()

        qij = num / part.expand_as(num)
        # Compute KLD
        loss_kld = pij * (torch.log(pij) - torch.log(qij))

        return loss_kld.sum()

    def __call__(self, *args):
        return self.forward(*args)
