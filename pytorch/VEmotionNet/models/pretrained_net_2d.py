# -*- coding: utf-8 -*-
import torch
from STML_projects.pytorch.VEmotionNet.models.net_sphere import sphere20a
from STML_projects.pytorch.VEmotionNet.models.resnet50_face_bn_dag import resnet50_face_bn_dag
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models
from STML_projects.pytorch.common.losses import *
from collections import OrderedDict



class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, data):
        x = data
        for name, module in self.submodule._modules.items():
            if len(module._modules.items()) != 0:
                for name2, module2 in module._modules.items():
                    x = module2(x)
            else:
                x = module(x)
        return x


class Flatten(nn.Module):

    def __init__(self):

        super(Flatten, self).__init__()

    def forward(self, x):

        return x.view(x.size(0), -1)


class CNNNet(nn.Module):
    def __init__(self, num_classes, depth, data_size, emb_name=[], pretrain_weight=None):
        super(CNNNet, self).__init__()
        sample_size = data_size['width']
        sample_duration = data_size['depth']

        # TODO: Реализуйте архитектуру нейронной сети

        # net = []
        # module = nn.Sequential()
        # module.add_module('conv1', nn.Conv2d(3, 32, 5, 1, 2))
        # module.add_module('conv2', nn.Conv2d(32, 96, 3, 1, 1))
        # module.add_module('pool', nn.AdaptiveAvgPool2d(1))
        # module.add_module('flatten', Flatten())
        # module.add_module('linear', nn.Linear(96, num_classes))
        # self.net = module

        # model_ft = models.resnet152(pretrained=True)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.add_module('dropout', nn.Dropout(p=0.5))
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)

        self.conv = nn.Sequential(
            nn.Conv3d(3, 64, 7, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            nn.Conv3d(64, 128, 5, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(128, 256, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(256, 512, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, (3, 3, 3), padding=2),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(512, 512, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(6144, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        # data = torch.squeeze(data)
        conv = self.conv(x)
        conv = conv.view(conv.size(0), -1)
        logit = self.fc(conv)
        return logit 


