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

        model_ft = models.resnet152(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.add_module('dropout', nn.Dropout(p=0.5))
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        self.net = model_ft

        # self.net = FeatureExtractor(net, emb_name)
        
        #model_ft = resnet50_face_bn_dag('/home/mdomrachev/Data/STML_projects/pytorch/VEmotionNet/models/resnet50_face_bn_dag.pth')
        #new_model_removed = torch.nn.Sequential(*list(model_ft.children())[:-1])
        #new_model_removed.add_module('fc', torch.nn.Linear(2048, num_classes))
        #self.net = model_ft

        # self.features_net = sphere20a(feature=True)
        # self.features_net.load_state_dict(torch.load('/home/mdomrachev/Data/STML_projects/pytorch/VEmotionNet/models/sphere20a_20171020.pth'))
        # for param in self.features_net.parameters():
        #    param.requires_grad = True
        
        #self.init = 512
        #self.layer1 = 512
        #self.layer2 = 512

        #module = nn.Sequential()
        #module.add_module('linear1', nn.Linear(512, 512))
        #module.add_module('relu1', nn.SELU())
        #module.add_module('linear2', nn.Linear(512, 256))
        #module.add_module('relu2', nn.SELU())
        #module.add_module('output', nn.Linear(256, num_classes))

        #module.add_module('LSTM1', nn.LSTM(self.init, self.layer1, 1, dropout= 0.2))
        #module.add_module('pool', nn.AvgPool2d(1))
        #module.add_module('output', nn.Linear(self.layer1, num_classes))

        #self.net = module


    def forward(self, data):
        data = torch.squeeze(data)
        # data = self.features_net.forward(data)
        output = self.net(data)
        return output 


