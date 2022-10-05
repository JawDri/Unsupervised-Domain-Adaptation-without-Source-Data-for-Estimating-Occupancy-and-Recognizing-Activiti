import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('ConvTranspose1d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=2),
                nn.BatchNorm1d(32),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=2),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=2),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.ReLU()
                )   
        self.in_features = 768#384 for OE#768 for AR

    def forward(self, x):
        x = x.view(-1, 1, 32)#32 for AR#9 for OE
        x = x.float()
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x

class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv1d(1, 20, kernel_size=2),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Conv1d(20, 50, kernel_size=2),
                nn.Dropout(p=0.2),
                nn.MaxPool1d(2),
                nn.ReLU(),
                )
        self.in_features = 350

    def forward(self, x):
        x = x.view(-1, 1, 32)
        x = x.float()
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x