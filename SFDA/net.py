from easydl import *
from data import *
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn

class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm1d):
                module.train(False)
            else:
                module.train(mode)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_nn = nn.Sequential(
            nn.Linear(len(FEATURES), 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,256)
            )
    
    def forward(self, x):
        logits = self.linear_nn(x)
        return logits

model=Net()
class ResNet50Fc(BaseFeatureExtractor):
    def __init__(self,model_path=None, normalize=True):
        super(ResNet50Fc, self).__init__()
        print (normalize)
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = model
                self.model_resnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_resnet = model

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            
        else:
            self.normalize = False
       
        model_resnet = self.model_resnet

        self.linear_nn = model_resnet.linear_nn
        #self.__in_features = model_resnet.fc.in_features

        # self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        #if self.normalize:
            #x = (x - self.mean) / self.std
       
        x = self.linear_nn(x)

        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return 256





class CLS(nn.Module):

    def __init__(self, in_dim, out_dim, bottle_neck_dim=256, pretrain=False):
        super(CLS, self).__init__()
        self.pretrain = pretrain
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)
            self.main = nn.Sequential(self.bottleneck,self.fc,nn.Softmax(dim=-1))
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(self.fc,nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out



class CLS_copy(nn.Module):

    def __init__(self, in_dim, out_dim, bottle_neck_dim=256, pretrain=False):
        super(CLS_copy, self).__init__()
        self.pretrain = pretrain
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)
            self.main = nn.Sequential(self.bottleneck,self.fc)
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(self.fc)

    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x


class AdversarialNetwork(nn.Module):

    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y

