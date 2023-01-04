from __future__ import absolute_import
from __future__ import division

from torch import nn
from torch.nn import functional as F
import torchvision

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class featureExtract(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(featureExtract, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:5])

    def forward(self, x):
        base_feature = self.base(x)
        #x = F.avg_pool2d(base_feature, base_feature.size()[2:])
        f = x.view(x.size(0), -1)

        return base_feature, f