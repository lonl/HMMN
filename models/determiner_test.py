from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.nn.functional as F

class determiner_test(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(determiner_test, self).__init__()
        self.loss = loss

        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[5:8])

        # self.classifier = nn.Sequential(
        #     nn.Linear(2048, 1500),
        #     nn.LeakyReLU(0.01, inplace=True),
        #     nn.Linear(1500, 1500),
        #     nn.LeakyReLU(0.01, inplace=True),
        #     nn.Linear(1500, 1500),
        #     nn.LeakyReLU(0.01, inplace=True),
        #     nn.Linear(1500, 1000),
        # )
        #
        # self.classifier2 = nn.Sequential(
        #     nn.LeakyReLU(0.01, inplace=True),
        #     nn.Linear(1000, 1000),
        #     nn.LeakyReLU(0.01, inplace=True),
        #     nn.Linear(1000, 1000),
        #     nn.LeakyReLU(0.01, inplace=True),
        #     nn.Linear(1000, 1000),
        #     nn.LeakyReLU(0.01, inplace=True),
        # )


        self.classifier = nn.Sequential(
            nn.Linear(2048, num_classes)
        )

        #self.classifier = nn.Linear(2048, num_classes)

        # self.fc2 = nn.Sequential(
        #     nn.Linear(num_classes * 2, num_classes),
        #     # nn.Linear(28*28, 50),
        #     nn.LeakyReLU(0.01, inplace=True),
        #     nn.Linear(num_classes, 100),
        #     nn.LeakyReLU(0.01, inplace=True),
        #     nn.Linear(100, 50),
        #     nn.LeakyReLU(0.01, inplace=True),
        #     nn.Linear(50, 1),
        # )


    def forward(self, inputs):
        if type(inputs) != list:
            if self.training:
                feature = self.base(inputs)
                feature = F.avg_pool2d(feature, feature.size()[2:])
                feature = feature.view(feature.size(0), -1)
                feature = self.classifier(feature)
                return feature
            else:
                feature = self.base(inputs)
                feature = F.avg_pool2d(feature, feature.size()[2:])
                feature = feature.view(feature.size(0), -1)
                return feature

        # n = len(inputs)
        # output = []
        #
        # for i in range(n):
        #     output.append(self.classifier(inputs[i]))
        #
        #
        # op_c = torch.cat([output[0], output[1]], dim=1)
        # f = self.fc2(op_c)
        # f = F.sigmoid(f)
        # return output[0], output[1], f