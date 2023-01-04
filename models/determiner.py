from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.nn.functional as F

class determiner(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(determiner, self).__init__()
        self.loss = loss
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1500),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(1500, 1000),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(1000, num_classes)
        )

        #self.classifier = nn.Linear(2048, num_classes)

        self.fc2 = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            # nn.Linear(28*28, 50),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(num_classes, 100),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(100, 50),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(50, 1),
        )


    def forward(self, inputs):
        if type(inputs) != list:
            feature = self.classifier(inputs)
            return feature

        n = len(inputs)
        output = []

        for i in range(n):
            output.append(self.classifier(inputs[i]))


        if self.training:
            output0 = output[0]
            output1 = output[1]
            output2 = output[2]
            output3 = output[3]

            op_1 = torch.cat([output0.detach(), output1.detach()], dim=1)
            op_2 = torch.cat([output0.detach(), output2.detach()], dim=1)
            op_3 = torch.cat([output1.detach(), output3.detach()], dim=1)

            f1 = self.fc2(op_1)
            f1 = F.sigmoid(f1)

            f2 = self.fc2(op_2)
            f2 = F.sigmoid(f2)

            f3 = self.fc2(op_3)
            f3 = F.sigmoid(f3)



            return output[0], output[1], output[2], output[3], f1, f2, f3


        else:
            op_c = torch.cat([output[0], output[1]], dim=1)
            f = self.fc2(op_c)
            f = F.sigmoid(f)
            return f