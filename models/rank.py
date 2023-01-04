from __future__ import absolute_import
from __future__ import division

from torch import nn
from torch.nn import functional as F
import torchvision

class rank(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(rank, self).__init__()
        self.loss = loss
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        f = self.classifier(x)
        return f