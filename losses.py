from __future__ import absolute_import
from __future__ import division

import sys

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def DeepSupervision(criterion, xs, y):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss

class deterLoss(nn.Module):
    def __init__(self):
        super(deterLoss, self).__init__()
        self.loss_fn = torch.nn.BCELoss(size_average=False)
    def forward(self, inputs, targets):
        softmax_loss = self.loss_fn(inputs, targets)
        return softmax_loss
        



class quaLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(quaLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, im1, im2, hard1, hard2):
        pairo = self.getDis(im1, im2)
        pair1 = self.getDis(im1, hard1)
        pair2 = self.getDis(im2, hard2)

        y = torch.ones_like(pairo)
        loss01 = self.ranking_loss(pair1, pairo, y)
        loss02 = self.ranking_loss(pair2, pairo, y)

        return loss01 + loss02

    def getDis(self, x0, x1):
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        return dist







class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes.
    - epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = Variable(targets, requires_grad=False)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = 3
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.softmargin = nn.SoftMarginLoss()
        self.prob = 0

    def forward(self, inputs, targets, batch_idx, f=None):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_ = mask.data.cpu().numpy()

        dist_ap, dist_an, dist_an_ = [], [], []
        batch = n / 4
        num = 0
        for i in range(n):
            # print (dist[i][mask[i]].max().data.cpu().numpy())
            # print (dist[i][i].data.cpu().numpy())

            # if dist[i][mask[i]].max().data.cpu().numpy()[0] == dist[i][i].data.cpu().numpy()[0] :
            #     continue


            if i < n / 2:
                # print (np.where(mask_[i]==0))
                # print (torch.min(dist[i][mask[i]==0], 0)[1])
                mask_diff = np.where(mask_[i]==0)
                max_id = torch.min(dist[i][mask[i]==0], 0)[1]
                if mask_diff[0][int(max_id.data.cpu().numpy())] == i + batch * 2:
                    num += 1

                mask_same = np.where(mask_[i])
                max_id = torch.max(dist[i][mask[i]], 0)[1]
                if mask_same[0][int(max_id.data.cpu().numpy())] == i + batch * 2:
                    num += 1

                if dist[i][mask[i]].max().data.view(1)[0] != dist[i][i].data.view(1)[0]:
                    dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                    dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))

                else:
                    dist_an_.append(dist[i][mask[i] == 0].min().unsqueeze(0))

        ap_num = len(dist_ap)

        for idx, an in enumerate(dist_an_):
            id_ = idx % ap_num
            ap = dist_ap[id_]
            dist_ap.append(ap)
            dist_an.append(an)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        print (dist_ap.size(0))

        # print (num)
        # print (batch)

        prob = num / (2*batch)
        if batch_idx == 3:
            self.prob = 0
        self.prob = 0
        #self.prob = (self.prob * (batch_idx-3) + prob) / (batch_idx-2)

        # print (prob)
        # print (self.prob)


        # Compute ranking hinge loss`
        y = torch.ones_like(dist_an)

        if f != None:
            f.write('{}\n'.format(self.prob))

        #loss = self.modified_huber(dist_an, dist_ap, y)


        

        dist_ = dist_an - dist_ap
        print (dist_)

        # dist_sorted, _ = torch.sort(dist_, 0)
        # #print(dist_sorted)
        # loss = self.softmargin(dist_sorted, y) / 0.6931

        loss = self.ranking_loss(dist_an, dist_ap, y)


        return loss

    def modified_huber(self, x1, x2, y):

        N_total = y.size(0)

        z = y * (x1 - x2)
        z, _ = torch.sort(z, 0)

        #print(z)

        #self.margin = float(max(0.3, z[int(N_total / 3)].data.cpu().numpy()[0]))


        loss = - (1 + self.margin)**2 * z / (self.margin)**2 + 1

        

        idx = z >= -0.0
        idx = idx.detach()
        if int(torch.sum(idx).data.cpu().numpy()) > 0:
            loss[idx] = (z[idx] - self.margin)**2 / (self.margin)**2


        id2 = z >= self.margin
        id2 = id2.detach()
        if int(torch.sum(id2).data.cpu().numpy()) > 0:
            loss[id2] = 0.0

        # id2 = z >= 0
        # id2 = id2.detach()
        # loss[id2] = torch.log(1.0 + torch.exp(-z[id2]))

        N_zero = id2.sum().data.cpu().numpy()[0]
        #loss_sum = 1.0 / (N_total - N_zero) * torch.sum(loss)
        loss_sum = 1.0 / (N_total) * torch.sum(loss)

        return loss_sum


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
    - num_classes (int): number of classes.
    - feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
        - x: feature matrix with shape (batch_size, feat_dim).
        - labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


class RingLoss(nn.Module):
    """Ring loss.

    Reference:
    Zheng et al. Ring loss: Convex Feature Normalization for Face Recognition. CVPR 2018.
    """

    def __init__(self):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.ones(1, dtype=torch.float))

    def forward(self, x):
        loss = ((x.norm(p=2, dim=1) - self.radius) ** 2).mean()
        return loss