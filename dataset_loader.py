from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import lmdb
import io

import random

import torch
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, use_lmdb=False, lmdb_path=None):
        self.dataset = dataset
        self.transform = transform
        self.use_lmdb = use_lmdb
        self.lmdb_path = lmdb_path

        if self.use_lmdb:
            assert self.lmdb_path is not None
            self.env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]

        if self.use_lmdb:
            with self.env.begin(write=False) as txn:
                imgbuf = txn.get(img_path)
            img = Image.open(io.BytesIO(imgbuf)).convert('RGB')
        else:
            img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid


class CreatePair(Dataset):

    def __init__(self, memory, cur_prob, prob_id):
        self.memory = memory
        self.cur_prob = cur_prob
        self.prob_id = prob_id

    def __len__(self):
        return len(self.memory.sample_store)

    def __getitem__(self, item):
        mem = self.memory.sample_store[item]
        x0_data = mem.pic
        x1_data = self.cur_prob

        # if mem.label == self.prob_id:
        #     label = 1
        # else:
        #     label = 0

        return x0_data, x1_data, mem.label, item


class Resample(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.imgiddic = self.get_imgiddic()
        #self.imgsortid = self.get_img_sortiddic()
        self.transform = transform

    def sortid(self, train_id):
        train_sortid = list(set(train_id))
        train_sortid.sort()
        i = 0
        for x in train_id:
            train_id[i] = train_sortid.index(x)
            i = i + 1

        return train_id

    def get_imgiddic(self):

        d = {}
        for img_path, pid, _ in self.dataset:
            if pid in d:
                l1 = d[pid]
                l1.append(img_path)
            else:
                l2 = []
                l2.append(img_path)
                d[pid] = l2

        return d

    # def get_img_sortiddic(self):
    #     d = {}
    #     img_name = {data[0] for data in self.dataset}
    #     img_label = [data[1] for data in self.dataset]
    #     img_label = self.sortid(img_label)
    #
    #     count = 0
    #     for i in img_name:
    #         d[i] = img_label[count]
    #         count = count + 1
    #
    #     return d

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        anchor = self.dataset[item][0]

        anchorid = self.dataset[item][1]
        ap = list(random.sample(self.imgiddic[anchorid], 1))[0]

        img1 = read_image(anchor)
        img2 = read_image(ap)
        # a_id = self.imgsortid[ap[0]]
        # p_id = self.imgsortid[ap[1]]

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, anchorid

        

class testResample(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.imgiddic = self.get_imgiddic()
        self.transform = transform

    def get_imgiddic(self):

        d = {}
        for img_path, pid, _ in self.dataset:
            if pid in d:
                l1 = d[pid]
                l1.append(img_path)
            else:
                l2 = []
                l2.append(img_path)
                d[pid] = l2

        return d


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        anchorid = self.dataset[item][1]
        coin = random.uniform(0, 1)
        if coin > 0.5:
            ap = list(random.sample(self.imgiddic[anchorid], 2))
            label = 1.0
        else:
            ap = []
            new_id = random.randint(0, len(self.imgiddic)-1)
            while new_id == anchorid:
                new_id = random.randint(0, len(self.imgiddic) - 1)
            ap.append(random.sample(self.imgiddic[anchorid], 1)[0])
            ap.append(random.sample(self.imgiddic[new_id], 1)[0])
            label = 0.0

        img1 = read_image(ap[0])
        img2 = read_image(ap[1])


        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label