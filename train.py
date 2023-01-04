from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable

import data_manager
from dataset_loader import ImageDataset, CreatePair, Resample, testResample
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, DeepSupervision, quaLoss, TripletLoss
from utils.iotools import save_checkpoint
from utils.avgmeter import AverageMeter
from utils.logger import Logger
from utils.torchtools import set_bn_to_eval, count_num_param
from utils.reidtools import visualize_ranked_results
from eval_metrics import evaluate
from optimizers import init_optim
from Memory.memory_new import MEMORY2

from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.metrics.pairwise import cosine_similarity

import Memory.memory_kd as kdtree

from clr import CyclicLR

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='../olcd6/data',
                    help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0,
                    help="split index")
parser.add_argument('--use-lmdb', action='store_true',
                    help="whether to use lmdb dataset")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=64, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=100, type=int,
                    help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[20, 40], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--fixbase-epoch', default=0, type=int,
                    help="epochs to fix base network (only train classifier, default: 0)")
parser.add_argument('--fixbase-lr', default=0.0003, type=float,
                    help="learning rate (when base network is frozen)")
parser.add_argument('--freeze-bn', action='store_true',
                    help="freeze running statistics in BatchNorm layers during training (default: False)")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10,
                    help="print frequency")
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--load-weights', type=str, default='',
                    help="load pretrained weights but ignores layers that don't match in size")
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--vis-ranked-res', action='store_true',
                    help="visualize ranked results, only available in evaluation mode (default: False)")

args = parser.parse_args()

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

f1 = open("sample_rate.txt",'w')
f2 = open('result_200.txt', 'w')

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_imgreid_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
        use_lmdb=args.use_lmdb,
    )

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train,
                     use_lmdb=args.use_lmdb, lmdb_path=dataset.train_lmdb_path),
        batch_size=args.train_batch , shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test,
                     use_lmdb=args.use_lmdb, lmdb_path=dataset.query_lmdb_path),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test,
                     use_lmdb=args.use_lmdb, lmdb_path=dataset.gallery_lmdb_path),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    # queryloader = DataLoader(
    #     ImageDataset(dataset.train, transform=transform_test,
    #                  use_lmdb=args.use_lmdb, lmdb_path=dataset.query_lmdb_path),
    #     batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False,
    # )
    #
    # galleryloader = DataLoader(
    #     ImageDataset(dataset.train, transform=transform_test,
    #                  use_lmdb=args.use_lmdb, lmdb_path=dataset.gallery_lmdb_path),
    #     batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False,
    # )

    finetuneloader = DataLoader(
        Resample(dataset.train, transform_train),
        batch_size=args.train_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    # testloader = DataLoader(
    #     testResample(dataset.train, transform_train),
    #     batch_size=args.test_batch//4, shuffle=True, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=True,
    # )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent'}, use_gpu=use_gpu)
    featureExtractmodel = models.init_model(name='featureExtract', num_classes=dataset.num_train_pids, loss={'xent'}, use_gpu=use_gpu)
    determinermodel = models.init_model(name='determiner', num_classes=dataset.num_train_pids, loss={'xent'}, use_gpu=use_gpu)
    determinertestmodel = models.init_model(name='determiner_test', num_classes=dataset.num_train_pids, loss={'xent'}, use_gpu=use_gpu)

    # class Network(nn.Module):
    #     def __init__(self):
    #         super(Network, self).__init__()
    #         self.feat = featureExtractmodel
    #         self.det = determinertestmodel
    #
    #     def forward(self, x):
    #         op1 = self.feat(x)
    #         op2 = self.det(op1)
    #         return op2
    #
    # net = Network()
    # for i in net.state_dict():
    #     print(i)
            

    #rankmodel = models.init_model(name='rank', num_classes=dataset.num_train_pids, loss={'xent'}, use_gpu=use_gpu)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    criterion = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    #scheduler = CyclicLR(optimizer)

    if args.fixbase_epoch > 0:
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
            optimizer_tmp = init_optim(args.optim, model.classifier.parameters(), args.fixbase_lr, args.weight_decay)
        else:
            print("Warn: model has no attribute 'classifier' and fixbase_epoch is reset to 0")
            args.fixbase_epoch = 0

    if args.load_weights:
        # load pretrained weights but ignore layers that don't match in size
        print("Loading pretrained weights from '{}'".format(args.load_weights))
        checkpoint = torch.load(args.load_weights)
        pretrain_dict = checkpoint['state_dict']

        #print (pretrain_dict.keys())

        model_dict = featureExtractmodel.state_dict()
        pretrain_dict1 = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict1)
        featureExtractmodel.load_state_dict(model_dict)

        model_train_dict = determinertestmodel.state_dict()
        #print (model_train_dict.keys())

        pretrain_dict2 = {}
        for k, v in pretrain_dict.items():
            strs = '.'
            k_rename = k.split('.')
            if k_rename[0] == 'base':
                k_rename[1] = str(int(k_rename[1]) - 5)
            k_rename = strs.join(k_rename)

            if k_rename in model_train_dict and model_train_dict[k_rename].size() == v.size():
                pretrain_dict2[k_rename] = v
        model_train_dict.update(pretrain_dict2)
        determinertestmodel.load_state_dict(model_train_dict)

    if args.resume:
        if osp.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            rank1 = checkpoint['rank1']
            print("Loaded checkpoint from '{}'".format(args.resume))
            print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, rank1))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    if use_gpu:
        featureExtractmodel = nn.DataParallel(featureExtractmodel).cuda()
        determinermodel = nn.DataParallel(determinermodel).cuda()
        determinertestmodel = nn.DataParallel(determinertestmodel).cuda()
        #net = nn.DataParallel(net).cuda()

    if args.evaluate:
        memory = MEMORY2(10, 1, (1, 98304))
        #memory = MEMORY2(10, 1, (1, 256))
        print("Evaluate only")

        memory2 = kdtree.create(dimensions=256)

        # p = featureExtractmodel.state_dict()
        # print(p.keys())
        #
        # p = determinertestmodel.state_dict()
        # print(p.keys())

        criterion1 = quaLoss()
        criterion2 = TripletLoss()
        optimizer = init_optim(args.optim, featureExtractmodel.parameters(), args.lr, args.weight_decay)
        optimizer2 = init_optim(args.optim, determinertestmodel.parameters(), args.lr, args.weight_decay)
        weight_shore = []

        # for p in featureExtractmodel.parameters():
        #     pd = p.data.clone()
        #     weight_shore.append(pd)

        for epoch in range(args.start_epoch, 25):
        #if True:
            scheduler.step()
            #feat_total_np, lab_total
            test_im = deter_cov(epoch, featureExtractmodel, determinertestmodel, queryloader, galleryloader, finetuneloader, use_gpu, memory, memory2, criterion1, criterion2, optimizer, optimizer2, weight_shore)

        #online_train(featureExtractmodel, determinermodel, finetuneloader, queryloader, galleryloader, use_gpu, memory, criterion1, criterion2, optimizer)


        feat_total = []
        lab_total = []
        ori_pic_total = []


        for key, values in memory.sample_store.items():
            for value in values:
                v_reshape = np.mean(value.pic, axis=2)
                v_reshape = np.mean(v_reshape, axis=1)
                #print (v_reshape.shape)
                feat_total.append(v_reshape)
                lab_total.append(key)
                ori_pic_total.append(value.ori_pic)

        feat_total_np = np.array(feat_total)


        print(feat_total_np.shape)
        print(memory.samples_f.shape)
        print(len(lab_total))

        f = np.vstack([feat_total_np, memory.samples_f[:memory.num_training_samples, 0, 0, :]])
        #f = f.reshape(f.shape[0], f.shape[1] * f.shape[2])
        lab_total += [32] * memory.num_training_samples


        #lab_uiq = list(set(lab_total))
        # print(lab_uiq)
        #
        # lab_choose = lab_uiq[:10]
        #
        # idex = [i for i, x in enumerate(lab_total) if x in lab_choose]
        # lab_ = [x for i, x in enumerate(lab_total) if x in lab_choose]

        

        print("################# compute tsne #################3")

        fig = plt.figure(figsize=(15, 8))
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        Y = tsne.fit_transform(f)
        ax = fig.add_subplot(1, 1, 1)

        # print(lab_total)
        # print(len(lab_total))
        # print(Y.shape)

        #Y = Y[idex]


        id_no_center = [i for i, x in enumerate(lab_total) if x != 32]
        lab_ = [x for i, x in enumerate(lab_total) if x != 32]
        id_center = [i for i, x in enumerate(lab_total) if x == 32]
        Y_no = Y[id_no_center]

        #plt.scatter(Y_no[:, 0], Y_no[:, 1], c=lab_, cmap=plt.cm.Spectral)

        print (Y_no.shape, len(ori_pic_total))
        print (ori_pic_total[0].shape)
        print (type(ori_pic_total[0]))

        visualize(Y_no, ori_pic_total, ax)

        Y_is = Y[id_center]
        ax.scatter(Y_is[:, 0], Y_is[:, 1], c='xkcd:black')

        # ax.xaxis.set_major_formatter(NullFormatter())
        # ax.yaxis.set_major_formatter(NullFormatter())
        # ax.legend()

        plt.axis('tight')
        plt.savefig('memory.png')


        # distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)
        # if args.vis_ranked_res:
        #     visualize_ranked_results(
        #         distmat, dataset,
        #         save_dir=osp.join(args.save_dir, 'ranked_results'),
        #         topk=20,
        #     )
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    if args.fixbase_epoch > 0:
        print("Train classifier for {} epochs while keeping base network frozen".format(args.fixbase_epoch))

        for epoch in range(args.fixbase_epoch):
            start_train_time = time.time()
            train(epoch, model, criterion, optimizer_tmp, trainloader, use_gpu, freeze_bn=True)
            train_time += round(time.time() - start_train_time)

        del optimizer_tmp
        print("Now open all layers for training")

    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
            epoch + 1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1

            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            print(model.training)

            # save_checkpoint({
            #     'state_dict': state_dict,
            #     'rank1': rank1,
            #     'epoch': epoch,
            # }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

from matplotlib import offsetbox

def visualize(embed, x_test, ax):
    feat = embed
    ax_min = np.min(embed, 0)
    ax_max = np.max(embed, 0)
    ax_dist_sq = np.sum((ax_max - ax_min) ** 2)

    # plt.figure()
    # ax = plt.subplot(111)
    shown_images = np.array([[1., 1.]])
    for i in range(feat.shape[0]):
        # print feat[i], i, shown_images
        dist = np.sum((feat[i] - shown_images) ** 2, 1)
        # if np.min(dist) < 3e-4 * ax_dist_sq:  # don't show points that are too close
        #     continue
        shown_images = np.r_[shown_images, [feat[i]]]
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(x_test[i].transpose(1,2,0), zoom=0.15, cmap=plt.cm.gray_r),
            xy=feat[i], frameon=False
        )
        ax.add_artist(imagebox)

    # plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    # plt.title('Embedding from the last layer of the network')
    # plt.show()


def deter_cov(epoch, feature_model, determiner_model, queryloader, galleryloader, testloader, use_gpu, memory, memory2, criterion1, criterion2, optimizer, optimizer2, weight_shore):
    feature_model.train()

    # for param, state in zip(model.feat.parameters(), model.feat.state_dict()):
    #     print(state)
    #     print(param.requires_grad)

    # feat_total = []
    # lab_total = []
    #
    # for batch_idx, (imgs, pids, _) in enumerate(testloader):
    #
    #     if use_gpu:
    #         imgs, pids = imgs.cuda(), pids.cuda()
    #     imgs, pids = Variable(imgs), Variable(pids)
    #
    #     feat, sq_feat = feature_model(imgs)
    #     outputs = determiner_model(feat)
    #
    #     sq_feat = sq_feat.data.cpu().numpy()
    #     feat_np = feat.data.cpu().numpy()
    #
    #     lab = pids.data.cpu().numpy()
    #
    #     print("batch: ", batch_idx)
    #
    #     loss = criterion2(outputs, pids)
    #
    #     optimizer.zero_grad()
    #     optimizer2.zero_grad()
    #     loss.backward()
    #     # for p in determiner_model.parameters():
    #     #     #print("data: ", p.data)
    #     #     print("grad: ", p.grad.data.clone())
    #     #optimizer.step()
    #     optimizer2.step()





    #     ###############################################
    #     ### vis
    #     ###############################################
    #     #sq_feat = sq_feat.data.cpu().numpy()
    #
    #     #print(sq_feat.shape)
    #
    #     # lab = pids.data.cpu().numpy()
    #     batch = lab.shape[0]
    #
    #
    #     for i in range(batch):
    #         sq_feat_re = sq_feat[i][np.newaxis, np.newaxis, :]
    #         memory.update_memory(sq_feat_re, feat_np[i], lab[i])
    #         lab_total.append(lab[i])
    #
    #     feat_total.append(sq_feat)
    #
    #
    # feat_total_np = np.vstack(feat_total)
    #
    # return feat_total_np, lab_total







        ##################################################
        ### reg
        ##################################################
        # loss_reg = 0
        # for i, p in enumerate(feature_model.parameters()):
        #     l = (p - Variable(weight_shore[i])).pow(2)
        #     loss_reg += l.sum()
        # print("reg: ", loss_reg)




        ####################################################
        ### train
        ###################################################

        # if isinstance(outputs, tuple):
        #     loss = DeepSupervision(criterion1, outputs, pids)
        # else:
        #     loss = criterion1(outputs, pids)
        #     print(loss)
        # #optimizer.zero_grad()
        # optimizer2.zero_grad()
        # loss.backward()
        # #optimizer.step()
        # optimizer2.step()

    ##############################################################################

    if epoch == 1:
        print("close f1")
        f1.close()

    feat_total = []
    lab_total = []

    determiner_model.train()

    for batch_idx, (img1, img2, label) in enumerate(testloader):


        # if batch_idx > 3:
        #     img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
        #     img1, img2, label = Variable(img1), Variable(img2), Variable(label)
        #     feat1, sq_feat1 = feature_model(img1)
        #
        #     memory2.search_knn(sq_feat1[0])
        #
        #
        #     break

        if use_gpu:
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
        img1, img2, label = Variable(img1), Variable(img2), Variable(label)



        feat1, sq_feat1 = feature_model(img1)
        #feat2, sq_feat2 = feature_model(img2)

        #feat1, feat2, sq_feat1, sq_feat2 = feat1.detach(), feat2.detach(), sq_feat1.detach(), sq_feat2.detach()
        feat1, sq_feat1 = feat1.detach(), sq_feat1.detach()

        feat1_ = feat1.data.cpu().numpy()
        #feat2_ = feat2.data.cpu().numpy()
        lab = label.data.cpu().numpy()
        sq_feat1 = sq_feat1.data.cpu().numpy()
        #sq_feat2 = sq_feat2.data.cpu().numpy()

        np.set_printoptions(threshold='nan')
        test_similarity = cosine_similarity(sq_feat1, sq_feat1)
        print(test_similarity, lab)
        input()

        #if batch_idx > 2:
        if (batch_idx > 2 and epoch == 0) or (epoch!=0):

            #print (memory.sample_store)
            # input()

            hard1_ = []
            hard1_pid = []
            hard2_ = []
            hard2_pid = []

            centers = memory.samples_f[:memory.num_training_samples, 0, 0, :]
            cs1 = cosine_similarity(sq_feat1, centers)

            print (cs1)
            for key, value in memory.sample_store.items():
                print ([v.mean for v in value])
                a = np.stack([v.ori_pic for v in value])[:,0,0,:]
                cs_t = cosine_similarity(a, centers)
                # print (cs_t.shape)
                print(np.argmax(cs_t, axis=1))

            min_id1 = np.argmax(cs1, axis=1)

            # print (min_id1)


            for i in range(min_id1.shape[0]):
                cen = min_id1[i]
                # print(cen)
                sam_list = memory.sample_store[cen]

                # print (len(memory.sample_store))
                # print (len(sam_list))

                # sam_list = [item for item in sam_list if item.label != lab[i]]
                # print("len: ", len(sam_list))
                if sam_list:
                    sam = random.sample(sam_list, 1)[0]
                # else:
                #     print("label error")

                # print(sam.label)
                # while sam.label == lab[i]:
                #     sam = random.sample(sam_list, 1)[0]
                #print("hard_sam: ", sam.pic.shape)
                hard1_.append(sam.pic)
                hard1_pid.append(sam.label)
            hard1_ = np.array(hard1_)
            hard1_pid = np.array(hard1_pid)
            #print ("hard1.shape: ", hard1_.shape)

            print ('hard_label', hard1_pid)
            print (label)


            # cs2 = cosine_similarity(sq_feat2, centers)
            # min_id2 = np.argmax(cs2, axis=1)
            #
            # for i in range(min_id2.shape[0]):
            #     cen = min_id2[i]
            #     sam_list = memory.sample_store[cen]
            #
            #     #sam_list = [item for item in sam_list if item.label!=lab[i]]
            #     #print("len: ", len(sam_list))
            #     if sam_list:
            #         sam = random.sample(sam_list, 1)[0]
            #     # else:
            #     #     print("label error")
            #     # while sam.label == lab[i]:
            #     #     sam = random.sample(sam_list, 1)[0]
            #     hard2_.append(sam.pic)
            #     hard2_pid.append(sam.label)
            #
            # hard2_ = np.array(hard2_)
            # hard2_pid = np.array(hard2_pid)
            # #print("hard2.shape: ", hard2_.shape)


            hard1 = torch.from_numpy(hard1_)
            hard1 = Variable(hard1).cuda()
            # hard2 = torch.from_numpy(hard2_)
            # hard2 = Variable(hard2).cuda()

            hard1_pid = torch.from_numpy(hard1_pid)
            hard1_pid = Variable(hard1_pid).cuda()
            # hard2_pid = torch.from_numpy(hard2_pid)
            # hard2_pid = Variable(hard2_pid).cuda()

            #ips = [feat1, feat2, hard1, hard2]
            ips = [feat1, hard1]

            ops = []
            for ip in ips:
                op = determiner_model(ip)
                ops.append(op)

            #loss1 = criterion1(*ops)


            ops = torch.cat(ops)
            op_l = torch.cat([label, hard1_pid])
            #op_l = torch.cat([label, label, hard1_pid, hard2_pid])


            re_op = ops[0].data.cpu().numpy().reshape(-1,1)
            np.set_printoptions(threshold='nan')
            test_similarity = cosine_similarity(re_op, re_op)
            print(test_similarity, lab)
            input()


            if epoch == 0:
                loss2 = criterion2(ops, op_l, batch_idx, f1)
                #print ('write to f1')

            else:
                loss2 = criterion2(ops, op_l, batch_idx)



            print (loss2)


            #loss = loss2 + loss1

            #print (loss2)

            # print(loss)


            #optimizer.zero_grad()
            optimizer2.zero_grad()
            loss2.backward()
            # for p in determiner_model.parameters():
            #     #print("data: ", p.data)
            #     print("grad: ", p.grad.data.clone())
            # optimizer.step()
            optimizer2.step()

        if batch_idx % 100 == 0 and batch_idx != 0:
            test(feature_model, queryloader, galleryloader, use_gpu, finetune_model=determiner_model, ol=True)


        if epoch == 0:

            batch = lab.shape[0]
            for i in range(batch):
                sq_feat_re = sq_feat1[i][np.newaxis, np.newaxis, :]
                memory.update_memory(sq_feat_re, feat1_[i], lab[i])
                lab_total.append(lab[i])
        #feat_total.append(sq_feat1)

            #memory2.add(sq_feat1[i])


        #     for i in range(batch):
        #         sq_feat_re = sq_feat2[i][np.newaxis, np.newaxis, :]
        #         memory.update_memory(sq_feat_re, feat2_[i], lab[i])
        #         lab_total.append(lab[i])
        # #feat_total.append(sq_feat2)


        # if epoch == 1:
        #     for key, value in memory.sample_store.items():
        #         print (key, len(value))
        #         if len(value) > 3 and len(value) < 100:
        #             for i in value:
        #                 print ("i<<", i.label)
        #     print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        #     input()

            #memory2.add(sq_feat2[i])

    #feat_total_np = np.vstack(feat_total)

    #return feat_total_np, lab_total

    # print ("starting test")



    #test(feature_model, queryloader, galleryloader, use_gpu, finetune_model=determiner_model, ol=True)



# def online_train(feature_model, determiner_model, finetuneloader, queryloader, galleryloader, use_gpu, memory, criterion1, criterion2, optimizer):
#     feature_model.eval()
#     # rank_model.train()
#
#
#     # for param, state in zip(model.module.parameters(), model.module.state_dict()):
#     #     if state.startswith('base'):
#     #         param.requires_grad = False
#
#     for batch_idx, (img1, img2, pid) in enumerate(finetuneloader):
#         determiner_model.eval()
#
#         if use_gpu:
#             img1, img2, pid = img1.cuda(), img2.cuda(), pid.cuda()
#         img1, img2, pid = Variable(img1), Variable(img2), Variable(pid)
#
#         output1_ = feature_model(img1)
#         output2_ = feature_model(img2)
#
#         hard_sample1 = []
#         hard_sample2 = []
#         hard_sample1_id = []
#         hard_sample2_id = []
#
#         output1 = output1_.cpu().data.numpy()
#         output2 = output2_.cpu().data.numpy()
#         pid = pid.cpu().data.numpy()
#
#         batch = output1.shape[0]
#
#         if batch_idx > 0:
#
#             # print("~~~~~~~~~~~~~~~~~~~~~~", len(memory.sample_store))
#
#             for i in range(batch):
#                 pair_loader1 = DataLoader(CreatePair(memory, output1[i], pid[i]), batch_size=5, shuffle=True, num_workers=args.workers,
#                             pin_memory=True, drop_last=True,)
#
#                 pair_loader2 = DataLoader(CreatePair(memory, output2[i], pid[i]), batch_size=5, shuffle=True, num_workers=args.workers,
#                             pin_memory=True, drop_last=True,)
#
#                 score1 = []
#                 x2_1 = []
#                 lab_1 = []
#                 for e_id, (x1, x2, lab, id) in enumerate(pair_loader1):
#                     x1,x2 = Variable(x1).cuda(), Variable(x2).cuda()
#                     score = determiner_model([x1, x2])
#                     score1.append(score.cpu().data.numpy())
#                     x2_1.append(x1.cpu().data.numpy())
#                     lab_1.append(lab.numpy())
#                     if e_id > 150:
#                         break
#
#
#                 score2 = []
#                 x2_2 = []
#                 lab_2 = []
#                 for e_id, (x1, x2, lab, id) in enumerate(pair_loader2):
#                     x1, x2 = Variable(x1).cuda(), Variable(x2).cuda()
#                     score = determiner_model([x1, x2])
#                     score2.append(score.cpu().data.numpy())
#                     x2_2.append(x1.cpu().data.numpy())
#                     lab_2.append(lab.numpy())
#                     if e_id > 150:
#                         break
#
#                 score1 = np.concatenate(score1)
#                 score2 = np.concatenate(score2)
#
#                 x2_1 = np.concatenate(x2_1)
#                 x2_2 = np.concatenate(x2_2)
#
#                 lab_1 = np.concatenate(lab_1)
#                 lab_2 = np.concatenate(lab_2)
#
#                 # min_id1 = np.argmin(score1, axis=0)
#                 # min_id2 = np.argmin(score2, axis=0)
#                 print(score1.shape)
#
#                 candidate1 = np.where((score1<0.6) & (score1>0.4))[0]
#                 if candidate1.shape[0] > 1:
#                     min_id1 = random.sample(candidate1, 1)
#                     print("case1: ", min_id1)
#                 elif not np.where((score1>=0.6))[0].size:
#                     min_id1 = np.argmax(score1, axis=0)
#                     print("case2: ", min_id1)
#                 elif not np.where((score1<=0.4))[0].size:
#                     min_id1 = np.argmin(score1, axis=0)
#                     print("case3: ", min_id1)
#
#                 candidate2 = np.where((score2<0.6) & (score2>0.4))[0]
#                 if candidate2.shape[0] > 1:
#                     min_id2 = random.sample(candidate2, 1)
#                 elif not np.where((score2>=0.6))[0].size:
#                     min_id2 = np.argmax(score2, axis=0)
#                 elif not np.where((score2<=0.4))[0].size:
#                     min_id2 = np.argmin(score2, axis=0)
#
#
#                 # min_id1 = random.sample(np.where((score1<0.6) & (score1>0.4))[0], 1)
#                 # min_id2 = random.sample(np.where((score2<0.6) & (score2>0.4))[0], 1)
#
#
#                 hard_sample1.append(x2_1[min_id1, :])
#                 hard_sample1_id.append(lab_1[min_id1])
#                 hard_sample2.append(x2_2[min_id2, :])
#                 hard_sample2_id.append(lab_2[min_id2])
#
#
#             determiner_model.train()
#
#             hard_sample1 = np.concatenate(hard_sample1)
#             hard_sample1_id = np.concatenate(hard_sample1_id)
#             print(hard_sample1_id)
#             hard_sample2 = np.concatenate(hard_sample2)
#             hard_sample2_id = np.concatenate(hard_sample2_id)
#
#
#             hard_sample1, hard_sample2 = torch.from_numpy(hard_sample1).cuda(), torch.from_numpy(hard_sample2).cuda()
#             hard_sample1, hard_sample2 = Variable(hard_sample1), Variable(hard_sample2)
#
#             op1, op2, op3, op4, f1, f2, f3 = determiner_model([output1_, output2_, hard_sample1, hard_sample2])
#
#
#             label1 = np.ones(output1.shape[0]).astype(np.float32)
#
#             label2 = np.zeros(output1.shape[0]).astype(np.float32)
#             label2[np.where(pid == hard_sample1_id)] = 1
#
#             label3 = np.zeros(output1.shape[0]).astype(np.float32)
#             label3[np.where(pid == hard_sample2_id)] = 1
#
#             label1, label2, label3 = torch.from_numpy(label1).cuda(), torch.from_numpy(label2).cuda(), torch.from_numpy(label3).cuda()
#             label1, label2, label3 = Variable(label1), Variable(label2), Variable(label3)
#
#
#             inputs = torch.cat([f1, f2, f3], 0)
#             labels = torch.cat([label1, label2, label3], 0)
#
#
#             feat1 = torch.cat([op1, op1, op2], 0)
#             feat2 = torch.cat([op2, op3, op4], 0)
#
#             #print(pid, hard_sample1_id, labels)
#             # print(feat1, feat2)
#
#
#             loss_rank = criterion1(feat1, feat2, labels)
#             loss_deter = criterion2(inputs, labels)
#
#             print(loss_rank, loss_deter)
#
#             loss = loss_rank + loss_deter
#             #loss = loss_rank
#
#
#
#             optimizer.zero_grad()
#             loss.backward()
#             # for p in determiner_model.module.classifier.parameters():
#             #     print("data: ", p.data)
#             #     print("grad: ", p.grad.data.clone())
# 
#             optimizer.step()
#
#
#         for i in range(batch):
#             memory.update_memory(output1[i], output1[i], pid[i])
#             memory.update_memory(output2[i], output2[i], pid[i])
#
#         test(feature_model, queryloader, galleryloader, use_gpu, finetune_model=determiner_model, ol=False)


def train(epoch, model, criterion, optimizer, trainloader, use_gpu, freeze_bn=False):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    if freeze_bn or args.freeze_bn:
        model.apply(set_bn_to_eval)

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        imgs, pids = Variable(imgs), Variable(pids)

        outputs = model(imgs)
        if isinstance(outputs, tuple):
            loss = DeepSupervision(criterion, outputs, pids)
        else:
            loss = criterion(outputs, pids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.cpu().data.numpy().item(), pids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses))

        end = time.time()

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False, finetune_model=None, ol=False):
    batch_time = AverageMeter()



    model.eval()

    if ol:
        finetune_model.eval()


    qf, q_pids, q_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
        if use_gpu: imgs = imgs.cuda()
        imgs = Variable(imgs)

        end = time.time()
        features, _ = model(imgs)
        #print("before", features)
        if ol:
            features = finetune_model(features)
        #print("after", features)
        batch_time.update(time.time() - end)

        features = features.data.cpu()
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    gf, g_pids, g_camids = [], [], []
    end = time.time()
    for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
        if use_gpu: imgs = imgs.cuda()
        imgs = Variable(imgs)

        end = time.time()
        features, _ = model(imgs)
        if ol:
            features = finetune_model(features)
        batch_time.update(time.time() - end)

        features = features.data.cpu()
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print(gf.size())

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    f2.write('{} {}\n'.format(cmc[0], mAP))

    if return_distmat:
        return distmat
    return cmc[0]


if __name__ == '__main__':
    main()
    f2.close()