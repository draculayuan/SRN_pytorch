from __future__ import division
# workaround of the bug where 'import torchvision' sets the start method to be 'fork'

import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')

from srn.model.resnet_srn import resnet50
from srn.utils import bbox_helper
from srn.utils.log_helper import init_log
from srn.utils.load_helper import restore_from
from srn.dataset import Wider_face
from srn.utils.criterion import SRN_criterion

import argparse
import logging
import os
import cv2
import math

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import json

parser = argparse.ArgumentParser(description='SRN Testing')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter in json format')
parser.add_argument('--img_list', dest='img_list', type=str,
                    help='meta file list for cross dataset validating')
parser.add_argument('--max_size', dest='max_size', type=int, required=True,
                    help='max test size')
parser.add_argument('--train_list', default='/media/liu/01D3895F4AD38170/pet_project/SRN/data/wider_face_split/trial.txt', type=str)
parser.add_argument('--img_root', default='/media/liu/01D3895F4AD38170/pet_project/SRN/data/WIDER_train/images', type=str)

def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    for key in cfg.keys():
        if key != 'shared':
            cfg[key].update(cfg['shared'])
    return cfg


def main():

    #init_log('global', logging.INFO)
    #logger = logging.getLogger('global')
    global args
    args = parser.parse_args()
    cfg = load_config(args.config)
   
    model = resnet50(pretrained=False, cfg = cfg['shared'])
    #print(model)
    '''
    # resume from the released model
    assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
    model = restore_from(model, args.resume)
    '''
    model = model.cuda()

    #logger.info('build dataloader done')
    train(model, cfg, args)


def train(model, cfg, args):
    logger = logging.getLogger('global')

    # switch to evaluate mode
    model.train()
    '''
    logger.info('start training')
    if not os.path.exists(args.results_dir):
        try:
            os.makedirs(args.results_dir)
        except Exception as e:
            print(e)
    '''
    # define the largest input size
    largest_input = args.max_size * args.max_size
    
    # generate dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    dataset = Wider_face(args.img_root, args.train_list, 300, 300, \
                        transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                        ]))
    train_loader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=dataset.collate_fn)
    # build criterion and optimizer
    criterion = SRN_criterion(0.3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,
                                    momentum=0.9)
    # training loops
    for ep in range(10):
        print('Epoch number {}'.format(ep+1))
        for batch_idx, (data, boxes, infos) in enumerate(train_loader):
            x = {
                'cfg': cfg,
                'image': data.cuda(),
                'image_info': infos,
                'ignore_regions': None
            }
            outputs = model(x)['predict'] # i dont plan to calculate loss in model, need make changes
            '''
            outputs has a length of 6, which is the number of levels. Each level has length of 2 (fs and ss) 
            within each shot, there is a tensor of size (num_proposals_in_whole_batch, 10)
            the 10 consists of bidx, x1, x2, x3, x4, score_0, score_1, ax1, ax2, ax3, ax4
            '''

            loss = 0
            for level in outputs:
                for shot in level:
                    loss += criterion(shot, boxes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 2 == 0:        
                print('batch: {}, Loss is {}'.format(batch_idx, loss))
            '''
            import time
            time.sleep(10)
            raise ValueError('Terminated by lord yuan!')
            '''

def write_wider_result(img_dir, dts, output_path):
    img_cls_label = img_dir.split('/')[-2]
    output_dir = os.path.join(output_path, img_cls_label)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, img_dir.split('/')[-1].replace('jpg', 'txt')), 'w') as f:
        f.write(img_dir.split('/')[-1].strip('.jpg') + '\n')
        f.write(str(dts.shape[0]) + '\n')
        for i in range(dts.shape[0]):
            f.write('{} {} {} {} {}\n'.format(round(float(dts[i][0])),
                                         round(float(dts[i][1])),
                                         round(float(dts[i][2] - dts[i][0] + 1)),
                                         round(float(dts[i][3] - dts[i][1] + 1)),
                                         round(float(dts[i][4]), 3)))


def bbox_flip(proposal, img_width):
    proposal[:, 1], proposal[:, 3] = img_width - proposal[:, 3] - 1, img_width - proposal[:, 1] - 1
    return proposal


def bbox_resize(proposal, resize_scale):
    proposal[:, 1:5] = (proposal[:, 1:5] - (resize_scale - 1) / 2.0) / resize_scale
    return proposal


def preprocess(img):
    totensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = totensor(img)
    img_tensor = normalize(img_tensor)
    return torch.autograd.Variable(img_tensor.unsqueeze(0).cuda(async=True))


def bbox_vote(det):
    order = det[:, 5].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 3] - det[:, 1] + 1) * (det[:, 4] - det[:, 2] + 1)
        xx1 = np.maximum(det[0, 1], det[:, 1])
        yy1 = np.maximum(det[0, 2], det[:, 2])
        xx2 = np.minimum(det[0, 3], det[:, 3])
        yy2 = np.minimum(det[0, 4], det[:, 4])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.5)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 1:5] = det_accu[:, 1:5] * np.tile(det_accu[:, -2:-1], (1, 4))
        max_score = np.max(det_accu[:, 5])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 1:5], axis=0) / np.sum(det_accu[:, -2:-1])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets


if __name__ == '__main__':
    main()


