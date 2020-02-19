from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import PIL
import cv2
import torchvision
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import uuid
#from model.utils.config import cfg

class Wider_face(Dataset):
    def __init__(self, img_root, list_path, width, height, transform, test=False):
        super(Wider_face, self).__init__()
        """
        WIDER Face data loader
        """
        self.img_list = list()
        self.label = list()
        self.img_root = img_root
        self.list_path = list_path
        self.width = width #Should I resize?
        self.height = height
        self.transform = transform
        self.test = test
        self.parse(self.img_root, self.list_path)
        print('Data loading finishes, total {} images.'.format(len(self.img_list)))
        
    def parse(self, img_root, list_path):
        file = open(list_path, encoding='utf-8')
        lines = file.readlines()
        idx = 0
        assert lines[0].strip().endswith('.jpg')
        while idx < len(lines):
            self.img_list.append(osp.join(img_root, lines[idx].strip())) # add img
            idx += 1
            num_boxes = int(lines[idx].strip()) # number of boxes
            idx += 1
            temp_boxes = []
            while idx < len(lines) and lines[idx].strip().endswith('.jpg') == False:
                box = lines[idx].strip().split(' ')[:4]
                temp_boxes.append(box)
                idx += 1
            if num_boxes != len(temp_boxes):
                print('Warning: number of boxes does not match box labels for image')
            self.label.append(temp_boxes)
        file.close()

    def __getitem__(self, index):
        """
        box = [x1, y1, w, h] in string format, fresh from the label file.
        """
        img = PIL.Image.open(self.img_list[index])
        imw, imh = img.size
        img = np.array(img)
        img = cv2.resize(img, (self.width, self.height))
        img = torchvision.transforms.ToPILImage()(img)

        # process box
        boxes = self.label[index]
        for idx, box in enumerate(boxes):
            x1 = min(max(0, int(box[0])), imw - 1)
            y1 = min(max(0, int(box[1])), imh - 1)
            w = abs(int(box[2]))
            h = abs(int(box[3]))
            x2 = min(max(x1 + w, 0), imw - 1)
            y2 = min(max(y1 + h, 0), imh - 1)
            boxes[idx] = [(x1/imw)*self.width, (y1/imh)*self.height, (x2/imw)*self.width\
                          , (y2/imh)*self.height] # normalize, so its invariant to input resize
        return self.transform(img), boxes, (self.width, self.height) # image, boxes, img_info

    def __len__(self):
        return len(self.img_list)
    
    def collate_fn(self, batch):
        images = list()
        boxes = list()
        sizes = list()
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            sizes.append(b[2])
        images = torch.stack(images, dim=0)
        return images, boxes, sizes