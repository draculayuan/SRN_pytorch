import torch
import torch.nn as nn
import torch.nn.functional as F
from .bbox_helper import match
import numpy as np

class SRN_criterion(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, overlap_thresh):
        super(SRN_criterion, self).__init__()
        self.threshold = overlap_thresh

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tensor of size (num_proposals_in_whole_batch, 10)
        the 10 consists of bidx, x1, x2, x3, x4, score_0, score_1, ax1, ax2, ax3, ax4
            targets (tensor): Ground truth boxes for a batch, it is a list of list.
        Each element in the outer list corresponding to batch. Each element in the inner list
        corresponding to all boxes in that image.
        """
        loc_data, conf_data, priors = predictions[:, 1:5], predictions[:, 5:7], predictions[:, -4:]
        #print(loc_data.shape, conf_data.shape, priors.shape)
        num = len(targets) #batch size
        #priors = priors[:loc_data.size(1), :]
        #num_priors = (priors.size(0))
        #num_classes = self.num_classes
        
        loss_l = 0
        loss_c = 0
        cnt_l = 0
        cnt_c = 0
        
        for idx in range(num):
            # match by batch
            num_priors = predictions[predictions[:, 0] == idx].shape[0]
            #print(num_priors)
            truths = torch.Tensor(targets[idx]).cuda()
            defaults = priors[predictions[:, 0] == idx]
            loc_t, conf_t = match(self.threshold, truths, defaults.detach()) # need to edit the code to avoid idx effect
            
            pos = conf_t > 0
            num_pos = pos.sum(dim=0)
            
            # loc loss (smooth l1)
            # do you have to do pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)??
            loc_p = loc_data[predictions[:, 0] == idx][pos] #.view(-1, 4)
            loc_t = loc_t[pos] #.view(-1, 4)
            loss_l += F.smooth_l1_loss(loc_p, loc_t, reduction='sum') # what is size_average?
            cnt_l += loc_p.size(0)
            # conf loss (focal, but im using softmax for the sake of convenience)
            conf_p = conf_data[predictions[:, 0] == idx]
            loss_c += F.cross_entropy(conf_p, conf_t)
            cnt_c += conf_p.size(0)

        loss_l /= cnt_l
        loss_c /= cnt_c
        return loss_l + loss_c