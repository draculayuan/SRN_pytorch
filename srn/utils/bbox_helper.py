#encoding: utf-8
import torch
import numpy as np
import warnings

def bbox_iou_overlaps(b1, b2):
    '''
    :argument
        b1,b2: [n, k], k>=4, x1,y1,x2,y2,...
    :returns
        intersection-over-union pair-wise.
    '''
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
    inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
    inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
    inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
    inter_h = np.maximum(inter_xmax - inter_xmin, 0)
    inter_w = np.maximum(inter_ymax - inter_ymin, 0)
    inter_area = inter_h * inter_w
    union_area1 = area1.reshape(-1, 1) + area2.reshape(1, -1)
    union_area2 = (union_area1 - inter_area)
    return inter_area / np.maximum(union_area2, 1)
    

def center_to_corner(boxes):
    '''
    :argument
        boxes: [N, 4] of center_x, center_y, w, h
    :returns
        boxes: [N, 4] of xmin, ymin, xmax, ymax
    '''
    xmin = boxes[:, 0] - boxes[:, 2] / 2.
    ymin = boxes[:, 1] - boxes[:, 3] / 2.
    xmax = boxes[:, 0] + boxes[:, 2] / 2.
    ymax = boxes[:, 1] + boxes[:, 3] / 2.
    return torch.cat([xmin.unsqueeze(1), ymin.unsqueeze(1), xmax.unsqueeze(1), ymax.unsqueeze(1)], dim=1)


def corner_to_center(boxes):
    '''
        inverse of center_to_corner
    '''
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.
    w = (boxes[:, 2] - boxes[:, 0])
    h = (boxes[:, 3] - boxes[:, 1])
    return torch.cat([cx.unsqueeze(1), cy.unsqueeze(1), w.unsqueeze(1), h.unsqueeze(1)], dim=1)


def compute_loc_bboxes(raw_bboxes, deltas):
    '''
    :argument
        raw_bboxes, delta:[N, k] first dim must be equal
    :returns
        bboxes:[N, 4]
    '''
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")
        bb = corner_to_center(raw_bboxes) # cx, cy, w, h
        dt_cx = deltas[:, 0] * bb[:, 2] + bb[:, 0]
        dt_cy = deltas[:, 1] * bb[:, 3] + bb[:, 1]
        dt_w = torch.exp(deltas[:, 2]) * bb[:, 2]
        dt_h = torch.exp(deltas[:, 3]) * bb[:, 3]
        dt = torch.cat([dt_cx.unsqueeze(1), dt_cy.unsqueeze(1), dt_w.unsqueeze(1), dt_h.unsqueeze(1)], dim=1)
        #np.vstack([dt_cx, dt_cy, dt_w, dt_h]).transpose()
        return center_to_corner(dt)


def clip_bbox(bbox, img_size):
    h, w = img_size[:2]
    bbox[:, 0] = np.clip(bbox[:, 0], 0, w - 1)
    bbox[:, 1] = np.clip(bbox[:, 1], 0, h - 1)
    bbox[:, 2] = np.clip(bbox[:, 2], 0, w - 1)
    bbox[:, 3] = np.clip(bbox[:, 3], 0, h - 1)
    return bbox

def match(threshold, truths, priors):
    '''
    truths: [num_gt, 4]
    priors: [num_priors, 4]
    loc_t (To be filled) : [num_priors, 4]
    conf_t (To be filled) : [num_priors]
    '''
    overlaps = torch.Tensor(bbox_iou_overlaps(truths.cpu().numpy(), center_to_corner(priors).cpu().numpy())) # return shape [num_gt, num_priors]
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx = best_truth_idx.squeeze(0)
    best_truth_overlap = best_truth_overlap.squeeze(0)
    best_prior_idx = best_prior_idx.squeeze(1)
    best_prior_overlap = best_prior_overlap.squeeze(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    loc_t = truths[best_truth_idx].cuda()
    
    conf_t = torch.ones(priors.size(0)).long().cuda()
    conf_t[best_truth_overlap < threshold] = 0

    return loc_t, conf_t
    
    
    
    
