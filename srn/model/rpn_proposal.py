#encoding: utf-8
from srn.utils import bbox_helper
from srn.utils import anchor_helper
import torch
from torch.autograd import Variable
import numpy as np
import logging

logger = logging.getLogger('global')

def to_np_array(x):
    if isinstance(x, Variable): x = x.data
    return x.cpu().numpy() if torch.is_tensor(x) else x

def compute_rpn_proposals(conv_cls_fs, conv_loc_fs, conv_cls_ss, conv_loc_ss,
                          multi_cls, multi_reg, cfg, image_info, test=False):
    '''
    :argument
        cfg: configs
        conv_cls: FloatTensor, [batch, num_anchors * num_classes, h, w], conv output of classification
        conv_loc: FloatTensor, [batch, num_anchors * 4, h, w], conv output of localization
        image_info: FloatTensor, [batch, 3], image size
    :returns
        proposals: Variable, [N, 7], 2-dim: batch_ix, x1, y1, x2, y2, score, label
    '''
    batch_size, num_anchors_num_classes, featmap_h, featmap_w = conv_cls_fs.shape
    # [K*A, 4]
    anchors_overplane = anchor_helper.get_anchors_over_plane(
        featmap_h, featmap_w, cfg['anchor_ratios'], cfg['anchor_scales'], cfg['anchor_stride'])
    anchors_overplane = torch.Tensor(anchors_overplane).cuda()

    B = batch_size
    A = num_anchors_num_classes // cfg['num_classes']
    assert(A * cfg['num_classes'] == num_anchors_num_classes)
    K = featmap_h * featmap_w
    cls_view_fs = conv_cls_fs.permute(0, 2, 3, 1).contiguous().view(B, K*A, cfg['num_classes'])#.cpu().numpy()
    loc_view_fs = conv_loc_fs.permute(0, 2, 3, 1).contiguous().view(B, K*A, 4)#.cpu().numpy()
    cls_view_ss = conv_cls_ss.permute(0, 2, 3, 1).contiguous().view(B, K * A, cfg['num_classes'])#.cpu().numpy()
    loc_view_ss = conv_loc_ss.permute(0, 2, 3, 1).contiguous().view(B, K * A, 4)#.cpu().numpy()
    #print(type(loc_view_ss))
    if cfg['cls_loss_type'] == 'softmax_focal_loss':
        cls_view_fs = cls_view_fs[:, :, 1:]
        cls_view_ss = cls_view_ss[:, :, 1:]
    nmsed_bboxes = []
    pre_nms_top_n = cfg['top_n_per_level']
    thresh = cfg['score_thresh'] if K >= 120 else 0.0
    for b_ix in range(B):
        anchors_fs = anchors_overplane # for training
        loc_delta_fs = loc_view_fs[b_ix, :, :]
        #print('inside rpn proposal, {}'.format(loc_delta_fs.requires_grad))
        if multi_reg:
            anchors_overplane = bbox_helper.compute_loc_bboxes(anchors_overplane, loc_delta_fs.data)
            # here it should not have gradient cuz its anchors
        anchors_ss = anchors_overplane # for training
        
        ka_ix_fs, cls_ix_fs = np.where(cls_view_fs[b_ix] > 0.01)
        ka_ix_ss, cls_ix_ss = np.where(cls_view_ss[b_ix] > thresh)
        if len(ka_ix_ss) == 0:
            ka_ix_ss, cls_ix_ss = np.where(cls_view_ss[b_ix] > 0.01)
        if not test:
            # for training
            anchors_fs = anchors_fs[ka_ix_fs, :]
            anchors_ss = anchors_ss[ka_ix_ss, :]
            scores_fs = cls_view_fs[b_ix, ka_ix_fs,:]
            scores_ss = cls_view_fs[b_ix, ka_ix_ss,:]
            loc_delta_fs = loc_view_fs[b_ix, ka_ix_fs, :]
            loc_delta_ss = loc_view_ss[b_ix, ka_ix_ss, :]
            boxes_fs = bbox_helper.compute_loc_bboxes(anchors_fs, loc_delta_fs)
            boxes_ss = bbox_helper.compute_loc_bboxes(anchors_ss, loc_delta_ss)
            #print(type(boxes_fs), boxes_fs.requires_grad)
            # construct return tensor Variable, [N, 7], 2-dim: batch_ix, x1, y1, x2, y2, score_0, score_1, ax1, ay1, ax2, ay2
            batch_ix_fs = torch.Tensor(np.full(boxes_fs.shape[0], b_ix)).cuda()
            batch_ix_ss = torch.Tensor(np.full(boxes_ss.shape[0], b_ix)).cuda()
            #print(batch_ix_fs.shape, boxes_fs.shape, scores_fs.shape, anchors_fs.shape)
            post_bboxes_fs = torch.cat([batch_ix_fs.unsqueeze(1), boxes_fs, scores_fs, anchors_fs], dim=1)
            post_bboxes_ss = torch.cat([batch_ix_ss.unsqueeze(1), boxes_ss, scores_ss, anchors_ss], dim=1)
            nmsed_bboxes.append([post_bboxes_fs, post_bboxes_ss])
        else:
            if multi_cls:
                ka_ix = np.intersect1d(ka_ix_fs, ka_ix_ss)
            else:
                ka_ix = ka_ix_ss
            cls_ix = np.zeros_like(ka_ix)

            if ka_ix.size == 0:
                continue

            scores = cls_view_ss[b_ix, ka_ix, cls_ix]
            loc_delta_ss = loc_view_ss[b_ix, ka_ix, :]
            loc_anchors = anchors_overplane[ka_ix, :]

            # nms ranking only during inference?
            if True or pre_nms_top_n <= 0 or pre_nms_top_n > scores.shape[0]:
                scores = scores.cpu().numpy()
                order = scores.argsort()[::-1][:pre_nms_top_n]
            else:
                scores = scores.cpu().numpy()
                inds = np.argpartition(-scores, pre_nms_top_n)[:pre_nms_top_n]
                order = np.argsort(-scores[inds])
                order = inds[order]

            scores = scores[order]
            cls_ix = cls_ix[order]
            cls_ix = cls_ix + 1
            loc_delta = torch.Tensor(loc_delta_ss.cpu().numpy()[order])
            loc_anchors = torch.Tensor(loc_anchors.cpu().numpy()[order])

            boxes = bbox_helper.compute_loc_bboxes(loc_anchors, loc_delta).cpu().numpy()

            batch_ix = np.full(boxes.shape[0], b_ix)
            post_bboxes = np.hstack([batch_ix[:, np.newaxis], boxes, scores[:, np.newaxis], cls_ix[:, np.newaxis]])
            nmsed_bboxes.append(post_bboxes)
    if test:
        if len(nmsed_bboxes) > 0:    
            nmsed_bboxes = np.vstack(nmsed_bboxes)
        else:
            nmsed_bboxes = np.array([])
    else:
        nmsed_bboxes_fs = torch.Tensor([]).cuda()
        nmsed_bboxes_ss = torch.Tensor([]).cuda()
        for bidx_proposal in nmsed_bboxes:
            nmsed_bboxes_fs = torch.cat((nmsed_bboxes_fs, bidx_proposal[0]), dim=0)  
            nmsed_bboxes_ss = torch.cat((nmsed_bboxes_ss, bidx_proposal[1]), dim=0) 
        nmsed_bboxes = [nmsed_bboxes_fs, nmsed_bboxes_ss]
    return nmsed_bboxes # it is a tensor for inference, but it is a list of list for training
