import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils
import utils.boxes as box_utils

import numpy as np

class fast_rcnn_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.weak_supervise = cfg.TRAIN.WEAK_SUPERVISE
        self.weak_supervise_with_pretrain = cfg.TRAIN.WEAK_SUPERVISE_WITH_PRETRAIN
        self.copy_cls_to_det = cfg.TRAIN.COPY_CLS_TO_DET
        self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        if self.weak_supervise:
            self.det_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)

        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:  # bg and fg
            self.bbox_pred = nn.Linear(dim_in, 4 * 2)
        else:
            self.bbox_pred = nn.Linear(dim_in, 4 * cfg.MODEL.NUM_CLASSES)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)

        if self.weak_supervise:
            init.normal_(self.det_score.weight, std=0.01)
            init.constant_(self.det_score.bias, 0)

        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        if not self.weak_supervise:
            detectron_weight_mapping = {
                'cls_score.weight': 'cls_score_w',
                'cls_score.bias': 'cls_score_b',
                'bbox_pred.weight': 'bbox_pred_w',
                'bbox_pred.bias': 'bbox_pred_b'
            }
        elif self.weak_supervise_with_pretrain and self.copy_cls_to_det:
            # initialize det weight as the same of pretrained cls
            detectron_weight_mapping = {
                'cls_score.weight': 'cls_score_w',
                'cls_score.bias': 'cls_score_b',
                'det_score.weight': 'cls_score_w',
                'det_score.bias': 'cls_score_b',                
                'bbox_pred.weight': 'bbox_pred_w',
                'bbox_pred.bias': 'bbox_pred_b'
            }
        elif self.weak_supervise_with_pretrain and not self.copy_cls_to_det:
            # initialize det weight as the same of pretrained cls
            detectron_weight_mapping = {
                'cls_score.weight': 'cls_score_w',
                'cls_score.bias': 'cls_score_b',
                'det_score.weight': None,
                'det_score.bias': None,                
                'bbox_pred.weight': 'bbox_pred_w',
                'bbox_pred.bias': 'bbox_pred_b'
            }
        else:
            # initialize det weight with det
            detectron_weight_mapping = {
                'cls_score.weight': 'cls_score_w',
                'cls_score.bias': 'cls_score_b',
                'det_score.weight': 'det_score_w',
                'det_score.bias': 'det_score_b',                
                'bbox_pred.weight': 'bbox_pred_w',
                'bbox_pred.bias': 'bbox_pred_b'
            }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cls_score = self.cls_score(x)
        if not self.training:
            cls_score = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)

        if not self.weak_supervise:
            return cls_score, bbox_pred
        else:
            det_score = self.det_score(x)
            return cls_score, det_score, bbox_pred

def image_level_loss(cls_score, det_score, rois, image_labels_vec, bceloss, box_feat):
    device_id = cls_score.get_device()

    assert device_id == det_score.get_device()

    rois_batch_idx = rois[:, 0]
    batch_idx_list = np.unique(rois_batch_idx).astype('int32').tolist()
    rois_batch_idx = torch.from_numpy(rois_batch_idx).cuda(device_id)
    print(f"rois_batch_idx: shape {rois_batch_idx.shape}\n {rois_batch_idx}")
    print(f"box_feat: shape {box_feat.shape}\n {box_feat}")

    # print(f"image_labels_vec: shape {image_labels_vec.shape}\n {image_labels_vec}")
    image_labels = Variable(torch.from_numpy(image_labels_vec.astype('float32'))).cuda(device_id)
    # exclude background class
    image_labels = image_labels[:, 1:]

    print(f"batch_idx_list: {batch_idx_list}")
    cls_probs = None
    reg = None
    assert len(batch_idx_list) == image_labels_vec.shape[0]
    for idx in batch_idx_list:
        ind = (rois_batch_idx == idx).nonzero().squeeze()
        # print(ind.shape, ind[-1])
        cls_ind = torch.index_select(cls_score, 0, ind)
        det_ind = torch.index_select(det_score, 0, ind)

        # exclude background
        softmax_cls = F.softmax(cls_ind[:, 1:], dim=1)
        softmax_det = F.softmax(det_ind[:, 1:], dim=0)
        roi_cls_scores = softmax_cls * softmax_det

        if cfg.TRAIN.SPATIAL_REG:
            # print(f"image_labels: shape {image_labels.shape}\n {image_labels}")
            # print(f"image_labels_vec: shape {image_labels_vec.shape}\n {image_labels_vec}")
            # print(f"image_labels[idx]: shape {image_labels[idx].shape}\n {image_labels[idx]}")
            # print(f"image_labels_vec[idx]: shape {image_labels_vec[idx].shape}\n {image_labels_vec[idx]}")
            
            # # find positive classes for one image
            # gt_classes_ind = (image_labels[idx].detach() == 1).nonzero()
            # print(f"no squeeze gt_classes_ind: shape {gt_classes_ind.shape}\n {gt_classes_ind}")
            gt_classes_ind = (image_labels[idx].detach() == 1).nonzero().squeeze(dim=1)
            # print(f"gt_classes_ind: shape {gt_classes_ind.shape}\n {gt_classes_ind}")

            roi_pos_cls_scores = roi_cls_scores[:, gt_classes_ind]
            max_roi_pos_cls_scores_ind = torch.argmax(roi_pos_cls_scores, dim=0)
            max_roi_pos_cls_scores = torch.max(roi_pos_cls_scores, dim=0)

            # print(f"max ind before \n {max_roi_pos_cls_scores_ind}")
            max_roi_pos_cls_scores_ind = ind[max_roi_pos_cls_scores_ind]
            # print(f"max ind after \n {max_roi_pos_cls_scores_ind}")
            # bugs on indexing max_roi_pos_cls_scores_ind is torch.Size(1)
            roi_max_in_cls = rois[max_roi_pos_cls_scores_ind.cpu().numpy(), 1:5]
            roi_in_one_image = rois[ind, 1:5]
            # print(f"roi_max_in_cls \n {roi_max_in_cls}")
            # print(f"roi_in_one_image \n {roi_in_one_image}")

            roi_overlaps_with_max = box_utils.bbox_overlaps(
                roi_in_one_image.astype(dtype=np.float32, copy=False),
                roi_max_in_cls.astype(dtype=np.float32, copy=False)
            )
            # print(f"roi_overlaps_with_max shape \n {roi_overlaps_with_max.shape}")
            pos_roi_overlaps_with_max_ind = np.where(roi_overlaps_with_max > 0.6)
            # print(f"index of roi_overlaps_with_max > 0.6 \n {pos_roi_overlaps_with_max_ind}")
            # print(f"roi_overlaps_with_max > 0.6 \n {roi_overlaps_with_max[pos_roi_overlaps_with_max_ind]}")
            selected_overlap_roi_ind  = ind[pos_roi_overlaps_with_max_ind[0]]
            max_roi_ind = max_roi_pos_cls_scores_ind[pos_roi_overlaps_with_max_ind[1]]
            print(f"pos_roi_overlaps_with_max_ind[1] : {pos_roi_overlaps_with_max_ind[1]}")
            print(f"max_roi_pos_cls_scores_ind : {max_roi_pos_cls_scores_ind}")
            print(f"max_roi_pos_cls_scores : {max_roi_pos_cls_scores}")

            max_roi_scores = max_roi_pos_cls_scores[pos_roi_overlaps_with_max_ind[1]]
            # print(f"box_feat selected shape: {box_feat[selected_overlap_roi_ind].shape}")
            print(f"box_feat corresbonding ind: {max_roi_ind}")
            print(f"box_feat corresbonding scores: {max_roi_scores}")
            diff_box_feat = (box_feat[selected_overlap_roi_ind] - box_feat[max_roi_ind])
            # print(torch.sum(diff_box_feat * diff_box_feat))
            # weighted spatial regularization
            if reg is None:
                reg = torch.sum(max_roi_scores * torch.sum(diff_box_feat * diff_box_feat, dim=1), keepdim=True)/ image_labels.shape[1]
            else:
                reg_new = torch.sum(max_roi_scores * torch.sum(diff_box_feat * diff_box_feat, dim=1), keepdim=True)/ image_labels.shape[1]
                reg = torch.cat((reg, reg_new), dim=0)

            print(f"reg shape: {reg.shape}\n {reg}")



        cls_prob = torch.sum(roi_cls_scores, dim=0)
        if cls_probs is None:
            cls_probs = cls_prob.unsqueeze(dim=0)
        else:
            cls_probs = torch.cat((cls_probs, cls_prob.unsqueeze(dim=0)), dim=0)
        print(f"cls_probs shape: {cls_probs.shape}\n {cls_probs}")

    
    # spatial regularization
    reg = reg / len(batch_idx_list)

    # if cls_probs.max() > 1 or cls_probs.min() < 0:
    #     print(f"cls probs : {cls_probs}\n shape : {cls_probs.shape}")
    # print(f"softmax_cls shape: {softmax_cls.shape} sum over dim 1 {softmax_cls.sum(dim=1)}\
    # \n softmax_det shape: {softmax_det.shape} sum over dim 0 {softmax_det.sum(dim=0)}")
    loss_cls = bceloss(cls_probs.clamp(0,1), image_labels)

    # multi label class accuracy
    acc_score = cls_probs.round().eq(image_labels).float().mean()
    # print(f"ap_score: shape {ap_score.shape}\n {ap_score}")
    return loss_cls, acc_score, reg

def fast_rcnn_losses(cls_score, bbox_pred, label_int32, bbox_targets,
                     bbox_inside_weights, bbox_outside_weights):
    device_id = cls_score.get_device()
    rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)
    loss_cls = F.cross_entropy(cls_score, rois_label)

    bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda(device_id)
    bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda(device_id)
    bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda(device_id)
    loss_bbox = net_utils.smooth_l1_loss(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    # class accuracy
    cls_preds = cls_score.max(dim=1)[1].type_as(rois_label)
    accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)

    return loss_cls, loss_bbox, accuracy_cls


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

class roi_2mlp_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x


class roi_Xconv1fc_head(nn.Module):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*2): 'head_conv%d_w' % (i+1),
                'convs.%d.bias' % (i*2): 'head_conv%d_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x


class roi_Xconv1fc_gn_head(nn.Module):
    """Add a X conv + 1fc head, with GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                             eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*3): 'head_conv%d_w' % (i+1),
                'convs.%d.weight' % (i*3+1): 'head_conv%d_gn_s' % (i+1),
                'convs.%d.bias' % (i*3+1): 'head_conv%d_gn_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x
