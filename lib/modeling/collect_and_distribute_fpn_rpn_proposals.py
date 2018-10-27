import numpy as np
from torch import nn

from core.config import cfg
from datasets import json_dataset
import roi_data.fast_rcnn
import utils.blob as blob_utils
import utils.fpn as fpn_utils


class CollectAndDistributeFpnRpnProposalsOp(nn.Module):
    """Merge RPN proposals generated at multiple FPN levels and then
    distribute those proposals to their appropriate FPN levels. An anchor
    at one FPN level may predict an RoI that will map to another level,
    hence the need to redistribute the proposals.

    This function assumes standard blob names for input and output blobs.

    Input blobs: [rpn_rois_fpn<min>, ..., rpn_rois_fpn<max>,
                  rpn_roi_probs_fpn<min>, ..., rpn_roi_probs_fpn<max>]
        - rpn_rois_fpn<i> are the RPN proposals for FPN level i; see rpn_rois
        documentation from GenerateProposals.
        - rpn_roi_probs_fpn<i> are the RPN objectness probabilities for FPN
        level i; see rpn_roi_probs documentation from GenerateProposals.

    If used during training, then the input blobs will also include:
        [roidb, im_info] (see GenerateProposalLabels).

    Output blobs: [rois_fpn<min>, ..., rois_rpn<max>, rois,
                   rois_idx_restore]
        - rois_fpn<i> are the RPN proposals for FPN level i
        - rois_idx_restore is a permutation on the concatenation of all
        rois_fpn<i>, i=min...max, such that when applied the RPN RoIs are
        restored to their original order in the input blobs.

    If used during training, then the output blobs will also include:
        [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights].
    """
    def __init__(self):
        super().__init__()
        self.weak_supervise = cfg.TRAIN.WEAK_SUPERVISE

    def forward(self, inputs, roidb, im_info):
        """
        Args:
            inputs: a list of [rpn_rois_fpn2, ..., rpn_rois_fpn6,
                               rpn_roi_probs_fpn2, ..., rpn_roi_probs_fpn6]
            im_info: [[im_height, im_width, im_scale], ...]
        """
        rois = collect(inputs, self.training)
        if self.training and not self.weak_supervise:
            # During training we reuse the data loader code. We populate roidb
            # entries on the fly using the rois generated by RPN.
            im_scales = im_info.data.numpy()[:, 2]
            # For historical consistency with the original Faster R-CNN
            # implementation we are *not* filtering crowd proposals.
            # This choice should be investigated in the future (it likely does
            # not matter).
            json_dataset.add_proposals(roidb, rois, im_scales, crowd_thresh=0)
            # Compute training labels for the RPN proposals; also handles
            # distributing the proposals over FPN levels
            output_blob_names = roi_data.fast_rcnn.get_fast_rcnn_blob_names()
            blobs = {k: [] for k in output_blob_names}
            roi_data.fast_rcnn.add_fast_rcnn_blobs(blobs, im_scales, roidb)
        elif self.training and self.weak_supervise:
            # When using image-level annotations to do weak supervise
            # For inference we have a special code path that avoids some data
            # loader overhead
            label_blobs = []
            for entry in roidb:
                label_blobs.append(entry['gt_labels_vec'])

            label_blobs = np.concatenate(label_blobs)

            blobs = distribute(rois, label_blobs)
        else:
            # When testing
            # For inference we have a special code path that avoids some data
            # loader overhead
            blobs = distribute(rois, None)

        return blobs


def collect(inputs, is_training):
    cfg_key = 'TRAIN' if is_training else 'TEST'
    post_nms_topN = int(cfg[cfg_key].RPN_POST_NMS_TOP_N * cfg.FPN.RPN_COLLECT_SCALE + 0.5)
    k_max = cfg.FPN.RPN_MAX_LEVEL
    k_min = cfg.FPN.RPN_MIN_LEVEL
    num_lvls = k_max - k_min + 1
    roi_inputs = inputs[:num_lvls]
    score_inputs = inputs[num_lvls:]

    # rois are in [[batch_idx, x0, y0, x1, y2], ...] format
    # Combine predictions across all levels and retain the top scoring
    rois = np.concatenate(roi_inputs)
    print(f"Before sort RoIs: shape {rois.shape}\n {rois}")

    scores = np.concatenate(score_inputs).squeeze()
    inds = np.argsort(-scores)[:post_nms_topN]
    rois = rois[inds, :]
    print(f"After sort RoIs: shape {rois.shape}\n {rois}")
    return rois


def distribute(rois, label_blobs=None):
    """To understand the output blob order see return value of
    roi_data.fast_rcnn.get_fast_rcnn_blob_names(is_training=False)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn_utils.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)

    # Delete roi entries that have negative area
    # idx_neg = np.where(lvls == -1)[0]
    # rois = np.delete(rois, idx_neg, axis=0)
    # lvls = np.delete(lvls, idx_neg, axis=0)

    output_blob_names = roi_data.fast_rcnn.get_fast_rcnn_blob_names(is_training=False)
    outputs = [None] * len(output_blob_names)
    outputs[0] = rois

    # Create new roi blobs for each FPN level
    # (See: utils.fpn.add_multilevel_roi_blobs which is similar but annoying
    # to generalize to support this particular case.)
    rois_idx_order = np.empty((0, ))
    for output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)):
        idx_lvl = np.where(lvls == lvl)[0]
        blob_roi_level = rois[idx_lvl, :]
        outputs[output_idx + 1] = blob_roi_level
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
    rois_idx_restore = np.argsort(rois_idx_order)
    outputs[-2] = rois_idx_restore.astype(np.int32)
    outputs[-1] = label_blobs

    return dict(zip(output_blob_names, outputs))
