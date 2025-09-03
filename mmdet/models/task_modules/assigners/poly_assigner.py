# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_overlaps

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from mmdet.structures.mask import mask_target, poly_target, PolygonMasks


@TASK_UTILS.register_module()
class PolyAssigner(BaseAssigner):

    def __init__(self, cfg={}) -> None:
        self.cfg = cfg

    def assign(self, pred_polys: PolygonMasks, gt_polys: PolygonMasks, gt_labels, device='cpu'):

        matched_pred_polys = []
        matched_gt_polys = []
        matched_gt_labels = []

        iou_mat = bbox_overlaps(
            torch.tensor(gt_polys.get_bounds(), device=device),
            torch.tensor(pred_polys.get_bounds(), device=device),
            mode='iof'
        ).cpu()

        iou_thr = self.cfg.get('iou_thr', 0.0)
        max_values, gt2pred_idxes = iou_mat.max(dim=1)
        for pred_id in range(len(pred_polys)):
            gt_idxes = ((gt2pred_idxes == pred_id) & (max_values > iou_thr)).nonzero().flatten()
            cur_matched_gt_polys = gt_polys[gt_idxes.numpy()]

            # cropped_gt_polys = []
            # inter_idxes = []
            # for j in range(len(cur_matched_gt_polys)):
            #     cropped_poly = cur_matched_gt_polys[j].intersect(pred_polys[pred_id])
            #     cropped_gt_polys.append(cropped_poly)
            #     if len(cropped_poly) > 0:
            #         inter_idxes.append(j)
            # if len(cropped_gt_polys) > 0:
            #     cur_matched_gt_polys = PolygonMasks.cat(cropped_gt_polys)
            # else:
            #     cur_matched_gt_polys = gt_polys[:0]
            # cur_matched_gt_labels = gt_labels[gt_idxes[torch.tensor(inter_idxes, dtype=torch.long)]]

            cur_matched_gt_labels = gt_labels[gt_idxes]

            if len(cur_matched_gt_polys) > 0:
                matched_pred_polys.append(pred_polys[pred_id])
                matched_gt_polys.append(cur_matched_gt_polys)
                matched_gt_labels.append(cur_matched_gt_labels)

        assign_result = dict(
            matched_pred_polys = matched_pred_polys,
            matched_gt_polys = matched_gt_polys,
            matched_gt_labels = matched_gt_labels,
        )
        return assign_result
