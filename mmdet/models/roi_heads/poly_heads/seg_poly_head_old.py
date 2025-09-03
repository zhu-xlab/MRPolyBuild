# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import scipy
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, ModuleList
from mmengine.structures import InstanceData, PixelData
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.mask import mask_target, poly_target, PolygonMasks
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox_overlaps
from mmdet.structures.det_data_sample import DetDataSample
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
import mmdet.utils.tanmlh_polygon_utils as polygon_utils
from mmdet.utils import tanmlh_utils
from mmdet.models.layers import Mask2FormerTransformerDecoder, SinePositionalEncoding

import shapely

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
#  determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


@MODELS.register_module()
class SegPolyHead(BaseModule):

    def __init__(self,
                 poly_head: ConfigType = None,
                 seg2ins_head: ConfigType = None,
                 poly_cfg: OptMultiConfig = {},
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.poly_cfg = poly_cfg

        self.poly_head = None
        if poly_head is not None:
            self.poly_head = MODELS.build(poly_head)

        self.seg2ins_head = None
        if seg2ins_head is not None:
            self.seg2ins_head = MODELS.build(seg2ins_head)

    def forward(self, x: Tensor) -> Tensor:

        return x

    def loss(self, imgs, seg_logits: Tensor, batch_data_samples: SampleList) -> dict:

        losses = {}
        B, C, H, W = seg_logits.shape

        sem_seg_probs = F.softmax(seg_logits, dim=1)
        sem_seg_mask = sem_seg_probs[:,1] > self.poly_cfg.get('sem_seg_thr', 0.5)
        mask_feats = torch.cat([sem_seg_probs, imgs], dim=1)

        batch_pred_polys = []
        import time
        t0 = time.time()
        for i in range(B):
            sem_seg_polys, _ = polygon_utils.polygonize_mask(
                sem_seg_mask[i].cpu().to(torch.uint8), mode='simple_mask_cv2', scale=1.
            )
            batch_pred_polys.append(sem_seg_polys)

        # print(f'polygonization time: {time.time() - t0}')

        if self.poly_cfg.get('train_seg2ins_head', False):
            losses_seg2ins = self.seg2ins_head.loss(mask_feats, batch_pred_polys, batch_data_samples)
            losses.update(losses_seg2ins)

        if self.poly_cfg.get('train_poly_head', False):
            losses_poly_head = self._cal_loss_poly_head(
                seg_logits, batch_pred_polys, batch_data_samples
            )
            losses.update(losses_poly_head)

        return losses

    def _cal_loss_poly_head(self, seg_logits, batch_pred_polys, batch_data_samples):
        B, C, H, W = seg_logits.shape
        bbox_iou_thr = self.poly_cfg.get('bbox_iou_thr', 0.5)
        poly_iou_thr = self.poly_cfg.get('poly_iou_thr', 0.5)
        num_max_sample = self.poly_cfg.get('num_max_sample', 200)

        sem_seg_probs = F.softmax(seg_logits, dim=1)

        matched_pred_polys = []
        matched_gt_polys = []
        batch_idxes = []
        for i in range(B):
            cur_pred_polys = PolygonMasks.from_json(batch_pred_polys[i], H, W)
            cur_gt_polys = batch_data_samples[i].gt_instances.masks

            """
            Calculate matched pred and gt polygons
            """

            if len(cur_pred_polys) > 0 and len(cur_gt_polys) > 0:

                # iou_mat = tanmlh_utils.poly_overlaps(
                #     cur_pred_polys, cur_gt_polys, bbox_iou_thr=bbox_iou_thr,
                #     iou_type='fast_iou'
                # )
                # iou_mat = torch.tensor(iou_mat)
                # iou_mat = bbox_overlaps(
                #     torch.tensor(cur_pred_polys.get_bounds(), device=seg_logits.device),
                #     torch.tensor(cur_gt_polys.get_bounds(), device=seg_logits.device)
                # ).cpu()
                iou_mat = bbox_overlaps(
                    torch.tensor(cur_pred_polys.get_bounds()),
                    torch.tensor(cur_gt_polys.get_bounds())
                ).cpu()
                # torch.tensor(cur_pred_polys.get_bounds()),
                # torch.tensor(cur_gt_polys.get_bounds())
                # iou_mat = torch.zeros(len(cur_pred_polys), len(cur_gt_polys))

                max_values, max_idxes = iou_mat.max(dim=0)
                gt_idxes = (max_values > poly_iou_thr).nonzero().flatten().numpy()
                pred_idxes = max_idxes[gt_idxes].numpy()

                # if len(gt_idxes) > 0:
                cur_matched_pred_polys = cur_pred_polys[pred_idxes]
                cur_matched_gt_polys = cur_gt_polys[gt_idxes]
                cur_batch_idxes = [i] * len(pred_idxes)
            else:
                cur_matched_pred_polys = cur_pred_polys[:0]
                cur_matched_gt_polys = cur_gt_polys[:0]
                cur_batch_idxes = []

            matched_pred_polys.append(cur_matched_pred_polys)
            matched_gt_polys.append(cur_matched_gt_polys)
            batch_idxes.extend(cur_batch_idxes)

        matched_pred_polys = PolygonMasks.cat(matched_pred_polys)
        matched_gt_polys = PolygonMasks.cat(matched_gt_polys)
        batch_idxes = torch.tensor(batch_idxes)
        mask_feats = sem_seg_probs

        if len(matched_pred_polys) > num_max_sample:
            sample_idxes = np.random.permutation(len(matched_pred_polys))[:num_max_sample]
            matched_pred_polys = matched_pred_polys[sample_idxes]
            matched_gt_polys = matched_gt_polys[sample_idxes]
            batch_idxes = batch_idxes[sample_idxes]

        losses = self.poly_head.loss(
            matched_pred_polys.to_json(),
            matched_gt_polys.to_json(),
            W, device=seg_logits.device,
            mask_feat=mask_feats,
            batch_idxes=batch_idxes
        )

        return losses

    def predict(self, imgs, seg_logits, batch_data_samples, **kwargs):
        B, C, H, W = seg_logits.shape
        assert B == 1
        sem_seg_thr = self.poly_cfg.get('sem_seg_thr', 0.5)

        seg_probs = F.softmax(seg_logits, dim=1)
        # mask_feats = seg_probs
        mask_feats = torch.cat([seg_probs, imgs], dim=1)
        seg_probs = seg_probs[0,1:]

        pdb.set_trace()

        seg_mask = (seg_probs > sem_seg_thr).long()
        pixel_data = PixelData(sem_seg=seg_mask)

        results = batch_data_samples
        results[0].pred_sem_seg = pixel_data
        results[0].pred_sem_seg_prob = seg_probs

        pred_sem_seg = seg_mask.cpu().numpy()[0]

        labeled_mask, num_components = scipy.ndimage.label(pred_sem_seg)
        flat_labels = labeled_mask.ravel()
        flat_prob = seg_probs.ravel().cpu().numpy()
        sum_prob = np.bincount(flat_labels, weights=flat_prob)
        pixel_counts = np.bincount(flat_labels)
        scores = sum_prob[1:] / pixel_counts[1:]


        sem_seg_polys, colors = polygon_utils.polygonize_mask(
            torch.tensor(labeled_mask).to(torch.int), mode='simple_mask_cv2', scale=1.
        )

        scores = torch.tensor([scores[x-1] for x in colors])

        if self.seg2ins_head is not None:
            sem_seg_polys, scores = self.seg2ins_head.predict(mask_feats, sem_seg_polys, batch_data_samples)

        elif self.poly_head is not None:

            pred_results = self.poly_head.predict(
                sem_seg_polys, W, mask_feats, torch.zeros(len(sem_seg_polys), dtype=torch.long),
                device=seg_logits.device, return_format='json'
            )
            sem_seg_polys = pred_results['simp_polygons']

        seg_instances = InstanceData(segmentations=sem_seg_polys, scores=scores)
        seg_instances.polygon_masks = PolygonMasks.from_json(sem_seg_polys, H, W)
        seg_instances.bboxes = torch.tensor(seg_instances.polygon_masks.get_bounds())
        seg_instances.labels = torch.zeros(len(seg_instances), dtype=torch.long, device=seg_logits.device)

        results[0].pred_instances = seg_instances

        return results


    def _predict_by_feat_single(self,
                                poly_preds: Tensor,
                                bboxes: Tensor,
                                labels: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: ConfigDict,
                                rescale: bool = False,
                                activate_map: bool = False) -> Tensor:

        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        # img_h, img_w = img_meta['ori_shape'][:2]
        img_h, img_w = img_meta['batch_input_shape'][:2]
        device = bboxes.device

        if rescale:  # in-placed rescale the bboxes
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)

        N = len(poly_preds)

        poly_masks = PolygonMasks([[x.cpu().numpy().reshape(-1)] for x in poly_preds], img_h, img_w)
        poly_masks = poly_masks.paste_by_bboxes(bboxes.cpu().numpy(), self.mask_size, self.mask_size)

        return poly_masks

