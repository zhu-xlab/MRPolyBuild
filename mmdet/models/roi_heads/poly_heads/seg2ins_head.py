# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple
import time

import shapely
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


@MODELS.register_module()
class Seg2InsHead(BaseModule):

    def __init__(self,
                 poly_roi_extractor: ConfigType,
                 ins_head: ConfigType = None,
                 poly_assigner: ConfigType = None,
                 poly_cfg: OptMultiConfig = {},
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.poly_roi_extractor = MODELS.build(poly_roi_extractor)
        self.ins_head = MODELS.build(ins_head)
        self.poly_assigner = TASK_UTILS.build(poly_assigner)
        self.poly_cfg = poly_cfg

    def loss(self, mask_feats, batch_pred_polys, batch_data_samples):
        import time
        t0 = time.time()
        B, C, H, W = mask_feats.shape
        bbox_iou_thr = self.poly_cfg.get('bbox_iou_thr', 0.5)
        poly_iou_thr = self.poly_cfg.get('poly_iou_thr', 0.5)
        num_max_sample = self.poly_cfg.get('num_max_sample', 200)
        mask_size = self.poly_roi_extractor.roi_layers[0].output_size
        bbox_buffer = self.poly_cfg.get('bbox_buffer', 2)

        t0 = time.time()

        matched_pred_polys = []
        matched_gt_polys = []
        matched_gt_labels = []
        batch_idxes = []
        for i in range(B):
            cur_pred_polys = PolygonMasks.from_json(batch_pred_polys[i], H, W)
            cur_gt_polys = batch_data_samples[i].gt_instances.masks
            cur_gt_labels = batch_data_samples[i].gt_instances.labels

            """
            Calculate matched pred and gt polygons
            """
            if len(cur_pred_polys) > 0 and len(cur_gt_polys) > 0:
                assign_result = self.poly_assigner.assign(
                    cur_pred_polys, cur_gt_polys, cur_gt_labels
                )
                matched_pred_polys.extend(assign_result['matched_pred_polys'])
                matched_gt_polys.extend(assign_result['matched_gt_polys'])
                matched_gt_labels.extend(assign_result['matched_gt_labels'])
                batch_idxes.extend([i] * len(assign_result['matched_pred_polys']))

        # print(f'{time.time() - t0}')

        if len(matched_pred_polys) <= 1:
            dummy_loss = self.ins_head.parameters().__next__()[:0].sum()
            losses_seg2ins = dict(
                loss_ce_ins=dummy_loss,
            )
            return losses_seg2ins

        t1 = time.time()

        matched_pred_polys = PolygonMasks.cat(matched_pred_polys)
        batch_idxes = torch.tensor(batch_idxes)

        num_max_sample = 200
        if len(matched_pred_polys) > num_max_sample:
            sample_idxes = np.random.permutation(len(matched_pred_polys))[:num_max_sample]
            matched_pred_polys = matched_pred_polys[sample_idxes]
            matched_gt_polys = [matched_gt_polys[_] for _ in sample_idxes]
            matched_gt_labels = [matched_gt_labels[_] for _ in sample_idxes]
            batch_idxes = batch_idxes[sample_idxes]

        bounds = matched_pred_polys.get_bounds(buffer=bbox_buffer)

        """
        ins_data_samples = []
        for i in range(len(matched_pred_polys)):
            gt_sem_seg = matched_gt_polys[i].crop(bounds[i]).resize(mask_size).merge().to_tensor(
                device=mask_feats.device, dtype=torch.long
            )
            pred_sem_seg = matched_pred_polys[i].crop(bounds[i]).resize(mask_size).to_tensor(
                device=mask_feats.device, dtype=torch.long
            )
            gt_sem_seg = PixelData(data=(gt_sem_seg*pred_sem_seg))
            ins_data_samples.append(
                DetDataSample(
                    metainfo=dict(
                        scale_factor=(1,1),
                        img_shape=mask_size,
                        ori_shape=mask_size,
                        batch_input_shape=mask_size
                    ),
                    gt_sem_seg=gt_sem_seg
                )
            )
        """

        t2 = time.time()
        rois = torch.tensor([(idx, *bound) for idx, bound in zip(batch_idxes, bounds)]).to(mask_feats.device)

        roi_levels = self.poly_cfg.get('roi_levels', [16, 64, 128])
        rois_list, idxes_list = self.separate_rois(rois, roi_levels)

        losses_ins2seg = {}
        for i, cur_rois in enumerate(rois_list):
            if len(cur_rois) >= 2:
                cur_rois = torch.stack(cur_rois)
                cur_idxes = idxes_list[i]
                cur_mask_size = (roi_levels[i], roi_levels[i])
                cur_gt_polys = [matched_gt_polys[j].crop(bounds[j]).resize(cur_mask_size).merge() for j in cur_idxes]
                # cur_gt_polys = cur_gt_polys.to_tensor(device=mask_feats.device, dtype=torch.long)
                # cur_ins_data_samples = [ins_data_samples[x] for x in idxes_list[i]]
                cur_gt_polys = PolygonMasks.cat(cur_gt_polys).to_tensor(device=mask_feats.device, dtype=torch.long)
                cur_ins_data_samples = [DetDataSample(
                    metainfo=dict(
                        scale_factor=(1,1),
                        img_shape=cur_mask_size,
                        ori_shape=cur_mask_size,
                        batch_input_shape=cur_mask_size
                    ),
                    gt_sem_seg=PixelData(data=cur_gt_polys[j:j+1])
                ) for j in range(len(cur_gt_polys))]

                roi_feats = self.poly_roi_extractor([mask_feats], cur_rois)

                seg_logits = self.ins_head.forward([roi_feats])
                if len(seg_logits) > 0:
                    losses = self.ins_head.loss_by_feat(seg_logits, cur_ins_data_samples)
                    for key, loss in losses.items():
                        losses_ins2seg[key + f'_ins_level{i}'] = losses[key]
                else:
                    losses_ins2seg[key + f'_ins_level{i}'] = seg_logits[:0].sum()

        t3 = time.time()
        print(f'{t2-t1} {t3-t2}')

        return losses_ins2seg

    @staticmethod
    def separate_rois(rois, roi_levels):
        """
        Separates ROIs into different levels based on maximum dimension (width/height)
        
        Args:
            rois: Tensor of shape (N, 5) where each row is [batch_id, x_min, y_min, x_max, y_max]
            roi_levels: List of level thresholds in original order (e.g., [16, 64, 256])
        
        Returns:
            Tuple of (rois_by_level, indices_by_level) where:
            - rois_by_level: List of ROI groups matching original roi_levels order
            - indices_by_level: List of original indices for each group
        """
        # Create mapping from level to original index
        level_to_idx = {level: i for i, level in enumerate(roi_levels)}
        sorted_levels = sorted(roi_levels)
        max_level = sorted_levels[-1]
        
        # Initialize empty lists for each level
        rois_by_level = [[] for _ in roi_levels]
        indices_by_level = [[] for _ in roi_levels]

        for idx, roi in enumerate(rois):
            # Extract coordinates and calculate dimensions
            x_min, y_min, x_max, y_max = roi[1], roi[2], roi[3], roi[4]
            width = x_max - x_min
            height = y_max - y_min
            max_dim = max(width, height)

            # Find appropriate level using early exit pattern
            selected_level = next((l for l in sorted_levels if l > max_dim), max_level)
            
            # Get original index from level mapping
            orig_idx = level_to_idx[selected_level]
            
            # Add to corresponding groups
            rois_by_level[orig_idx].append(roi)
            indices_by_level[orig_idx].append(idx)

        return rois_by_level, indices_by_level


    def predict(self, mask_feats, pred_polys, batch_data_samples):

        B, C, H, W = mask_feats.shape

        mask_size = self.poly_roi_extractor.roi_layers[0].output_size
        pred_polys = PolygonMasks.from_json(pred_polys, H, W)
        bounds = pred_polys.get_bounds()
        rois = torch.tensor([(0, *bound) for bound in bounds]).to(mask_feats.device)

        pdb.set_trace()


        if len(rois) > 0:
            roi_feats = self.poly_roi_extractor([mask_feats], rois)

            seg2ins_data_samples = [
                dict(
                    scale_factor=(1,1),
                    img_shape=mask_size,
                    ori_shape=mask_size,
                    batch_input_shape=mask_size
                ) for i in range(len(pred_polys))
            ]
            ins_masks = self.ins_head.predict([roi_feats], seg2ins_data_samples, None)
            ins_probs = F.softmax(ins_masks, dim=1)[:,1]

            # poly_masks = ins_masks.paste_by_bboxes(bounds, mask_size, mask_size)
            if len(ins_masks) > 0:
                pred_polys, colors = polygon_utils.polygonize_mask(
                    (ins_probs > 0.5).to(torch.int), mode='concat_mask_cv2', scale=1.
                )
                pred_polys = polygon_utils.paste_poly_json(pred_polys, rois[colors-1][:,1:].cpu(), mask_size[0], mask_size[1], H, W)
                scores = torch.zeros(len(pred_polys), device=mask_feats.device) + 0.5

                # labeled_mask, num_components = scipy.ndimage.label(pred_sem_seg)
                # flat_labels = labeled_mask.ravel()
                # flat_prob = seg_probs.ravel().cpu().numpy()
                # sum_prob = np.bincount(flat_labels, weights=flat_prob)
                # pixel_counts = np.bincount(flat_labels)
                # scores = sum_prob[1:] / pixel_counts[1:]

            else:
                pred_polys = []
                scores = torch.zeros(0, device=mask_feats.device)
        else:
            pred_polys = []
            scores = torch.zeros(0, device=mask_feats.device)

        return pred_polys, scores








        """
        seg2ins_data_samples = [DetDataSample(
            metainfo=dict(
                scale_factor=(1,1),
                img_shape=mask_size,
                ori_shape=mask_size,
                batch_input_shape=mask_size
            ),
            gt_instances=InstanceData(
                masks=matched_gt_polys[i].crop(bounds[i]).resize(mask_size),
                labels=matched_gt_labels[i]
                # masks=matched_pred_polys[i].crop(bounds[i]).resize(mask_size),
                # labels=matched_gt_labels[i][:1]
            )
        ) for i in range(len(matched_pred_polys))]
        lens = [len(x) for x in matched_gt_polys]








            mask_size = self.poly_roi_extractor.roi_layers[0].output_size
            pred_polys = PolygonMasks.from_json(sem_seg_polys, H, W)
            bounds = pred_polys.get_bounds()
            rois = torch.tensor([(0, *bound) for bound in bounds]).to(seg_logits.device)

            if len(rois) > 0:
                roi_feats = self.poly_roi_extractor([mask_feats], rois)

                seg2ins_data_samples = [DetDataSample(
                    metainfo=dict(
                        scale_factor=(1,1),
                        img_shape=mask_size,
                        ori_shape=mask_size,
                        batch_input_shape=mask_size
                    ),
                ) for i in range(len(pred_polys))]
                mask_cls_results, mask_pred_results = self.seg2ins_head.predict([roi_feats], seg2ins_data_samples)

                # mask_scores = F.softmax(mask_cls_results, dim=-1)
                # mask_idxes = (mask_scores.max(dim=-1)[1] == 0)
                mask_scores = F.softmax(mask_cls_results, dim=-1)[:,:,0]
                mask_idxes = (mask_scores >= 0.3)
                ins_masks = mask_pred_results[mask_idxes]
                ins_rois = rois.unsqueeze(1).repeat(1, mask_cls_results.shape[1], 1)
                ins_rois = ins_rois[mask_idxes]
                # ins_masks = mask_pred_results[:,0]

                # poly_masks = ins_masks.paste_by_bboxes(bounds, mask_size, mask_size)
                if len(ins_masks) > 0:
                    sem_seg_polys, colors = polygon_utils.polygonize_mask(
                        (ins_masks.sigmoid() > 0.5).to(torch.int), mode='concat_mask_cv2', scale=1.
                    )
                    sem_seg_polys = polygon_utils.paste_poly_json(sem_seg_polys, ins_rois[colors-1][:,1:].cpu(), mask_size[0], mask_size[1], H, W)
                    scores = mask_scores[mask_idxes][colors-1]
                    # pred_results = self.poly_head.predict(
                    #     sem_seg_polys, W, mask_feats, torch.zeros(len(sem_seg_polys), dtype=torch.long),
                    #     device=seg_logits.device, return_format='json'
                    # )
                    # sem_seg_polys = pred_results['simp_polygons']
                else:
                    sem_seg_polys = []
                    scores = scores[:0]
            else:
                sem_seg_polys = []
                scores = scores[:0]
        """




