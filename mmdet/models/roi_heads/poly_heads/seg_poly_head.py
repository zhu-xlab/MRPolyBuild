# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import time
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

def make_cnn_stack(in_channels, out_channels, num_layers, kernel_size):
    layers = []
    for i in range(num_layers):
        conv_in = in_channels if i == 0 else out_channels
        layers.append(nn.Conv2d(conv_in, out_channels, kernel_size, padding=kernel_size // 2))
        if i != num_layers - 1:
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

@MODELS.register_module()
class SegPolyHead(BaseModule):

    def __init__(self,
                 poly_roi_extractor: ConfigType = None,
                 mask_feat_embed: ConfigType=None,
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

        if self.poly_cfg.get('use_roi_mask_feat', False):
            assert poly_roi_extractor is not None
            assert mask_feat_embed is not None
            self.poly_roi_extractor = MODELS.build(poly_roi_extractor)
            self.mask_feat_embed = make_cnn_stack(**mask_feat_embed)


    def forward(self, x: Tensor) -> Tensor:

        return x

    def loss(self, imgs, seg_logits: Tensor, batch_data_samples: SampleList) -> dict:

        losses = {}
        B, C, H, W = seg_logits.shape

        sem_seg_probs = F.softmax(seg_logits, dim=1)
        sem_seg_mask = sem_seg_probs[:,1] > self.poly_cfg.get('sem_seg_thr', 0.5)

        batch_pred_polys = []
        import time
        t0 = time.time()

        if self.seg2ins_head is None:
            for i in range(B):
                sem_seg_polys, _ = polygon_utils.polygonize_mask(
                    sem_seg_mask[i].cpu().to(torch.uint8), mode='simple_mask_cv2', scale=1.
                )
                batch_pred_polys.append(sem_seg_polys)
        else:
            for i in range(B):
                pred_polys, scores = self.seg2ins_head.predict(imgs[i], sem_seg_probs[i, 1], batch_data_samples[i])
                batch_pred_polys.append(pred_polys)

        # print(f'polygonization time: {time.time() - t0}')

        if self.poly_cfg.get('train_seg2ins_head', False):
            mask_feats = torch.cat([sem_seg_probs, imgs], dim=1)
            losses_seg2ins = self.seg2ins_head.loss(mask_feats, batch_pred_polys, batch_data_samples)
            losses.update(losses_seg2ins)

        if self.poly_cfg.get('train_poly_head', False):
            losses_poly_head = self._cal_loss_poly_head(
                imgs, seg_logits, batch_pred_polys, batch_data_samples
            )
            losses.update(losses_poly_head)

        return losses

    def _cal_loss_poly_head(self, imgs, seg_logits, batch_pred_polys, batch_data_samples):
        B, C, H, W = seg_logits.shape
        bbox_iou_thr = self.poly_cfg.get('bbox_iou_thr', 0.5)
        poly_iou_thr = self.poly_cfg.get('poly_iou_thr', 0.5)
        num_max_sample = self.poly_cfg.get('num_max_sample', 200)
        num_iter = self.poly_cfg.get('num_iter', 2)

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

        if len(matched_pred_polys) > num_max_sample:
            sample_idxes = np.random.permutation(len(matched_pred_polys))[:num_max_sample]
            matched_pred_polys = matched_pred_polys[sample_idxes]
            matched_gt_polys = matched_gt_polys[sample_idxes]
            batch_idxes = batch_idxes[sample_idxes]

        mask_feat_type = self.poly_head.poly_cfg.get('mask_feat_type', 'img_prob')
        if mask_feat_type == 'img_prob':
            mask_feats = torch.cat([sem_seg_probs, imgs], dim=1)
        elif mask_feat_type == 'prob':
            mask_feats = sem_seg_probs

        if self.poly_cfg.get('use_roi_mask_feat', False):
            bbox_buffer = self.poly_cfg.get('bbox_buffer', 2)
            bounds = torch.tensor(matched_pred_polys.get_bounds(buffer=bbox_buffer))
            mask_size = self.poly_roi_extractor.roi_layers[0].output_size

            rois = torch.cat([batch_idxes.unsqueeze(1), bounds], dim=-1)
            roi_feats = self.poly_roi_extractor([mask_feats], rois)
            mask_feats = self.mask_feat_embed(roi_feats)
            matched_pred_polys = matched_pred_polys.crop_and_resize(bounds.numpy(), mask_size, torch.arange(len(rois)))
            matched_gt_polys = matched_gt_polys.crop_and_resize(bounds.numpy(), mask_size, torch.arange(len(rois)))
            batch_idxes = torch.arange(len(roi_feats))


        losses = self.poly_head.loss(
            matched_pred_polys.to_json(),
            matched_gt_polys.to_json(),
            W, device=seg_logits.device,
            mask_feat=mask_feats,
            batch_idxes=batch_idxes
        )

        return losses

    def predict_seg2ins(self, imgs, batch_data_samples):

        seg_probs = batch_data_samples[0].seg_probs
        B, C, H, W = seg_probs.shape
        sem_seg_thr = self.poly_cfg.get('sem_seg_thr', 0.5)

        seg_mask = (seg_probs[:,1] > sem_seg_thr).long()

        # seg_probs = F.softmax(seg_logits, dim=1)
        # batch_data_samples[0].seg_probs = seg_probs
        # del batch_data_samples[0].seg_probs

        if self.seg2ins_head is not None:
            pred_polys, scores = self.seg2ins_head.predict(imgs[0], seg_probs[0, 1], batch_data_samples)

        else:
            pred_sem_seg = seg_mask.cpu().numpy()[0]
            labeled_mask, num_components = scipy.ndimage.label(pred_sem_seg)
            flat_labels = labeled_mask.ravel()
            flat_prob = seg_probs[0,1].ravel().cpu().numpy()
            sum_prob = np.bincount(flat_labels, weights=flat_prob)
            pixel_counts = np.bincount(flat_labels)
            scores = sum_prob[1:] / pixel_counts[1:]

            pred_polys, colors = polygon_utils.polygonize_mask(
                torch.tensor(labeled_mask).to(torch.int), mode='simple_mask_cv2', scale=1.
            )

            scores = torch.tensor([scores[x-1] for x in colors])

        batch_data_samples[0].pred_polys = pred_polys
        batch_data_samples[0].scores = scores.tolist()

        return batch_data_samples

    def predict_prepare_instances(self, imgs, batch_data_samples):
        seg_probs = batch_data_samples[0].seg_probs
        polygons = batch_data_samples[0].pred_polys
        scores = batch_data_samples[0].scores
        B, C, H, W = seg_probs.shape

        seg_instances = InstanceData(segmentations=polygons, scores=scores)
        seg_instances.polygon_masks = PolygonMasks.from_json(polygons, H, W)
        seg_instances.bboxes = torch.tensor(seg_instances.polygon_masks.get_bounds())
        seg_instances.labels = torch.zeros(len(seg_instances), dtype=torch.long)

        batch_data_samples[0].pred_instances = seg_instances

        return batch_data_samples


    def predict_poly(self, imgs, batch_data_samples):

        if self.poly_head is not None:
            batch_data_samples = self.poly_head.predict(imgs, batch_data_samples)

        else:

            seg_logits = batch_data_samples[0].seg_logits
            pred_polys = batch_data_samples[0].pred_polys
            scores = batch_data_samples[0].scores
            B, _, H, W = seg_probs.shape
            seg_instances = InstanceData(segmentations=pred_polys, scores=scores)
            seg_instances.polygon_masks = PolygonMasks.from_json(pred_polys, H, W)
            seg_instances.bboxes = torch.tensor(seg_instances.polygon_masks.get_bounds())
            seg_instances.labels = torch.zeros(len(seg_instances), dtype=torch.long, device=seg_logits.device)
            batch_data_samples[0].pred_instances = seg_instances

        return batch_data_samples

    def predict(self, imgs, batch_data_samples):
        assert len(imgs) == 1
        assert len(imgs) == len(batch_data_samples)

        t0 = time.time()

        batch_data_samples = self.predict_seg2ins(imgs, batch_data_samples)

        t1 = time.time()

        batch_data_samples = self.predict_poly(imgs, batch_data_samples)

        t2 = time.time()
        # print(f'Poly time: {t1-t0} {t2-t1}')


        return batch_data_samples


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

    def _get_mask_feats(self, imgs, seg_probs, poly_jsons):
        pdb.set_trace()


