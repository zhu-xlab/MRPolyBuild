# Copyright (c) OpenMMLab. All rights reserved.
import torch
import copy
import logging
from torch import Tensor
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.logging import print_log
from mmengine.structures import InstanceData, PixelData
import time
import torch.fft as fft
import scipy
from scipy.ndimage import binary_opening

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import tanmlh_utils
from mmdet.structures.det_data_sample import DetDataSample
from mmdet.structures.mask import PolygonMasks
from typing import List, Tuple, Union
import pdb
import numpy as np
from tqdm import tqdm
from mmdet.utils import tanmlh_polygon_utils as polygon_utils
from .base import BaseDetector
# from mmcv.cnn import auto_fp16

@MODELS.register_module()
class SegBasedDetector(BaseDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone: ConfigDict,
                 seg_head: ConfigDict,
                 train_cfg: ConfigDict,
                 test_cfg: ConfigDict,
                 seg_poly_head: ConfigDict = None,
                 ang_head: ConfigDict = None,
                 frozen_parameters: ConfigDict = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:

        self.frozen_parameters = frozen_parameters
        super().__init__(
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor
        )
        self.backbone = MODELS.build(backbone)
        self.seg_head = MODELS.build(seg_head)

        self.seg_poly_head = None
        if seg_poly_head is not None:
            self.seg_poly_head = MODELS.build(seg_poly_head)

        self.ang_head = None
        if ang_head is not None:
            self.ang_head = MODELS.build(ang_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self, **kwargs):
        super().init_weights(**kwargs)

        frozen_parameters = self.frozen_parameters
        # freeze parameters by prefix
        if frozen_parameters is not None:
            print_log(f'Frozen parameters: {frozen_parameters}', logger='current', level=logging.INFO)
            for name, param in self.named_parameters():
                for frozen_prefix in frozen_parameters:
                    if frozen_prefix in name:
                        param.requires_grad = False
                if param.requires_grad:
                    print_log(f'Training parameters: {name}', logger='current', level=logging.INFO)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        inf_cfg = self.test_cfg.get('inf_cfg', {})

        batch_size, _, h_img, w_img = batch_inputs.size()
        h_stride, w_stride = inf_cfg.stride
        h_crop, w_crop = inf_cfg.crop_size
        h_crop_up, w_crop_up = inf_cfg.crop_up_size
        scale = h_crop_up / h_crop
        out_h, out_w = inf_cfg.out_size if inf_cfg.out_size is not None else (h_img, w_img)
        mask_shape = (round((scale * out_h)), round((scale * out_w)))
        pad_size_h, pad_size_w = inf_cfg.get('pad_size', (out_h, out_w))
        out_pad_size_h, out_pad_size_w = inf_cfg.get('out_pad_size', (out_h, out_w))
        eval_proposal = inf_cfg.get('eval_proposal', False)
        post_processor = MODELS.build(self.test_cfg.get('post_cfg', {}))

        pad_offset = (out_pad_size_h - out_h) // 2

        if pad_size_h != out_h:
            batch_inputs = tanmlh_utils.pad_tensor_to_center(batch_inputs, pad_size_h, pad_size_w)

        out_size_scale = inf_cfg.get('out_size_scale', None)
        if out_size_scale is not None:
            out_h = round(out_h * out_size_scale)
            out_w = round(out_w * out_size_scale)

        if pad_size_h != h_img:
            crop_boxes = tanmlh_utils.get_crop_boxes(pad_size_h, pad_size_w, (h_crop, w_crop), (h_stride, w_stride))
        else:
            crop_boxes = tanmlh_utils.get_crop_boxes(h_img, w_img, (h_crop, w_crop), (h_stride, w_stride))

        assert batch_size == 1
        split_batch_size = inf_cfg.get('split_batch_size', 4)


        selected_crop_imgs = []
        selected_crop_boxes = []
        for crop_idx, crop_box in enumerate(crop_boxes):
            start_x, start_y, end_x, end_y = crop_box
            crop_img = batch_inputs[:, :, start_y:end_y, start_x:end_x]
            # if (crop_img > 0).sum() > 0:
            selected_crop_imgs.append(crop_img)
            selected_crop_boxes.append(crop_box)

        start_idx = 0
        # stop = len(selected_crop_imgs) if len(selected_crop_imgs) % split_batch_size != 0 else len(selected_crop_imgs) + split_batch_size
        # splits = list(np.arange(0, stop, split_batch_size))
        splits = list(np.arange(0, len(selected_crop_imgs), split_batch_size))
        splits.append(len(selected_crop_imgs))

        # file_str = str(img_meta[0]["filename"]).split('/')[-1]
        # box_str = '_'.join([str(x) for x in img_meta[0]["crop_boxes"]])

        new_results_list = []
        if eval_proposal:
            new_proposals_list = []
        if self.seg_head is not None:
            merged_sem_seg_list = []

        # for j in range(len(splits) - 1):
        for j in tqdm(range(len(splits) - 1), desc='extracting building footprint...'):
            cur_crop_boxes = torch.tensor(
                np.stack(selected_crop_boxes[splits[j]:splits[j+1]]), device=batch_inputs.device
            )
            # is_border_boxes = (cur_crop_boxes == 0).any(dim=1) | (cur_crop_boxes == out_h).any(dim=1)

            cur_img = torch.cat(selected_crop_imgs[splits[j]:splits[j+1]])
            cur_img = F.interpolate(cur_img, size=(h_crop_up, w_crop_up), mode='bilinear')

            ori_x = self.backbone(cur_img)

            pseudo_data_samples = [DetDataSample(
                metainfo=dict(
                    # scale_factor=(h_crop_up / h_crop, w_crop_up / w_crop),
                    scale_factor=(1,1),
                    img_shape=(h_crop_up, w_crop_up),
                    ori_shape=(h_crop, w_crop),
                    batch_input_shape=(h_crop_up, w_crop_up),
                    img_path=batch_data_samples[0].metainfo['img_path'],
                )
            ) for _ in cur_img]

            seg_feats = ori_x
            if self.test_cfg.get('seg_head', {}).get('up_feat_levels', None) is not None:
                seg_feats = []
                for level in self.test_cfg['seg_head']['up_feat_levels']:
                    _, _, h, w = ori[level].shape
                    new_x = F.interpolate(ori[level], (h * 2, w * 2))
                    seg_feats.append(new_x)

            pseudo_meta_infos = [data_sample.metainfo for data_sample in pseudo_data_samples]
            pred_sem_seg = self.seg_head.predict(seg_feats, pseudo_meta_infos, None)
            pred_sem_seg = pred_sem_seg.cpu()
            sem_seg_list = [
                InstanceData(
                    sem_seg=cur_sem_seg[None], offsets=offset[None]
                ) for cur_sem_seg, offset in zip(pred_sem_seg, cur_crop_boxes[:,:2] * scale - pad_offset)
            ]

            merged_sem_seg_list.extend(sem_seg_list)


        offsets = torch.cat([x.offsets for x in merged_sem_seg_list])
        merged_sem_seg = tanmlh_utils.mosaic_instance_data(
            merged_sem_seg_list,
            offsets.to(torch.int), mask_shape=(out_h, out_w)
        )

        results = self.seg_poly_head.predict(batch_inputs, merged_sem_seg.sem_seg, batch_data_samples)
        pdb.set_trace()

        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        # x = self.extract_feat(batch_inputs)
        losses = dict()

        x = self.backbone(batch_inputs)

        seg_data_samples = []
        for data_sample in batch_data_samples:
            # gt_sem_seg = data_sample.gt_instances.masks.merge().to_tensor(device=x[0].device, dtype=torch.long)
            # out_h, out_w = gt_sem_seg.shape[1:]
            # data_sample.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)
            # data_sample.gt_sem_seg = PixelData(gt_sem_seg=gt_sem_seg)
            # seg_data_samples.append(PixelData(gt_sem_seg=gt_sem_seg))
            gt_sem_seg = data_sample.gt_sem_seg
            gt_sem_seg.gt_sem_seg = gt_sem_seg.sem_seg
            seg_data_samples.append(gt_sem_seg)

        seg_feats = x
        if self.train_cfg.get('seg_head', {}).get('up_feat_levels', None) is not None:
            seg_feats = []
            for level in self.train_cfg['seg_head']['up_feat_levels']:
                _, _, h, w = x[level].shape
                new_x = F.interpolate(x[level], (h * 2, w * 2))
                seg_feats.append(new_x)

        # losses_seg = self.seg_head.loss(x, seg_data_samples, None)
        seg_logits = self.seg_head.forward(seg_feats)

        losses_seg = self.seg_head.loss_by_feat(seg_logits, seg_data_samples)
        losses.update(losses_seg)

        seg_logits = F.interpolate(
            input=seg_logits,
            size=gt_sem_seg.shape,
            mode='bilinear',
            align_corners=False)

        if self.seg_poly_head is not None:
            losses_poly_seg = self.seg_poly_head.loss(batch_inputs, seg_logits, batch_data_samples)
            losses.update(losses_poly_seg)

        return losses

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pdb.set_trace()

    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images."""
        pdb.set_trace()
