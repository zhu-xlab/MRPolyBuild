# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, ModuleList
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.structures.mask import mask_target, poly_target, PolygonMasks
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
import mmdet.utils.tanmlh_polygon_utils as polygon_utils
from mmdet.utils import tanmlh_utils
from mmdet.models.layers import Mask2FormerTransformerDecoder, SinePositionalEncoding

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
#  determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


@MODELS.register_module()
class FCNPolyHead(BaseModule):

    def __init__(self,
                 num_convs: int = 4,
                 roi_feat_size: int = 14,
                 mask_size: int = 36,
                 in_channels: int = 256,
                 conv_kernel_size: int = 3,
                 conv_out_channels: int = 256,
                 poly_out_channels: int = 32 * 2,
                 num_classes: int = 80,
                 class_agnostic: int = False,
                 upsample_cfg: ConfigType = dict(
                     type='deconv', scale_factor=2),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 predictor_cfg: ConfigType = dict(type='Conv'),
                 loss_poly: ConfigType = dict(
                     type='SmoothL1Loss', use_mask=True, loss_weight=1.0),
                 loss_poly_right_ang: ConfigType = dict(
                     type='SmoothL1Loss', use_mask=True, loss_weight=1.0),
                 init_cfg: OptMultiConfig = None,
                 poly_cfg: OptMultiConfig = None,
                 num_queries: int = 32,
                 poly_decoder: ConfigType = None) -> None:
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.mask_size = mask_size
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.predictor_cfg = predictor_cfg
        self.poly_cfg = poly_cfg

        self.loss_poly = MODELS.build(loss_poly)
        if loss_poly_right_ang is not None and self.poly_cfg.get('apply_right_angle_loss', False):
            self.loss_poly_right_ang = MODELS.build(loss_poly_right_ang)


        self.convs = ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=upsample_in_channels, scale_factor=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        else:
            # suppress warnings
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=self.scale_factor,
                mode=self.upsample_method,
                align_corners=align_corners)
            self.upsample = build_upsample_layer(upsample_cfg_)

        reg_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)

        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

        self.poly_decoder = None
        if poly_decoder is not None:
            self.num_queries = num_queries
            self.poly_decoder = Mask2FormerTransformerDecoder(**poly_decoder)
            self.positional_encoding = SinePositionalEncoding(num_feats=128, normalize=True)
            self.query_feat = nn.Embedding(self.num_queries, conv_out_channels)
            self.query_embed = nn.Embedding(self.num_queries, conv_out_channels)
            self.poly_reg_head = nn.Sequential(
                nn.Linear(conv_out_channels, conv_out_channels), nn.ReLU(inplace=True),
                nn.Linear(conv_out_channels, conv_out_channels), nn.ReLU(inplace=True),
                nn.Linear(conv_out_channels, 2)
            )
        else:
            self.conv_reg = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Step 1: Global Average Pooling
                nn.Flatten(),  # Step 2: Flatten to shape (B, C)
                # build_conv_layer(self.predictor_cfg, reg_in_channel, poly_out_channels, 1)
                nn.Linear(reg_in_channel, poly_out_channels)
            )

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        """
        for m in [self.upsample, self.conv_reg]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            elif hasattr(m, 'weight') and hasattr(m, 'bias'):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        """

    def forward(self, x: Tensor) -> Tensor:
        """Forward features from the upstream network.

        Args:
            x (Tensor): Extract mask RoI features.

        Returns:
            Tensor: Predicted foreground masks.
        """
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        if self.poly_decoder is not None:
            batch_size = x.shape[0]
            query_feat = self.query_feat.weight.unsqueeze(0).repeat(
                (batch_size, 1, 1))
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(
                (batch_size, 1, 1))

            decoder_positional_encoding = self.positional_encoding(x[:,0])
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(0, 2, 1)
            decoder_input = x.flatten(2).permute(0, 2, 1)

            num_decoder_layers = len(self.poly_decoder.layers)
            for i in range(num_decoder_layers):
                # if a mask is all True(all background), then set it all False.
                # mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
                # attn_mask = attn_mask & mask_sum
                # cross_attn + self_attn
                layer = self.poly_decoder.layers[i]
                query_feat = layer(
                    query=query_feat,
                    # key=decoder_inputs[level_idx],
                    # value=decoder_inputs[level_idx],
                    key=decoder_input,
                    value=decoder_input,
                    query_pos=query_embed,
                    key_pos=decoder_positional_encoding,
                    # cross_attn_mask=attn_mask,
                    cross_attn_mask=None,
                    query_key_padding_mask=None,
                    # here we do not apply masking on padded region
                    key_padding_mask=None)

            poly_preds = self.poly_reg_head(query_feat)
            poly_preds = (poly_preds + 1) / 2 * self.mask_size

        else:
            poly_preds = self.conv_reg(x).view(len(x), -1, 2) # (B,N,2)
            poly_preds = (poly_preds + 1) / 2 * self.mask_size

        return poly_preds

    def get_targets(self, sampling_results: List[SamplingResult],
                    batch_gt_instances: InstanceList,
                    rcnn_train_cfg: ConfigDict) -> Tensor:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            Tensor: Mask target of each positive proposals in the image.
        """
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_masks = [res.masks for res in batch_gt_instances]
        poly_targets = poly_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)

        return poly_targets

    def loss_and_target(self, poly_preds: Tensor,
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        rcnn_train_cfg: ConfigDict) -> dict:
        """Calculate the loss based on the features extracted by the mask head.

        Args:
            mask_preds (Tensor): Predicted foreground masks, has shape
                (num_pos, num_classes, h, w).
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            dict: A dictionary of loss and targets components.
        """
        poly_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg).to(poly_preds.device)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        N = poly_preds.size(1)
        # num_max_poly = rcnn_train_cfg.get('num_max_poly', 512)
        # poly_preds = poly_preds[:num_max_poly]
        # poly_targets = poly_targets[:num_max_poly]

        losses = dict()
        if poly_preds.size(0) == 0:
            loss_poly = poly_preds.sum()
        else:
            loss_poly = self.loss_poly(poly_preds, poly_targets)

        losses['loss_poly'] = loss_poly
        # TODO: which algorithm requires mask_targets?

        sizes = torch.zeros(len(poly_preds), dtype=int, device=poly_preds.device) + poly_preds.shape[1]

        if self.poly_cfg.get('apply_gcp', True):
            if poly_preds.size(0) == 0:
                loss_dp = poly_preds.sum()
            else:
                dp, dp_points = polygon_utils.batch_decode_ring_dp(
                    poly_preds, sizes, max_step_size=sizes.max(),
                    lam=self.poly_cfg.get('lam', 4),
                    device=poly_preds.device, return_both=True,
                    result_device=poly_preds.device
                )
                dp_points = [x[:-1] for x in dp_points]
                opt_dis = torch.gather(dp, 2, sizes.unsqueeze(1).unsqueeze(1).repeat(1,N,1)).min(dim=1)[0]
                loss_dp = opt_dis.mean() * self.poly_cfg.get('loss_weight_dp', 0.01)

            losses['loss_dp'] = loss_dp

        if self.poly_cfg.get('apply_right_angle_loss', True):
            loss_right_ang = poly_preds[:0].sum()
            eps = 1e-6
            num_base_angles = self.poly_cfg.get('num_base_angles', 16)
            base_angles = tanmlh_utils.generate_angles(num_base_angles).to(poly_preds.device)

            if not self.poly_cfg.get('apply_gcp', True):
                pred_angles = tanmlh_utils.batch_get_angles(poly_preds)
                gt_angles = tanmlh_utils.batch_get_angles(poly_targets)
                pred_angle_dis, pred_angle_idxes = tanmlh_utils.get_base_angle_idxes(pred_angles, base_angles)
                _, gt_angle_idxes = tanmlh_utils.get_base_angle_idxes(gt_angles, base_angles)
                mean_pred_angle_dis = torch.gather(pred_angle_dis, 1, gt_angle_idxes.view(-1,1)).mean()
                loss_right_ang = self.loss_poly_right_ang(
                    mean_pred_angle_dis, torch.zeros_like(mean_pred_angle_dis))

            else:

                angles = []
                gt_angles = []
                for i in range(len(dp_points)):
                    if len(dp_points[i]) >= 3:
                        angle = tanmlh_utils.get_angles(dp_points[i])
                        gt_angle = tanmlh_utils.get_angles(poly_targets[i])

                        angles.append(angle)
                        gt_angles.append(gt_angle)

                if len(angles) > 0:
                    # angles = torch.cat(angles)
                    sum_min_ang_dis_list = []
                    gt_ang_idx_list = []

                    for angle in gt_angles:
                        diff = angle.view(-1,1,1) - base_angles.unsqueeze(0)
                        d1 = (diff.abs() % (torch.pi * 2))
                        d2 = 2 * torch.pi - (diff.abs() % (torch.pi * 2))
                        min_ang_dis = torch.where(d1 < d2, d1, d2)
                        gt_ang_idx = min_ang_dis.min(dim=-1)[0].sum(dim=0).argmin()

                        gt_ang_idx_list.append(gt_ang_idx)

                    for j, angle in enumerate(angles):
                        diff = angle.view(-1,1,1) - base_angles.unsqueeze(0)
                        d1 = (diff.abs() % (torch.pi * 2))
                        d2 = 2 * torch.pi - (diff.abs() % (torch.pi * 2))
                        min_ang_dis = torch.where(d1 < d2, d1, d2)
                        min_ang_dis[:, gt_ang_idx_list[j]].min(dim=-1)[0]

                        sum_min_ang_dis = min_ang_dis[:, gt_ang_idx_list[j]].min(dim=-1)[0].mean()
                        # sum_min_ang_dis = min_ang_dis.min(dim=-1)[0].sum(dim=0).min()

                        sum_min_ang_dis_list.append(sum_min_ang_dis)

                    sum_min_ang_dis = torch.stack(sum_min_ang_dis_list)
                    loss_right_ang = self.loss_poly_right_ang(sum_min_ang_dis, torch.zeros_like(sum_min_ang_dis))

            losses['loss_poly_right_ang'] = loss_right_ang

        return losses

    def predict_by_feat(self,
                        poly_preds: Tuple[Tensor],
                        results_list: List[InstanceData],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: ConfigDict,
                        rescale: bool = False,
                        activate_map: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        mask results.

        Args:
            mask_preds (tuple[Tensor]): Tuple of predicted foreground masks,
                each has shape (n, num_classes, h, w).
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            activate_map (book): Whether get results with augmentations test.
                If True, the `mask_preds` will not process with sigmoid.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        assert len(poly_preds) == len(results_list) == len(batch_img_metas)

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = results_list[img_id]
            bboxes = results.bboxes

            if bboxes.shape[0] == 0:
                results_list[img_id] = empty_instances(
                    [img_meta],
                    bboxes.device,
                    task_type='poly',
                    instance_results=[results],
                    mask_thr_binary=rcnn_test_cfg.mask_thr_binary)[0]
            else:
                segmentations = self._predict_by_feat_single(
                    poly_preds=poly_preds[img_id],
                    bboxes=bboxes,
                    labels=results.labels,
                    img_meta=img_meta,
                    rcnn_test_cfg=rcnn_test_cfg,
                    rescale=rescale,
                    activate_map=activate_map)
                results.segmentations = segmentations
        return results_list

    def _predict_by_feat_single(self,
                                poly_preds: Tensor,
                                bboxes: Tensor,
                                labels: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: ConfigDict,
                                rescale: bool = False,
                                activate_map: bool = False) -> Tensor:
        """Get segmentation masks from mask_preds and bboxes.

        Args:
            mask_preds (Tensor): Predicted foreground masks, has shape
                (n, num_classes, h, w).
            bboxes (Tensor): Predicted bboxes, has shape (n, 4)
            labels (Tensor): Labels of bboxes, has shape (n, )
            img_meta (dict): image information.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            activate_map (book): Whether get results with augmentations test.
                If True, the `mask_preds` will not process with sigmoid.
                Defaults to False.

        Returns:
            Tensor: Encoded masks, has shape (n, img_w, img_h)

        Example:
            >>> from mmengine.config import Config
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> self = FCNMaskHead(num_classes=C, num_convs=0)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_preds = self.forward(inputs)
            >>> # Each input is associated with some bounding box
            >>> bboxes = torch.Tensor([[1, 1, 42, 42 ]] * N)
            >>> labels = torch.randint(0, C, size=(N,))
            >>> rcnn_test_cfg = Config({'mask_thr_binary': 0, })
            >>> ori_shape = (H * 4, W * 4)
            >>> scale_factor = (1, 1)
            >>> rescale = False
            >>> img_meta = {'scale_factor': scale_factor,
            ...             'ori_shape': ori_shape}
            >>> # Encoded masks are a list for each category.
            >>> encoded_masks = self._get_seg_masks_single(
            ...     mask_preds, bboxes, labels,
            ...     img_meta, rcnn_test_cfg, rescale)
            >>> assert encoded_masks.size()[0] == N
            >>> assert encoded_masks.size()[1:] == ori_shape
        """
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

