# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple, Union
import pdb
import rasterio
import shapely
import numpy as np
from rasterio.features import shapes
import pycocotools.mask as mask_util

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmengine.model import ModuleList, caffe2_xavier_init
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean, InstanceList
from ..layers import Mask2FormerTransformerDecoder, SinePositionalEncoding
from ..utils import get_uncertain_point_coords_with_randomness, multi_apply, preprocess_panoptic_gt, multi_apply_v2
from .anchor_free_head import AnchorFreeHead
from .maskformer_head import MaskFormerHead
import mmdet.utils.tanmlh_polygon_utils as polygon_utils


@MODELS.register_module()
class PolygonizerHeadV2(MaskFormerHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`ConfigDict` or dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`ConfigDict` or dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer decoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`ConfigDict` or dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`ConfigDict` or dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            Mask2Former head.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            Mask2Former head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 feat_channels: int,
                 out_channels: int,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 num_queries: int = 100,
                 num_primitive_queries: int = 100,
                 num_inter_points: int = 64 + 1,
                 num_transformer_feat_level: int = 3,
                 max_offsets: float = 30.,
                 apply_poly_refine: bool = True,
                 pixel_decoder: ConfigType = ...,
                 enforce_decoder_input_project: bool = False,
                 transformer_decoder: ConfigType = ...,
                 polyformer_decoder: ConfigType = ...,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=2.0,
                     reduction='mean',
                     class_weight=[1.0] * 133 + [0.1]),
                 loss_mask: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=5.0),
                 loss_dice: ConfigType = dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     reduction='mean',
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=5.0),
                 loss_poly_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=2.0,
                     reduction='mean',
                 ),
                 loss_poly_reg: ConfigType = dict(
                     type='SmoothL1Loss',
                     reduction='mean',
                     loss_weight=0.1
                 ),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_primitive_queries = num_primitive_queries
        self.num_inter_points = num_inter_points
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        self.num_polyformer_decoder_layers = polyformer_decoder.num_layers
        self.max_offsets = max_offsets
        self.apply_poly_refine = apply_poly_refine
        assert pixel_decoder.encoder.layer_cfg. \
            self_attn_cfg.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            **transformer_decoder)
        self.polyformer_decoder = Mask2FormerTransformerDecoder(
            **polyformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = SinePositionalEncoding(
            **positional_encoding)

        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)

        self.primitive_embed = nn.Embedding(self.num_primitive_queries, feat_channels)
        self.primitive_feat = nn.Embedding(self.num_primitive_queries, feat_channels)

        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))
        self.poly_embed = nn.Linear(2, feat_channels)
        self.primitive_reg = nn.Linear(feat_channels, 3) # point primitive, (t, off_x, off_y)
        self.primitive_cls = nn.Linear(feat_channels, 2)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)
        self.loss_poly_cls = MODELS.build(loss_poly_cls)
        self.loss_poly_reg = MODELS.build(loss_poly_reg)

    def init_weights(self) -> None:
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def preprocess_gt(
            self, batch_gt_instances: InstanceList,
            batch_gt_semantic_segs: List[Optional[PixelData]]) -> InstanceList:
        """Preprocess the ground truth for all images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``labels``, each is
                ground truth labels of each bbox, with shape (num_gts, )
                and ``masks``, each is ground truth masks of each instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[Optional[PixelData]]): Ground truth of
                semantic segmentation, each with the shape (1, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.

        Returns:
            list[obj:`InstanceData`]: each contains the following keys

                - labels (Tensor): Ground truth class indices\
                    for a image, with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (Tensor): Ground truth mask for a\
                    image, with shape (n, h, w).
        """
        num_things_list = [self.num_things_classes] * len(batch_gt_instances)
        num_stuff_list = [self.num_stuff_classes] * len(batch_gt_instances)
        gt_labels_list = [
            gt_instances['labels'] for gt_instances in batch_gt_instances
        ]
        gt_masks_list = [
            gt_instances['masks'] for gt_instances in batch_gt_instances
        ]
        gt_poly_jsons_list = [
            gt_instances['masks'].to_json() for gt_instances in batch_gt_instances
        ]
        gt_semantic_segs = [
            None if gt_semantic_seg is None else gt_semantic_seg.sem_seg
            for gt_semantic_seg in batch_gt_semantic_segs
        ]
        targets = multi_apply(preprocess_panoptic_gt, gt_labels_list,
                              gt_masks_list, gt_semantic_segs, num_things_list,
                              num_stuff_list)
        labels, masks = targets
        batch_gt_instances = [
            InstanceData(labels=label, masks=mask, poly_jsons=poly_json)
            for label, mask, poly_json in zip(labels, masks, gt_poly_jsons_list)
        ]
        return batch_gt_instances

    def polygonize_mask(self, imgs, cls_pred, scale=4., num_inter=65):
        temp = imgs
        B, Q, H, W  = imgs.shape
        polygons = []

        # if cls_pred is not None:
        #     imgs[cls_pred == 0] = 0

        scores, labels = F.softmax(cls_pred, dim=-1).max(-1)

        keep = labels.ne(self.num_classes)
        cur_scores = scores[keep]
        cur_classes = labels[keep]

        mask = (imgs > 0).any(dim=1).numpy()
        imgs = imgs.float()
        imgs *= scores.view(B,-1,1,1)

        imgs = imgs.argmax(dim=1).int().numpy()
        # imgs = torch.where(mask, 0, imgs + 1).cpu().int().numpy()
        polygons = torch.zeros(B, Q, num_inter, 2)

        for i in range(B):
            cur_shapes = shapes(imgs[i], mask=mask[i])
            cur_polygons = {}
            for shape, value in cur_shapes:
                # value = int(value) - 1
                value = int(value)

                coords = shape['coordinates']
                scaled_coords = [
                    (polygon_utils.interpolate_ring(torch.tensor(x) * scale, num_bins=num_inter)) for x in coords
                ]
                # scaled_coords = [x.tolist() for x in scaled_coords]

                # shape['coordinates'] = scaled_coords[0]
                cur_ring_len = polygon_utils.get_ring_len(scaled_coords[0])

                if value in cur_polygons and cur_polygons[value][1] < cur_ring_len:
                    cur_polygons[value] = [scaled_coords[0], cur_ring_len]
                else:
                    cur_polygons[value] = [scaled_coords[0], cur_ring_len]

            for key, value in cur_polygons.items():
                polygons[i, key] = value[0]

        return polygons


    def _get_poly_targets_single(self, poly_pred, poly_gt_json):

        """
        poly_gt_shape = shapely.geometry.shape(poly_gt_json)
        poly_gt_contour = shapely.geometry.Polygon(poly_gt_shape.exterior)

        poly_pred_np = poly_pred.cpu().numpy()
        poly_pred_shape = shapely.geometry.LinearRing(poly_pred_np)

        ring_targets_numpy = polygon_utils.get_nearest_ring(poly_pred_shape, poly_gt_contour)
        """

        poly_gt_torch = torch.tensor(poly_gt_json['coordinates'])[0].float() # use the exterior
        poly_targets = polygon_utils.align_rings(poly_pred, poly_gt_torch, self.num_primitive_queries)

        return poly_targets



    def _get_poly_targets(self, poly_preds, poly_gt_jsons):

        if len(poly_preds) == 0:
            return torch.zeros(0, self.num_primitive_queries), torch.zeros(0, self.num_primitive_queries, 2)

        poly_ind_targets_list, poly_offset_targets_list = [], []
        for poly_pred, poly_gt_json in zip(poly_preds.cpu(), poly_gt_jsons):
            _, gt_inds, gt_offsets = self._get_poly_targets_single(poly_pred, poly_gt_json)
            poly_ind_targets_list.append(gt_inds)
            poly_offset_targets_list.append(gt_offsets)

        poly_ind_targets = torch.stack(poly_ind_targets_list, dim=0)
        poly_offset_targets = torch.stack(poly_offset_targets_list, dim=0)

        return poly_ind_targets, poly_offset_targets

    def get_targets(
        self,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        return_sampling_results: bool = False,
        **pred_results,
    ) -> Tuple[List[Union[Tensor, int]]]:

        results = multi_apply_v2(
            self._get_targets_single, batch_gt_instances, batch_img_metas,
            # cls_scores_list, mask_preds_list, poly_preds_list,
            **pred_results
        )
        # (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
        #  poly_ind_targets_list, poly_offset_targets_list, pos_inds_list, neg_inds_list, sampling_results_list) = results[:9]
        # rest_results = list(results[9:])
        sampling_results_list = results['sampling_result']

        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])

        results['avg_factor'] = avg_factor
        return results

        res = (labels_list, label_weights_list, mask_targets_list,
               mask_weights_list, poly_ind_targets_list, poly_offset_targets_list, avg_factor)
        if return_sampling_results:
            res = res + (sampling_results_list)

        return res + tuple(rest_results)

    def _get_targets_single(
        self, gt_instances: InstanceData, img_meta: dict,
        **pred_results
        # cls_score: Tensor, mask_pred: Tensor, poly_pred: Tensor,
    ) -> Tuple[Tensor]:

        cls_score = pred_results['cls_pred']
        mask_pred = pred_results['mask_pred']

        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        gt_poly_jsons = gt_instances.poly_jsons

        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        results = dict(
            labels=labels,
            label_weights=label_weights,
            mask_targets=mask_targets,
            mask_weights=mask_weights,
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            sampling_result=sampling_result
        )

        if 'prim_reg_pred' in pred_results:

            poly_pred = pred_results['poly_pred'][pos_inds]
            poly_targets_jsons = [gt_poly_jsons[x] for x in sampling_result.pos_assigned_gt_inds]
            # poly_targets = self._get_poly_targets(poly_pred[pos_inds].detach().cpu(), gt_poly_jsons)
            poly_ind_targets, poly_offset_targets = self._get_poly_targets(poly_pred, poly_targets_jsons)
            results['poly_ind_targets'] = poly_ind_targets
            results['poly_offset_targets'] = poly_offset_targets

        return results

        return (labels, label_weights, mask_targets, mask_weights, poly_ind_targets,
                poly_offset_targets, pos_inds,
                neg_inds, sampling_result)

    def loss(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
    ) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation of the panoptic
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
            else:
                batch_gt_semantic_segs.append(None)

        # forward
        # all_cls_scores, all_mask_preds, all_poly_preds
        pred_results = self(x, batch_data_samples)

        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances,
                                                batch_gt_semantic_segs)

        # loss
        losses = self.loss_by_feat(pred_results, batch_gt_instances, batch_img_metas)

        return losses

    def loss_by_feat(self, pred_results: dict,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        all_cls_scores = pred_results['cls_pred']
        num_dec_layers = len(all_cls_scores)

        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]

        # losses_cls, losses_mask, losses_dice, losses_poly
        losses = multi_apply_v2(
            self._loss_by_feat_single, batch_gt_instances_list, img_metas_list,
            **pred_results
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses['loss_cls'][-1]
        loss_dict['loss_mask'] = losses['loss_mask'][-1]
        loss_dict['loss_dice'] = losses['loss_dice'][-1]
        loss_dict['loss_poly_cls'] = losses['loss_poly_cls'][-1]
        loss_dict['loss_poly_reg'] = losses['loss_poly_reg'][-1]
        # loss from other decoder layers
        """
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_poly_i in zip(
            losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_poly[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_poly'] = loss_poly_i
            num_dec_layer += 1
        """

        for num_dec_layer in range(len(losses['loss_cls'])):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = losses['loss_cls'][num_dec_layer]
            loss_dict[f'd{num_dec_layer}.loss_mask'] = losses['loss_mask'][num_dec_layer]
            loss_dict[f'd{num_dec_layer}.loss_dice'] = losses['loss_dice'][num_dec_layer]


        return loss_dict



    def _loss_by_feat_single(self, batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict], **pred_results) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = len(batch_gt_instances)
        # cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        # mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        new_pred_results = {}
        for key, value in pred_results.items():
            if value is not None:
                new_pred_results[key] = [value[i] for i in range(num_imgs)]

        target_dict = self.get_targets(
            batch_gt_instances, batch_img_metas, **new_pred_results
        )

        # shape (batch_size, num_queries)
        labels = torch.stack(target_dict['labels'], dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(target_dict['label_weights'], dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(target_dict['mask_targets'], dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(target_dict['mask_weights'], dim=0)
        avg_factor = target_dict['avg_factor']

        cls_scores = pred_results['cls_pred']
        mask_preds = pred_results['mask_pred']

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            loss_poly = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice, loss_poly

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        losses = dict(
            loss_cls=loss_cls,
            loss_mask=loss_mask,
            loss_dice=loss_dice,
        )

        loss_poly = mask_preds[:0].sum() # a dummy loss first
        if 'prim_reg_pred' in pred_results and pred_results['prim_reg_pred'] is not None:
            # poly_targets = torch.cat(poly_targets_list, dim=0)
            poly_preds = pred_results['poly_pred'][mask_weights > 0]
            prim_cls_pred = pred_results['prim_cls_pred'][mask_weights > 0]
            prim_reg_pred = pred_results['prim_reg_pred'][mask_weights > 0]
            poly_ind_targets = torch.cat(target_dict['poly_ind_targets'])
            poly_offset_targets = torch.cat(target_dict['poly_offset_targets'])
            poly_labels = (poly_ind_targets >= 0).to(torch.uint8)

            valid_mask = ~(poly_preds == 0).all(dim=[1,2])

            A = prim_cls_pred[valid_mask].view(-1,2)
            B = poly_labels.to(prim_cls_pred.device)[valid_mask].view(-1)
            mask = B > 0
            loss_poly_cls = self.loss_poly_cls(
                A, B, avg_factor=sum(valid_mask) * self.num_inter_points
            )

            A = prim_reg_pred[valid_mask].reshape(-1, 2)
            B = poly_offset_targets.to(prim_reg_pred.device)[valid_mask].view(-1, 2)
            loss_poly_reg = self.loss_poly_reg(
                A[mask], B[mask], avg_factor=sum(mask)
            )
            losses['loss_poly_cls'] = loss_poly_cls
            losses['loss_poly_reg'] = loss_poly_reg


        return losses

    def _forward_head(
        self, decoder_out: Tensor, mask_feature: Tensor,
        attn_mask_target_size: Tuple[int, int], img_size: Tuple[int, int],
        return_poly: bool = False
    ) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, num_inter_points * 2)
        batch_size, num_queries = mask_embed.shape[:2]

        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)

        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        poly_pred = None
        if return_poly:
            # poly_mask = (mask_pred > 0.5)
            # valid_mask_idx = cls_pred.argmax(dim=-1) != 0
            # poly_mask[valid_mask_idx]
            # poly_pred = self.polygonize_mask((mask_pred > 0.5).cpu().to(torch.uint8).numpy(), num_inter=self.num_inter_points).to(mask_pred.device)
            poly_pred = self.polygonize_mask(
                (mask_pred > 0).cpu().to(torch.uint8),
                cls_pred=cls_pred.cpu(),
                num_inter=self.num_inter_points,
            ).to(mask_pred.device)

        return cls_pred, mask_pred, poly_pred, attn_mask

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        batch_size = x[0].shape[0]
        img_size = batch_data_samples[0].img_shape
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))


        cls_pred_list = []
        mask_pred_list = []
        poly_pred_list = []

        cls_pred, mask_pred, poly_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:], img_size,
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        poly_pred_list.append(poly_pred)

        query_feat_list = [query_feat]

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, poly_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:], img_size,
                return_poly = i == (self.num_transformer_decoder_layers - 1)
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            poly_pred_list.append(poly_pred)
            query_feat_list.append(query_feat)

        pred_results = dict(
            cls_pred=cls_pred_list,
            mask_pred=mask_pred_list,
        )

        if self.apply_poly_refine:
            poly_pred_results = self._forward_poly(poly_pred_list[-1], query_feat_list, mask_features)
            for key, value in poly_pred_results.items():
                pred_results[key] = [None] * self.num_transformer_decoder_layers + [value]
            pred_results['poly_pred'] = poly_pred_list

        return pred_results

    def _forward_poly(self, poly_pred, query_feat_list, mask_features):
        B, Q, N, _ = poly_pred.shape
        C = mask_features.shape[1]
        P =  self.num_primitive_queries

        poly_feat = self.poly_embed(poly_pred).view(B*Q, N, C)
        poly_pos_embed = self.decoder_positional_encoding(poly_feat.new_zeros(B*Q, N, 1))
        poly_pos_embed = poly_pos_embed.view(B*Q, C, N).permute(0,2,1)

        query_feat = self.primitive_feat.weight.unsqueeze(0).repeat((B*Q, 1, 1))
        query_embed = self.primitive_embed.weight.unsqueeze(0).repeat((B*Q, 1, 1))

        prim_pred_cls_list = []
        prim_pred_reg_list = []
        for i in range(self.num_polyformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            layer = self.polyformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=poly_feat,
                value=poly_feat,
                query_pos=query_embed,
                key_pos=poly_pos_embed,
                cross_attn_mask=None,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)

            prim_pred_cls = self.primitive_cls(query_feat).view(B, Q, self.num_primitive_queries, -1)
            prim_pred_reg = self.primitive_reg(query_feat).view(B, Q, self.num_primitive_queries, -1)

            prim_pred_cls_list.append(prim_pred_cls)
            prim_pred_reg_list.append(prim_pred_reg)

        prim_pred_reg = prim_pred_reg_list[-1]
        prim_pred_cls = prim_pred_cls_list[-1]
        t = prim_pred_reg[..., 0].clip(-1,1) # B, Q, P
        prim_pred_base = F.grid_sample(
            poly_pred.view(B*Q, N, 1, -1).permute(0,3,1,2),
            torch.cat([t.view(B*Q, 1, P, 1), t.new_zeros(B*Q, 1, P, 1)], dim=-1),
            align_corners=True
        ).permute(0,3,1,2).view(B,Q,P,2)
        prim_pred_reg = prim_pred_base + prim_pred_reg[..., :2] * self.max_offsets

        results = dict(
            prim_reg_pred = prim_pred_reg,
            prim_cls_pred = prim_pred_cls
        )

        return results

    def predict(self, x: Tuple[Tensor],
                batch_data_samples: SampleList) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two tensors.

                - mask_cls_results (Tensor): Mask classification logits,\
                    shape (batch_size, num_queries, cls_out_channels).
                    Note `cls_out_channels` should includes background.
                - mask_pred_results (Tensor): Mask logits, shape \
                    (batch_size, num_queries, h, w).
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        # all_cls_scores, all_mask_preds, all_poly_preds
        pred_results = self(x, batch_data_samples)

        # mask_cls_results = all_cls_scores[-1]
        # mask_pred_results = all_mask_preds[-1]

        mask_cls_results = pred_results['cls_pred'][-1]
        mask_pred_results = pred_results['mask_pred'][-1]
        poly_pred_results = pred_results['poly_pred'][-1]

        pdb.set_trace()


        B, Q, N, _ = poly_pred_results.shape

        loss = self.loss(x, batch_data_samples)

        # upsample masks
        img_shape = batch_img_metas[0]['batch_input_shape']
        ori_shape = batch_img_metas[0]['ori_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        poly_pred_results = poly_pred_results * ori_shape[0] / img_shape[0]
        poly_pred_results = poly_pred_results.view(B, Q, 1, N * 2).tolist()

        return mask_cls_results, mask_pred_results, poly_pred_results
