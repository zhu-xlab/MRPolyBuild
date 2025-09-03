# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple, Union
import pdb
import rasterio
import shapely
import numpy as np
from rasterio.features import shapes
import pycocotools.mask as mask_util
import multiprocessing

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
from mmdet.structures.mask import mask2bbox
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean, InstanceList
from ..layers import Mask2FormerTransformerDecoder, SinePositionalEncoding, PolyFormerTransformerDecoder
from ..utils import get_uncertain_point_coords_with_randomness, multi_apply, preprocess_panoptic_gt, multi_apply_v2
from .anchor_free_head import AnchorFreeHead
from .maskformer_head import MaskFormerHead
import mmdet.utils.tanmlh_polygon_utils as polygon_utils


@MODELS.register_module()
class PolygonizerHeadV19(MaskFormerHead):
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
                 num_transformer_feat_level: int = 3,
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
                 loss_dice_wn: ConfigType = dict(
                     type='DiceLoss',
                     use_sigmoid=False,
                     activate=False,
                     reduction='mean',
                     loss_weight=1.0),
                 loss_poly_reg: ConfigType = dict(
                     type='SmoothL1Loss',
                     reduction='mean',
                     loss_weight=0.1
                 ),
                 loss_poly_right_ang: ConfigType = dict(
                     type='SmoothL1Loss',
                     reduction='mean',
                     loss_weight=1.
                 ),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 poly_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        self.num_polyformer_decoder_layers = polyformer_decoder.num_layers
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.poly_cfg = poly_cfg

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

        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))
        # self.poly_reg_embed = nn.Linear(feat_channels, self.num_inter_points * 2)
        self.poly_reg_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, self.poly_cfg.get('num_inter_points', 96) * 2))

        self.poly_embed = nn.Linear(2, feat_channels)
        self.primitive_reg = nn.Linear(feat_channels, 2) # point primitive, (t, off_x, off_y)

        if self.poly_cfg.get('map_features', False):
            if self.poly_cfg.get('enhance_features', False):
                self.mask_feat_proj = nn.Sequential(
                    nn.Conv2d(feat_channels, feat_channels, 3, stride=1, padding=1),
                    nn.BatchNorm2d(feat_channels), nn.ReLU(),
                    nn.Conv2d(feat_channels, feat_channels, 3, stride=1, padding=1),
                    nn.BatchNorm2d(feat_channels), nn.ReLU(),
                    nn.Conv2d(feat_channels, feat_channels, 3, stride=1, padding=1),
                )
            else:
                self.mask_feat_proj = nn.Conv2d(feat_channels, feat_channels, 3, stride=1, padding=1)

        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.prim_assigner = TASK_UTILS.build(self.train_cfg['prim_assigner']) if 'prim_assigner' in self.train_cfg else None
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
        self.loss_dice_wn = MODELS.build(loss_dice_wn)
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
        batch_gt_semantic_segs: List[Optional[PixelData]],
        batch_img_metas: List[Dict]
    ) -> InstanceList:
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
        # scale = batch_img_metas[0]['img_shape'][0] / batch_img_metas[0]['ori_shape'][0]
        scale=1.
        gt_poly_jsons_list = [
            gt_instances['masks'].to_json(scale=scale) for gt_instances in batch_gt_instances
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

    def polygonize_mask(self, imgs, scale=4., sample_points=False, clockwise=True):

        import time
        coords_time = 0
        t0 = time.time()

        N, H, W  = imgs.shape
        polygons = []
        imgs_cpu = imgs.cpu().numpy()

        # Set the bound before polygonization and use high-res mask
        arange_H = torch.arange(1, H+1, device=imgs.device).unsqueeze(0)
        arange_W = torch.arange(1, W+1, device=imgs.device).unsqueeze(0)
        col_idx = (imgs.sum(dim=1) > 0) * arange_H
        row_idx = (imgs.sum(dim=2) > 0) * arange_W
        x_max, y_max = col_idx.max(dim=1)[0] - 1, row_idx.max(dim=1)[0] - 1
        row_idx = torch.where(row_idx == 0, H+1, row_idx)
        col_idx = torch.where(col_idx == 0, W+1, col_idx)
        x_min, y_min = col_idx.min(dim=1)[0] - 1, row_idx.min(dim=1)[0] - 1
        x_max, x_min = x_max.cpu().numpy(), x_min.cpu().numpy()
        y_max, y_min = y_max.cpu().numpy(), y_min.cpu().numpy()

        t1 = time.time()

        num_coords = 0
        for i in range(N):

            cropped_img = imgs_cpu[i, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
            if cropped_img.sum() > 0:
                cur_shapes = shapes(cropped_img, mask=cropped_img > 0)
                offset = np.array([x_min[i], y_min[i]]).reshape(1,2)
                all_coords = []
                tt0 = time.time()
                for shape, value in cur_shapes:
                    coords = shape['coordinates']
                    scaled_coords = []
                    for x in coords:
                        x = (np.array(x) + offset) * scale
                        num_coords += len(x)
                        if clockwise:
                            x = x[::-1]
                            # x = x.flip(dims=[0])
                        # x = polygon_utils.interpolate_ring(
                        #     x, interval=self.poly_cfg.get('step_size', 8),
                        #     # pad_length=self.poly_cfg.get('num_inter_points', 96),
                        #     type='numpy'
                        # )
                        scaled_coords.append(x.tolist())

                    all_coords.extend(scaled_coords)
                coords_time += time.time() - tt0
            else:
                all_coords = [[[0,0], [0,1], [1,1], [1,0]]] # dummy polygon


            polygons.append(
                dict(
                    type='Polygon',
                    coordinates=all_coords
                )
            )

        # if coords_time > 0.8:
        #     pdb.set_trace()
        # print(f'{N} {num_coords} all time: {time.time() - t0} {t1-t0} coords time: {coords_time}')

        return polygons

    def _get_poly_targets_single(self, poly_pred, poly_gt_json, sampled_segments):

        import time
        t0 = time.time()

        targets = {}

        N = self.poly_cfg.get('num_inter_points', 96)
        max_align_dis = self.poly_cfg.get('max_align_dis', 1e8)

        prim_reg_targets = torch.zeros(N, 2) - 1
        prim_ind_targets = torch.zeros(N, dtype=torch.long)
        prim_ref_targets = torch.zeros(N, 2) - 1

        K = (sampled_segments >= 0).all(dim=-1).sum()

        poly_gt_torch = torch.tensor(poly_gt_json['coordinates'][0]).float() # use the exterior

        """
        poly_gt_torch = torch.tensor(poly_gt_json['coordinates'][0]).float()[:-1] # use the exterior

        if K > 0 and len(poly_gt_torch) > 0:
            dis = ((poly_gt_torch.unsqueeze(1) - sampled_segments[:K].unsqueeze(0)) ** 2).sum(dim=-1) ** 0.5
            valid_gt_mask = dis.min(dim=1)[0] < max_align_dis
            poly_gt_torch = poly_gt_torch[valid_gt_mask]
        """

        if K == 0 or (poly_gt_torch == 0).all():
            targets['prim_ind_targets'] = prim_ind_targets
            targets['prim_reg_targets'] = prim_reg_targets
            # targets['prim_ang_targets'] = prim_ang_targets
            # targets['prim_seq_targets'] = prim_seq_targets
            return targets

        gt_instances = InstanceData(
            labels=torch.zeros(len(poly_gt_torch[:-1]), dtype=torch.long),
            points=poly_gt_torch[:-1]
        ) # (num_classes, N)

        # pred_instances = InstanceData(points=poly_pred[:K])
        pred_instances = InstanceData(points=sampled_segments[:K])

        assign_result = self.prim_assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=None)

        gt_inds = assign_result.gt_inds
        seg_inds = gt_inds.nonzero().view(-1)
        gt_inds = gt_inds[seg_inds]

        dis = ((poly_gt_torch[gt_inds - 1] - sampled_segments[seg_inds]) ** 2).sum(dim=1) ** 0.5
        max_align_dis = self.poly_cfg.get('max_align_dis', 1e8)
        valid_mask = dis < max_align_dis

        prim_reg_targets[seg_inds[valid_mask]] = poly_gt_torch[gt_inds[valid_mask] - 1]
        prim_ind_targets[seg_inds[valid_mask]] = 1

        targets['prim_ind_targets'] = prim_ind_targets
        targets['prim_reg_targets'] = prim_reg_targets

        return targets

    def get_targets(
        self,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        return_sampling_results: bool = False,
        **pred_results,
    ) -> Tuple[List[Union[Tensor, int]]]:

        results = multi_apply_v2(
            self._get_targets_single, batch_gt_instances, batch_img_metas,
            **pred_results
        )

        sampling_results_list = results['sampling_result']

        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])

        results['avg_factor'] = avg_factor
        return results

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
        label_weights = gt_labels.new_ones((self.num_queries,))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0
        poly_targets = [gt_poly_jsons[x] for x in sampling_result.pos_assigned_gt_inds]

        results = dict(
            labels=labels,
            label_weights=label_weights,
            mask_targets=mask_targets,
            mask_weights=mask_weights,
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            poly_targets=poly_targets,
            sampling_result=sampling_result,
        )

        return results

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
        batch_gt_instances = self.preprocess_gt(
            batch_gt_instances, batch_gt_semantic_segs, batch_img_metas
        )

        # loss
        losses = self.loss_by_feat(pred_results, batch_gt_instances, batch_img_metas)

        keys = []
        for key, value in losses.items():
            if key.startswith('vis|'):
                keys.append(key)
                name = key.split('vis|')[1]
                for cur_vis_data, cur_data_samples in zip(value, batch_data_samples):
                    cur_data_samples.set_field(cur_vis_data, name)

        for key in keys:
            losses.pop(key)

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

        if 'loss_poly_cls' in losses:
            loss_dict['loss_poly_cls'] = losses['loss_poly_cls'][-1]

        if 'loss_poly_cls_pos' in losses:
            loss_dict['loss_poly_cls_pos'] = losses['loss_poly_cls_pos'][-1]

        if 'loss_poly_cls_neg' in losses:
            loss_dict['loss_poly_cls_neg'] = losses['loss_poly_cls_neg'][-1]

        if 'loss_poly_reg' in losses:
            loss_dict['loss_poly_reg'] = losses['loss_poly_reg'][-1]

        if 'loss_poly_ts' in losses:
            loss_dict['loss_poly_ts'] = losses['loss_poly_ts'][-1]

        if 'loss_poly_ang' in losses:
            loss_dict['loss_poly_ang'] = losses['loss_poly_ang'][-1]

        if 'loss_ctc' in losses:
            loss_dict['loss_ctc'] = losses['loss_ctc'][-1]

        if 'loss_ctc_ang' in losses:
            loss_dict['loss_ctc_ang'] = losses['loss_ctc_ang'][-1]

        if 'loss_poly_vec' in losses:
            loss_dict['loss_poly_vec'] = losses['loss_poly_vec'][-1]

        if 'loss_poly_right_ang' in losses:
            loss_dict['loss_poly_right_ang'] = losses['loss_poly_right_ang'][-1]

        if 'loss_dp' in losses:
            loss_dict['loss_dp'] = losses['loss_dp'][-1]

        if 'loss_dp_gt' in losses:
            loss_dict['loss_dp_gt'] = losses['loss_dp_gt'][-1]

        if 'loss_poly_iou' in losses:
            loss_dict['loss_poly_iou'] = losses['loss_poly_iou'][-1]


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
            # loss_dict[f'd{num_dec_layer}.loss_poly_reg'] = losses['loss_poly_reg'][num_dec_layer]

        for key, value in losses.items():
            if key.startswith('vis|'):
                loss_dict[key] = value[-1]


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
        _, H, W = mask_targets.shape

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
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

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
        if 'mask_feat' in pred_results and pred_results['mask_feat'] is not None:

            B, Q = mask_weights.shape
            N = self.poly_cfg.get('num_inter_points', 96)
            K = mask_weights.sum().int().item()
            mask_feat = pred_results['mask_feat'];
            query_feat = pred_results['query_feat']
            batch_idxes = (mask_weights.view(-1).nonzero() // mask_weights.shape[1]).view(-1)

            gt_poly_jsons = []
            for cur_poly_jsons in target_dict['poly_targets']:
                gt_poly_jsons += cur_poly_jsons

            scale = self.poly_cfg.get('polygonized_scale', 1.)
            if scale > 1:
                poly_mask = F.interpolate(
                    mask_preds.unsqueeze(1),
                    scale_factor=(scale, scale),
                    mode='bilinear',
                    align_corners=False).detach()
                poly_mask = poly_mask[:,0]
            else:
                poly_mask = mask_preds.detach()

            poly_mask = (poly_mask > 0).to(torch.uint8)
            nonzero_mask = poly_mask.sum(dim=[1,2]) > 0
            import time
            t0 = time.time()

            poly_mask_jsons = self.polygonize_mask(
                poly_mask,
                scale=4. / scale,
                sample_points=False
            )
            sampled_rings, _, _ = polygon_utils.sample_rings_from_json(
                poly_mask_jsons, interval=self.poly_cfg.get('step_size'), only_exterior=True,
                num_min_bins=self.poly_cfg.get('num_min_bins', 8),
                num_bins=self.poly_cfg.get('num_bins', None)
            )
            # sampled_segments = torch.stack([x[:-1] for x in sampled_rings], dim=0)
            sampled_segments, is_complete = polygon_utils.sample_segments_from_rings(sampled_rings, self.poly_cfg.get('num_inter_points'))

            # print(f'polygonize time: {time.time() - t0}')

            prim_reg_preds = torch.zeros(K, N, 2, device=mask_preds.device)
            prim_reg_targets = torch.zeros(K, N, 2, device=mask_preds.device)
            full_poly_pred = torch.zeros(B, Q, N, 2, device=mask_preds.device) - 1


            t0 = time.time()
            poly_pred = sampled_segments.to(mask_preds.device)

            full_poly_pred[mask_weights > 0] = poly_pred
            poly_pred_results = self._forward_poly(
                full_poly_pred, query_feat, mask_feat, mask=mask_weights > 0
            )
            prim_reg_pred = poly_pred_results['prim_reg_pred']

            # print(f'forward_poly time: {time.time() - t0}')


            match_idxes = []
            t0 = time.time()
            for i in range(K):
                prim_target = self._get_poly_targets_single(
                    prim_reg_pred[i], gt_poly_jsons[i],
                    sampled_segments=sampled_segments[i]
                )
                prim_reg_targets[i] = prim_target['prim_reg_targets']
                if is_complete[i]:
                    seg_mask = (sampled_segments[i] >= 0).all(dim=-1)
                    pred_poly = shapely.geometry.Polygon(sampled_segments[i][seg_mask].tolist())
                    gt_poly = shapely.geometry.Polygon(gt_poly_jsons[i]['coordinates'][0])
                    iou = polygon_utils.polygon_iou(pred_poly, gt_poly)
                    if iou > self.poly_cfg.get('align_iou_thre', 0.5):
                        match_idxes.append(i)

            match_idxes = torch.tensor(match_idxes)

            sizes = (prim_reg_pred >= 0).all(dim=-1).sum(dim=1)

            # decoded_rings = polygon_utils.batch_decode_ring_dp(prim_reg_pred, sizes, max_step_size=64, lam=4, device=prim_reg_pred.device)
            if self.poly_cfg.get('apply_poly_iou_loss', False):
                # prim_reg_pred = torch.cat([prim_reg_pred, prim_reg_pred[:, :1]], dim=1)
                dp, dp_points = polygon_utils.batch_decode_ring_dp(
                    prim_reg_pred, sizes, max_step_size=sizes.max(),
                    lam=self.poly_cfg.get('lam', 4),
                    device=prim_reg_pred.device, return_both=True,
                    result_device=prim_reg_pred.device
                )
            else:
                dp = polygon_utils.batch_decode_ring_dp(
                    prim_reg_pred, sizes, max_step_size=sizes.max(),
                    lam=self.poly_cfg.get('lam', 4),
                    device=prim_reg_pred.device, only_return_dp=True
                )

            # opt_dis = torch.gather(dp[:,0], 1, sizes.unsqueeze(1)-1)

            opt_dis_comp = torch.gather(dp[is_complete], 2, sizes[is_complete].unsqueeze(1).unsqueeze(1).repeat(1,N,1)).min(dim=1)[0]
            opt_dis_incomp = torch.gather(dp[~is_complete, 0], 1, sizes[~is_complete].unsqueeze(1)-1)

            losses['loss_dp'] = (opt_dis_comp.sum() + opt_dis_incomp.sum()) / K * self.poly_cfg.get('loss_weight_dp', 0.01)

            dp_points = [x[:-1] for x in dp_points]
            if self.poly_cfg.get('apply_poly_iou_loss', False):
                mask_point_targets = mask_point_targets.view(K, -1)
                loss_poly_iou = mask_preds[:0].sum()
                wn_list = []
                # for i, (pred_ring, sampled_coords) in enumerate(zip(dp_points, points_coords)):
                for i in match_idxes:
                    pred_ring = dp_points[i]
                    sampled_coords = points_coords[i] * H
                    # pred_ring = torch.tensor([[0, 1], [1,1], [1,0], [0,0]])
                    # sampled_coords = torch.tensor([[-1, -1], [0.5, 0.5], [2, 1], [1,2], [0.7, 0.7]])
                    wn = polygon_utils.cal_winding_number(pred_ring, sampled_coords)
                    wn_list.append(wn)

                if len(wn_list) > 0:
                    wns = torch.stack(wn_list, dim=0)
                    loss_poly_iou = self.loss_dice_wn(wns, mask_point_targets[match_idxes])
                    losses['loss_poly_iou'] = loss_poly_iou
                else:
                    losses['loss_poly_iou'] = prim_reg_pred[:0].sum()

            # Polygon regression
            A = prim_reg_pred.reshape(-1, 2)
            B = prim_reg_targets.view(-1, 2)

            if self.poly_cfg.get('reg_targets_type', 'vertice') == 'contour':
                mask = (poly_pred >= 0).all(dim=-1).view(-1)
                loss_poly_reg = self.loss_poly_reg(A[mask], B[mask])
            elif self.poly_cfg.get('reg_targets_type', 'vertice') == 'vertice':
                mask = (prim_reg_targets >= 0).all(dim=-1).view(-1)
                loss_poly_reg = self.loss_poly_reg(A[mask], B[mask])
            else:
                raise ValueError()

            losses['loss_poly_reg'] = loss_poly_reg
            if self.poly_cfg.get('use_right_angle_loss', False):
                mask = (prim_ind_targets == 0)
                loss_right_angle = prim_reg_pred[:0].sum()
                if K > 0:
                    for i in range(K):
                        if mask[i].sum() > 0:
                            cur_pred = prim_reg_pred[i, mask[i]]
                            cur_target = prim_reg_targets[i, mask[i]]
                            angle_pred, valid_mask = polygon_utils.calculate_polygon_angles(cur_pred)
                            angle_target, valid_mask2 = polygon_utils.calculate_polygon_angles(cur_target)
                            valid_mask = valid_mask & valid_mask2
                            loss_right_angle += self.loss_poly_right_ang(angle_pred[valid_mask], angle_target[valid_mask])
                    loss_right_angle = loss_right_angle / K

                losses['loss_poly_right_ang'] = loss_right_angle

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

        return cls_pred, mask_pred, poly_pred, attn_mask

    def forward(
        self, x: List[Tensor],
        batch_data_samples: SampleList,
        return_poly=False
    ) -> Tuple[List[Tensor]]:
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
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            query_feat_list.append(query_feat)

        pred_results = dict(
            cls_pred=cls_pred_list,
            mask_pred=mask_pred_list,
            query_feat=query_feat_list,
            mask_feat=[None] * (self.num_transformer_decoder_layers) + [mask_features]
        )

        if return_poly:

            mask_pred = mask_pred_list[-1]
            cls_pred = cls_pred_list[-1]
            scale = self.poly_cfg.get('polygonized_scale', 1.)
            B, Q, H, W = mask_pred.shape

            max_per_image = self.test_cfg.get('max_per_image', 100)
            scores = F.softmax(cls_pred, dim=-1)[...,0]
            topk_inds = scores.topk(dim=1, k=max_per_image, sorted=False)[1]
            topk_mask = torch.zeros(B, Q, dtype=torch.bool, device=scores.device)
            topk_inds = topk_inds.sort(dim=1)[0]
            for i, inds in enumerate(topk_inds):
                topk_mask[i, inds] = True

            pred_results['cls_pred'][-1] = torch.gather(
                pred_results['cls_pred'][-1], 1, topk_inds.view(B,-1,1).repeat(1,1,2)
            )
            pred_results['mask_pred'][-1] = torch.gather(
                pred_results['mask_pred'][-1], 1, topk_inds.view(B,-1,1,1).repeat(1,1,H,W)
            )

            N = self.poly_cfg.get('num_inter_points', 96)
            if scale > 1:
                poly_mask = F.interpolate(
                    mask_pred.view(B*Q, 1, H, W),
                    scale_factor=(scale, scale),
                    mode='bilinear',
                    align_corners=False).detach()
                poly_mask = poly_mask[:,0].view(B, Q, int(H*scale), int(W*scale))
            else:
                poly_mask = mask_pred.detach()

            poly_mask = (poly_mask > 0).to(torch.uint8)

            thre = self.poly_cfg.get('mask_cls_thre', 0.1)
            valid_mask = (scores > thre) & (poly_mask.sum(dim=[2,3]) > 0)
            poly_mask[~valid_mask] = 0

            valid_poly_mask = poly_mask[topk_mask]

            import time
            t0 = time.time()
            poly_pred_jsons = self.polygonize_mask(
                valid_poly_mask,
                scale=4. / scale,
                sample_points=False
            )
            # print(f'polygonize time: {time.time() - t0}')

            pred_results['poly_pred_jsons'] = poly_pred_jsons

            if self.poly_cfg.get('return_poly_json', False):
                pred_results['query_idxes'] = valid_mask.nonzero()
                pred_results['poly_jsons'] = poly_pred_jsons
                pred_results['poly_ind_mask'] = valid_mask

                return pred_results

            sampled_segs, seg_sizes, poly2segs_idxes, segs2poly_idxes = polygon_utils.sample_segments_from_json(
                poly_pred_jsons, interval=self.poly_cfg.get('step_size'),
                seg_len=N, stride=self.poly_cfg.get('stride_size', 64),
                num_min_bins=self.poly_cfg.get('num_min_bins', 8),
                num_bins=self.poly_cfg.get('num_bins', None),
            )

            # sampled_segs = sampled_segs[:, :-1]

            """
            all_rings, all_idxes, all_ring_sizes = polygon_utils.sample_rings(
                poly_pred_jsons, interval=self.poly_cfg.get('step_size'),
                length=N, ring_stride=self.poly_cfg.get('stride_size', 64)
            )
            """
            if len(sampled_segs) > 0:
                query_idx = topk_mask.view(-1).nonzero().view(-1)[segs2poly_idxes[:,0]]
                query_feat = query_feat_list[-1].view(B*Q, -1)[query_idx]
                poly_pred = torch.from_numpy(sampled_segs).to(query_feat.device).float()

                poly_pred_results = self._forward_poly(poly_pred.unsqueeze(0), query_feat.unsqueeze(0), mask_features, mask=None)
                pred_results.update(poly_pred_results)
                pred_results['poly_pred'] = poly_pred
                pred_results['seg_idxes'] = poly2segs_idxes
                pred_results['seg_sizes'] = seg_sizes
                pred_results['query_idxes'] = topk_mask.nonzero()
                pred_results['sampled_segs'] = sampled_segs

        return pred_results

    def _forward_poly(self, poly_pred, query_feat, mask_feat, mask=None):

        B, Q, N, _ = poly_pred.shape
        B, C, H, W = mask_feat.shape
        _, Q, _ = query_feat.shape


        results = dict()
        norm_poly_pred = (poly_pred / (H * 4.) - 0.5) * 2 # scale is manually set to 4.
        if mask is not None:
            K = mask.sum()
        else:
            mask = (poly_pred >= 0).any(dim=[2,3])
            K = mask.sum().int().item()

        if self.poly_cfg.get('map_features', False):
            mask_feat = self.mask_feat_proj(mask_feat)

        # norm_poly_pred = polygon_utils.normalize_rings(poly_pred)
        poly_valid_mask = (poly_pred[mask] >= 0).all(dim=-1)

        poly_feat = torch.zeros(K, N, C).to(poly_pred.device)
        if self.poly_cfg.get('use_point_feat_in_poly_feat', False):
            point_feat = F.grid_sample(
                mask_feat,
                norm_poly_pred.view(B, Q, N, 2),
                align_corners=True
            )
            point_feat = point_feat.permute(0,2,3,1).reshape(B, Q, N, -1)
            temp = point_feat[mask]
            temp[~poly_valid_mask] = 0
            poly_feat += temp

        if self.poly_cfg.get('use_coords_in_poly_feat', False):
            temp = self.poly_embed(norm_poly_pred[mask]).view(K, N, C)
            temp[~poly_valid_mask] = 0
            poly_feat += temp

        if self.poly_cfg.get('use_decoded_feat_in_poly_feat', False):
            poly_feat += query_feat[mask].detach().view(K, 1, -1)

        poly_pos_embed = self.decoder_positional_encoding(poly_feat.new_zeros(K, N, 1))
        poly_pos_embed = poly_pos_embed.view(K, C, N).permute(0,2,1)

        query_feat = poly_feat
        query_embed = poly_pos_embed

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

            if i == self.num_polyformer_decoder_layers - 1:

                prim_pred_reg = self.primitive_reg(query_feat).view(K, N, -1)
                prim_pred_reg_list.append(prim_pred_reg)

        prim_pred_reg = prim_pred_reg_list[-1]

        prim_pred_reg = poly_pred[mask] + prim_pred_reg * self.poly_cfg.get('max_offsets', 10)
        prim_pred_reg = torch.clamp(prim_pred_reg, 0, H * 4)
        prim_pred_reg[(poly_pred[mask] < 0).all(dim=-1)] = -1

        results['prim_reg_pred'] = prim_pred_reg

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
        import time
        t0 = time.time()

        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        max_per_image = self.test_cfg.get('max_per_image', 100)
        img_shape = batch_img_metas[0]['batch_input_shape']
        ori_shape = batch_img_metas[0]['ori_shape']
        scale = ori_shape[0] / img_shape[0]
        # all_cls_scores, all_mask_preds, all_poly_preds
        pred_results = self(x, batch_data_samples, return_poly=True)

        mask_cls_results = pred_results['cls_pred'][-1]
        mask_pred_results = pred_results['mask_pred'][-1]
        B, Q, _ = mask_cls_results.shape

        results = dict(
            mask_cls_results=mask_cls_results,
            mask_pred_results=mask_pred_results,
        )

        poly_pred_results = [[[[0,0,0,0,0,0]]] * max_per_image] * B

        if 'seg_idxes' in pred_results:
            poly_pred = pred_results['poly_pred']
            prim_reg_pred = pred_results['prim_reg_pred'] * scale
            seg_idxes = pred_results['seg_idxes']
            seg_sizes = pred_results['seg_sizes']
            query_idxes = pred_results['query_idxes']
            poly_pred_jsons = pred_results['poly_pred_jsons']
            sampled_segs = torch.tensor(pred_results['sampled_segs']) * scale


            rings, poly2ring_idxes, others = polygon_utils.assemble_segments(
                prim_reg_pred.cpu(), seg_idxes, seg_sizes,
                length=self.poly_cfg.get('num_inter_points', 96),
                stride=self.poly_cfg.get('stride_size', 64),
            )

            if self.poly_cfg.get('poly_decode_type', 'dp') == 'dp':
                rings = [ring.to(poly_pred.device) for ring in rings]
                # rings = [torch.cat([ring, ring[:1]], dim=0) for ring in rings]

                simp_rings = polygon_utils.simplify_rings_dp(
                    rings, lam=self.poly_cfg.get('lam', 4), device=x[0].device,
                    ref_rings=sampled_rings if self.poly_cfg.get('use_ref_rings', False) else None,
                    drop_last=False
                )

            elif self.poly_cfg.get('poly_decode_type', 'dp') == 'cls':
                rings_cls = others['cls_pred']
                simp_rings = []
                for i, cur_ring in enumerate(rings):
                    cur_scores = F.softmax(rings_cls[i], dim=1)[:,1]
                    cur_mask = cur_scores > self.poly_cfg.get('prim_cls_thre', 0.1)
                    if cur_mask.sum() >= 4:
                        simp_rings.append(cur_ring[cur_mask])
                    else:
                        simp_rings.append(sampled_rings[i])

            else:
                simp_rings = rings

            simp_polygons = polygon_utils.assemble_rings(
                simp_rings, poly2ring_idxes
            )

            poly_pred_results = [simp_polygons]

        elif 'poly_jsons' in pred_results:
            poly_jsons = pred_results['poly_jsons']
            query_idxes = pred_results['query_idxes']
            if self.poly_cfg.get('use_gt_jsons', False):
                poly_jsons = batch_data_samples[0].gt_instances.masks.to_json()
                scale=1.

            if self.poly_cfg.get('poly_decode_type', 'dp') == 'dp':
                if poly_jsons.__len__() > 0:
                    pred_polygons = polygon_utils.simplify_poly_jsons(
                        poly_jsons, interval=self.poly_cfg.get('step_size'),
                        scale=scale, lam=self.poly_cfg.get('lam', 4), device=x[0].device,
                        num_min_bins=self.poly_cfg.get('num_min_bins', 8),
                        num_bins=self.poly_cfg.get('num_bins', None)
                    )
                else:
                    pred_polygons = [[[0,0,0,0,0,0]]]

            elif self.poly_cfg.get('poly_decode_type', 'dp') == 'none':

                pred_polygons = []
                for i, poly_json in enumerate(poly_jsons):
                    cur_pred_poly = polygon_utils.poly_json2coco(poly_json, scale=scale)
                    pred_polygons.append(cur_pred_poly)

            if self.poly_cfg.get('use_gt_jsons', False):
                num_gts = len(pred_polygons)
                results['mask_cls_results'] = torch.zeros(B, max_per_image, 2)
                results['mask_pred_results'] = torch.zeros(B, max_per_image, *ori_shape)
                results['mask_cls_results'][:,:num_gts, 0] = 1e8
                results['mask_pred_results'][:,:,0,0] = 1
                for i, cur_polygon in enumerate(pred_polygons):
                    if cur_polygon == [] or len(cur_polygon[0]) < 6:
                        cur_polygon = [[0,0,0,0,0,0]]
                    poly_pred_results[0][i] = cur_polygon

            else:
                # for idx, cur_polygon in zip(query_idxes, pred_polygons):
                #     if cur_polygon == [] or len(cur_polygon[0]) < 6:
                #         cur_polygon = [[0,0,0,0,0,0]]
                #     poly_pred_results[idx[0]][idx[1]] = cur_polygon

                for i, cur_polygon in enumerate(pred_polygons):
                    if cur_polygon == [] or len(cur_polygon[0]) < 6:
                        cur_polygon = [[0,0,0,0,0,0]]
                    poly_pred_results[0][i] = cur_polygon


            # if self.poly_cfg.get('use_gt_jsons', False):
            #     results['poly_jsons'] = poly_jsons

        results['poly_pred_results'] = poly_pred_results

        return results
