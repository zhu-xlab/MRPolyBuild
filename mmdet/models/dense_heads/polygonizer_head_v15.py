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
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean, InstanceList
from ..layers import Mask2FormerTransformerDecoder, SinePositionalEncoding, PolyFormerTransformerDecoder
from ..utils import get_uncertain_point_coords_with_randomness, multi_apply, preprocess_panoptic_gt, multi_apply_v2
from .anchor_free_head import AnchorFreeHead
from .maskformer_head import MaskFormerHead
import mmdet.utils.tanmlh_polygon_utils as polygon_utils


@MODELS.register_module()
class PolygonizerHeadV15(MaskFormerHead):
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
                 poly_align_type: str = 'sort_by_angle',
                 interpolate_interval: float = 4,
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
                 loss_vert: ConfigType = dict(
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
                 loss_poly_ang: ConfigType = dict(
                     type='SmoothL1Loss',
                     reduction='mean',
                     loss_weight=1.
                 ),
                 loss_poly_vec: ConfigType = dict(
                     type='SmoothL1Loss',
                     reduction='mean',
                     loss_weight=1.
                 ),
                 loss_poly_right_ang: ConfigType = dict(
                     type='SmoothL1Loss',
                     reduction='mean',
                     loss_weight=1.
                 ),
                 loss_poly_ts: ConfigType = dict(
                     type='MSELoss',
                     reduction='mean',
                     loss_weight=5.
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
        self.interpolate_interval = interpolate_interval
        self.poly_align_type = poly_align_type
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

        num_primitive_queries = self.poly_cfg.get('num_primitive_queries', 64)
        self.primitive_embed = nn.Embedding(num_primitive_queries, feat_channels)
        self.primitive_feat = nn.Embedding(num_primitive_queries, feat_channels)

        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.vert_embed = nn.Sequential(
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
        self.primitive_cls = nn.Linear(feat_channels, self.poly_cfg.get('num_cls_channels', 2))

        if self.poly_cfg.get('pred_angle', False):
            # self.primitive_ang = nn.Linear(feat_channels, self.poly_cfg.get('num_angle_bins', 1) + 1)
            self.primitive_ang = nn.Linear(feat_channels, 1)
        # self.primitive_ts = nn.Linear(feat_channels, 2)

        if self.poly_cfg.get('map_features', False):
            self.mask_feat_proj = nn.Conv2d(feat_channels, feat_channels, 1, stride=1, padding=0)
            self.vert_feat_proj = nn.Conv2d(feat_channels, feat_channels, 1, stride=1, padding=0)

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
        self.loss_vert = MODELS.build(loss_vert)
        self.loss_dice = MODELS.build(loss_dice)
        self.loss_poly_cls = MODELS.build(loss_poly_cls)
        # self.loss_poly_cls = nn.BCELoss()
        self.loss_poly_reg = MODELS.build(loss_poly_reg)
        self.loss_poly_ts = MODELS.build(loss_poly_ts)
        self.loss_poly_right_ang = MODELS.build(loss_poly_right_ang)
        if self.poly_cfg.get('use_ctc_loss', False):
            self.loss_ctc = nn.CTCLoss(blank=self.poly_cfg.get('num_inter_points', 96)-1,
                                       reduction='mean')
            self.loss_ctc_ang = nn.CTCLoss(blank=self.poly_cfg.get('num_angle_bins', 32),
                                           reduction='mean', zero_infinity=True)

        if self.poly_cfg.get('pred_angle', False):
            self.loss_poly_ang = MODELS.build(loss_poly_ang)
            self.loss_poly_vec = MODELS.build(loss_poly_vec)

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
                all_coords = [[[-1,-1], [-1,-1], [-1,-1]]]


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

    def _get_poly_targets_single(self, poly_pred, poly_gt_json, sampled_segments=None):

        import time
        t0 = time.time()

        targets = {}

        P = self.poly_cfg.get('num_primitive_queries', 64)
        N = self.poly_cfg.get('num_inter_points', 96)
        max_align_dis = self.poly_cfg.get('max_align_dis', 1e8)

        prim_reg_targets = torch.zeros(P, 2) - 1
        prim_ind_targets = torch.zeros(P, dtype=torch.long)
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


        if self.train_cfg.get('prim_assigner', None) is None:
            t1 = time.time()
            inter_gt_torch = polygon_utils.interpolate_ring(poly_gt_torch, num_bins=K+1)[:-1]
            reg_targets = polygon_utils.align_rings_by_roll(inter_poly_pred, inter_gt_torch)

            _, pos_inds = (((reg_targets[:-1].unsqueeze(0) - poly_gt_torch[:-1].unsqueeze(1)) ** 2).sum(dim=-1) ** 0.5).min(dim=1)
            shift = -pos_inds.argmin()
            pos_inds = torch.roll(pos_inds, shifts=[shift])
            poly_gt_torch_roll = torch.roll(poly_gt_torch[:-1], shifts=[shift], dims=[0])

            poly_gt_torch_roll = poly_gt_torch_roll[pos_inds < P]
            pos_inds = pos_inds[pos_inds < P]

            t2 = time.time()

            if len(pos_inds.unique()) != len(pos_inds) or len(pos_inds) == 0:
                targets['prim_reg_preds'] = prim_reg_preds
                # targets['prim_ind_targets'] = prim_ind_targets
                targets['prim_reg_targets'] = prim_reg_targets
                # targets['prim_ang_targets'] = prim_ang_targets
                # targets['prim_seq_targets'] = prim_seq_targets
                return targets

            prim_reg_preds[:len(inter_poly_pred)] = inter_poly_pred[:P]

            if self.poly_cfg.get('reg_targets_type', 'vertice') == 'contour':
                prim_reg_targets[:len(reg_targets)] = reg_targets[:P]

            elif self.poly_cfg.get('reg_targets_type', 'vertice') == 'vertice':
                prim_reg_targets[pos_inds] = poly_gt_torch_roll
            else:
                raise ValueError()
            t3 = time.time()

            # prim_ind_targets[pos_inds] = pos_inds

            # ang_targets = polygon_utils.cal_angle_for_ring(reg_targets)[:-1]
            # prim_ang_targets[:len(ang_targets)] = ang_targets[:P]
            # ang_targets = polygon_utils.cal_angle_for_ring(poly_gt_torch_roll)
            # ang_targets = polygon_utils.binarize(
            #     (ang_targets + torch.pi * 2).fmod(torch.pi * 2),
            #     self.poly_cfg.get('num_angle_bins', 36), 0, torch.pi * 2
            # )
            # prim_ang_targets[:len(ang_targets)] = ang_targets[:P]

            # prim_seq_targets[:len(pos_inds)] = pos_inds[:P]

        else:

            gt_instances = InstanceData(
                # labels=pos_inds, points=poly_gt_torch_roll
                labels=torch.zeros(len(poly_gt_torch[:-1]), dtype=torch.long),
                points=poly_gt_torch[:-1]
            ) # (num_classes, N)

            pred_instances = InstanceData(points=poly_pred[:K])

            assign_result = self.prim_assigner.assign(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=None)

            gt_inds = assign_result.gt_inds
            temp = gt_inds.nonzero().view(-1)
            point_pos_inds = gt_inds[temp]

            # prim_reg_targets[temp] = poly_gt_torch[point_pos_inds - 1]
            # prim_ind_targets[temp] = 1

            dis = ((poly_gt_torch[point_pos_inds - 1] - sampled_segments[temp]) ** 2).sum(dim=1) ** 0.5
            max_align_dis = self.poly_cfg.get('max_align_dis', 1e8)
            valid_mask = dis < max_align_dis

            prim_reg_targets[temp[valid_mask]] = poly_gt_torch[point_pos_inds[valid_mask] - 1]
            prim_ind_targets[temp[valid_mask]] = 1


        """
        if self.poly_cfg.get('pred_angle', False):
            assert self.poly_cfg['point_as_prim'] is True
            prim_ang_pred = pred_results['prim_ang_pred']
        """
        # targets['prim_reg_preds'] = prim_reg_preds
        targets['prim_ind_targets'] = prim_ind_targets
        targets['prim_reg_targets'] = prim_reg_targets
        # targets['prim_ang_targets'] = prim_ang_targets
        # targets['prim_seq_targets'] = prim_seq_targets

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
            # cls_scores_list, mask_preds_list, poly_preds_list,
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
        gt_vert_masks = torch.zeros_like(gt_masks)
        _, H, W = gt_masks.shape

        for i, gt_json in enumerate(gt_poly_jsons):
            for ring in gt_json['coordinates']:
                idxes = torch.tensor(ring).round().long()[:-1]
                idxes = idxes.clamp(0, H-1)
                gt_vert_masks[i, idxes[:,1], idxes[:,0]] = 1

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
        vert_targets = gt_vert_masks[sampling_result.pos_assigned_gt_inds]

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
            vert_targets=vert_targets,
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
        loss_dict['loss_vert_pos'] = losses['loss_vert_pos'][-1]
        loss_dict['loss_vert_neg'] = losses['loss_vert_neg'][-1]

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
            loss_dict[f'd{num_dec_layer}.loss_vert_pos'] = losses['loss_vert_pos'][num_dec_layer]
            loss_dict[f'd{num_dec_layer}.loss_vert_neg'] = losses['loss_vert_neg'][num_dec_layer]


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
        vert_targets = torch.cat(target_dict['vert_targets'], dim=0)
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
            loss_vert = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice, loss_vert

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

        vert_preds = pred_results['vert_pred'][mask_weights > 0]

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                vert_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            vert_point_targets = point_sample(
                vert_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        vert_point_preds = point_sample(
            vert_preds.unsqueeze(1), points_coords).squeeze(1)

        vert_pos_mask = vert_point_targets > 0
        loss_vert_pos = self.loss_vert(
            vert_point_preds[vert_pos_mask],
            vert_point_targets[vert_pos_mask],
            avg_factor=vert_pos_mask.sum())

        loss_vert_neg = self.loss_vert(
            vert_point_preds[~vert_pos_mask],
            vert_point_targets[~vert_pos_mask],
            avg_factor=(~vert_pos_mask).sum())

        losses = dict(
            loss_cls=loss_cls,
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            loss_vert_pos=loss_vert_pos,
            loss_vert_neg=loss_vert_neg
        )

        return losses

    def _forward_head(
        self, decoder_out: Tensor, mask_feature: Tensor, vert_feature: Tensor,
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
        vert_embed = self.vert_embed(decoder_out)
        # shape (num_queries, batch_size, num_inter_points * 2)
        batch_size, num_queries = mask_embed.shape[:2]

        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        vert_pred = torch.einsum('bqc,bchw->bqhw', vert_embed, vert_feature)

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

        return cls_pred, mask_pred, vert_pred, attn_mask

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
        decoded_features, multi_scale_memorys = self.pixel_decoder(x)

        if self.poly_cfg.get('separate_mask_features', False):
            mask_features = self.mask_feat_proj(decoded_features)
        else:
            mask_features = decoded_features

        vert_features = self.vert_feat_proj(decoded_features)


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
        vert_pred_list = []

        cls_pred, mask_pred, vert_pred, attn_mask = self._forward_head(
            query_feat, mask_features, vert_features, multi_scale_memorys[0].shape[-2:], img_size,
            # return_poly=True
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        vert_pred_list.append(vert_pred)

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
            cls_pred, mask_pred, vert_pred, attn_mask = self._forward_head(
                query_feat, mask_features, vert_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:], img_size,
                # return_poly = i == (self.num_transformer_decoder_layers - 1)
                # return_poly = True
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            vert_pred_list.append(vert_pred)


        pred_results = dict(
            cls_pred=cls_pred_list,
            mask_pred=mask_pred_list,
            vert_pred=vert_pred_list
        )

        if return_poly:

            mask_pred = mask_pred_list[-1]
            cls_pred = cls_pred_list[-1]
            scale = self.poly_cfg.get('polygonized_scale', 1.)
            B, Q, H, W = mask_pred.shape
            assert B == 1
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
            scores = F.softmax(cls_pred, dim=-1)[:,:,0]
            valid_mask = (scores > thre) & (poly_mask.sum(dim=[2,3]) > 0)
            valid_poly_mask = poly_mask.detach()[valid_mask]

            poly_pred_jsons = self.polygonize_mask(
                valid_poly_mask,
                scale=4. / scale,
                sample_points=False
            )
            pred_results['poly_pred_jsons'] = poly_pred_jsons

        return pred_results

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
        img_shape = batch_img_metas[0]['batch_input_shape']
        ori_shape = batch_img_metas[0]['ori_shape']
        scale = ori_shape[0] / img_shape[0]
        # all_cls_scores, all_mask_preds, all_poly_preds
        pred_results = self(x, batch_data_samples, return_poly=True)

        # mask_cls_results = all_cls_scores[-1]
        # mask_pred_results = all_mask_preds[-1]

        mask_cls_results = pred_results['cls_pred'][-1]
        mask_pred_results = pred_results['mask_pred'][-1]
        vert_pred_results = pred_results['vert_pred'][-1].sigmoid()
        B, Q, _ = mask_cls_results.shape
        poly_pred_results = [[[[0,0,0,0,0,0]]] * Q] * B

        scores = F.softmax(mask_cls_results, dim=-1)[...,0]
        vert_pred_results = vert_pred_results[scores > 0.1].unsqueeze(0)
        vert_pred_results[vert_pred_results < 0.2] = 0

        if 'seg_idxes' in pred_results:
            poly_pred = pred_results['poly_pred']
            prim_reg_pred = pred_results['prim_reg_pred'] * scale
            seg_idxes = pred_results['seg_idxes']
            seg_sizes = pred_results['seg_sizes']
            query_idxes = pred_results['query_idxes']
            poly_pred_jsons = pred_results['poly_pred_jsons']
            sampled_segs = torch.tensor(pred_results['sampled_segs']) * scale

            import time
            t0 = time.time()

            rings, poly2ring_idxes = polygon_utils.assemble_segments(
                prim_reg_pred.cpu(), seg_idxes, seg_sizes,
                length=self.poly_cfg.get('num_inter_points', 96),
                stride=self.poly_cfg.get('stride_size', 64),
            )
            rings = [ring.to(poly_pred.device) for ring in rings]

            sampled_rings, _ = polygon_utils.assemble_segments(
                sampled_segs, seg_idxes, seg_sizes,
                length=self.poly_cfg.get('num_inter_points', 96),
                stride=self.poly_cfg.get('stride_size', 64),
            )
            sampled_rings = [ring.to(poly_pred.device) for ring in sampled_rings]

            simp_rings = polygon_utils.simplify_rings_dp(
                rings, lam=self.poly_cfg.get('lam', 4), device=x[0].device,
                ref_rings=sampled_rings
            )
            simp_polygons = polygon_utils.assemble_rings(
                simp_rings, poly2ring_idxes
            )

            # print(f'{time.time() - t0}')
            for idx, polygon in zip(query_idxes, simp_polygons):
                if len(polygon) == 0 or len(polygon[0]) < 6:
                    polygon = [[0,0,0,0,0,0]]
                poly_pred_results[idx[0]][idx[1]] = polygon

        elif 'poly_jsons' in pred_results and 'poly_jsons' in pred_results:
            poly_jsons = pred_results['poly_jsons']
            query_idxes = pred_results['query_idxes']

            if self.poly_cfg.get('poly_decode_type', 'dp') == 'dp':
                pred_polygons = polygon_utils.simplify_poly_jsons(
                    poly_jsons, step_size=self.poly_cfg.get('step_size'),
                    scale=scale, lam=self.poly_cfg.get('lam', 4), device=x[0].device
                )
            elif self.poly_cfg.get('poly_decode_type', 'dp') == 'none':

                pred_polygons = []
                for i, poly_json in enumerate(poly_jsons):
                    cur_pred_poly = polygon_utils.poly_json2coco(poly_json, scale=scale)
                    pred_polygons.append(cur_pred_poly)

            for idx, cur_polygon in zip(query_idxes, pred_polygons):
                if cur_polygon == [] or len(cur_polygon[0]) < 6:
                    cur_polygon = [[0,0,0,0,0,0]]
                poly_pred_results[idx[0]][idx[1]] = cur_polygon

        results = dict(
            mask_cls_results=mask_cls_results,
            mask_pred_results=mask_pred_results,
            poly_pred_results=poly_pred_results,
            vert_pred_results=vert_pred_results
        )
        return results
