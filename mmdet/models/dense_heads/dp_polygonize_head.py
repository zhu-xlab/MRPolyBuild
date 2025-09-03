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

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean, InstanceList
from mmdet.registry import MODELS, TASK_UTILS
from ..layers import Mask2FormerTransformerDecoder, SinePositionalEncoding, PolyFormerTransformerDecoder
import mmdet.utils.tanmlh_polygon_utils as polygon_utils
from mmdet.utils import tanmlh_utils

from ..utils import get_point_coords_around_ring, get_point_coords_around_ring_v2

@MODELS.register_module()
class DPPolygonizeHead(nn.Module):

    def __init__(self, poly_cfg, decoder=None, feat_channels=256, loss_dice_wn=None,
                 loss_poly_reg=None, loss_poly_cls=None, loss_poly_right_ang=None,
                 loss_poly_ang=None, loss_poly_dp=None):
        super().__init__()

        self.poly_cfg = poly_cfg

        self.positional_encoding = SinePositionalEncoding(num_feats=128, normalize=True)
        self.decoder = None
        if decoder is not None:
            self.decoder = Mask2FormerTransformerDecoder(**decoder)
            self.num_decoder_layers = decoder.num_layers
            self.poly_reg_head = nn.Sequential(
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, 2)
            )
            if self.poly_cfg.get('apply_cls', False):
                self.poly_cls_head = nn.Sequential(
                    nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                    nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                    nn.Linear(feat_channels, 2)
                )

            self.poly_embed = nn.Linear(2, feat_channels)
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='PointL1Cost', weight=1.),
                ]
            )
            self.assigner = TASK_UTILS.build(assigner)
            self.feat_channels = feat_channels

            if loss_dice_wn is not None:
                self.loss_dice_wn = MODELS.build(loss_dice_wn)

            if loss_poly_reg is not None:
                self.loss_poly_reg = MODELS.build(loss_poly_reg)

            if loss_poly_dp is not None:
                self.loss_poly_dp = MODELS.build(loss_poly_dp)

            if loss_poly_cls is not None and self.poly_cfg.get('apply_cls', False):
                self.loss_poly_cls = MODELS.build(loss_poly_cls)

            if loss_poly_right_ang is not None and self.poly_cfg.get('apply_right_angle_loss', False):
                self.loss_poly_right_ang = MODELS.build(loss_poly_right_ang)

            if loss_poly_ang is not None and self.poly_cfg.get('apply_angle_loss', False):
                self.loss_poly_ang = MODELS.build(loss_poly_ang)

    def loss(
        self, pred_jsons, gt_jsons, W, device='cpu', points_coords=None,
        point_targets=None, mask_targets=None, **kwargs
    ):

        assert len(pred_jsons) == len(gt_jsons)
        N = self.poly_cfg.get('num_inter_points', 96)
        K = len(pred_jsons)

        assert K > 0

        sampled_rings, _, _ = polygon_utils.sample_rings_from_json(
            pred_jsons, interval=self.poly_cfg.get('step_size'), only_exterior=True,
            num_min_bins=self.poly_cfg.get('num_min_bins', 8),
            num_bins=self.poly_cfg.get('num_bins', None),
            sample_type=self.poly_cfg.get('sample_type', 'interpolate')
        )
        sampled_segments, is_complete = polygon_utils.sample_segments_from_rings(sampled_rings, self.poly_cfg.get('num_inter_points'))

        prim_reg_targets = torch.zeros(K, N, 2, device=device)
        prim_cls_targets = torch.zeros(K, N, dtype=torch.long, device=device)

        sampled_segments = sampled_segments.to(device)

        poly_pred_results = self.forward(sampled_segments, W, **kwargs)

        prim_reg_pred = poly_pred_results['prim_reg_pred']
        if self.poly_cfg.get('apply_cls', False):
            prim_cls_pred = poly_pred_results['prim_cls_pred']

        losses = dict()

        match_idxes = []
        seg_inds = []
        matched_masks = []
        dp_points = None
        for i in range(K):
            prim_target = self._get_poly_targets_single(
                prim_reg_pred[i].detach().cpu(), gt_jsons[i],
                sampled_segments=sampled_segments[i].cpu()
            )
            prim_reg_targets[i] = prim_target['prim_reg_targets']
            prim_cls_targets[i] = prim_target['prim_cls_targets']
            if 'seg_inds' in prim_target:
                seg_inds.append(prim_target['seg_inds'])

            if 'matched_mask' in prim_target:
                matched_masks.append(prim_target['matched_mask'])

            if is_complete[i]:
                seg_mask = (sampled_segments[i] >= 0).all(dim=-1)
                pred_poly = shapely.geometry.Polygon(sampled_segments[i][seg_mask].tolist())
                gt_poly = shapely.geometry.Polygon(gt_jsons[i]['coordinates'][0])
                iou = polygon_utils.polygon_iou(pred_poly, gt_poly)
                if iou > self.poly_cfg.get('align_iou_thre', 0.5):
                    match_idxes.append(i)

        match_idxes = torch.tensor(match_idxes)

        sizes = (prim_reg_pred >= 0).all(dim=-1).sum(dim=1)

        # decoded_rings = polygon_utils.batch_decode_ring_dp(prim_reg_pred, sizes, max_step_size=64, lam=4, device=prim_reg_pred.device)
        if self.poly_cfg.get('apply_poly_iou_loss', False) or self.poly_cfg.get('apply_poly_dis_loss', False) \
           or self.poly_cfg.get('apply_right_angle_loss', False):
            # prim_reg_pred = torch.cat([prim_reg_pred, prim_reg_pred[:, :1]], dim=1)
            if K > 0:
                dp, dp_points = polygon_utils.batch_decode_ring_dp(
                    prim_reg_pred, sizes, max_step_size=sizes.max(),
                    lam=self.poly_cfg.get('lam', 4),
                    device=device, return_both=True,
                    result_device=device
                )
                dp_points = [x[:-1] for x in dp_points]
        else:
            dp = polygon_utils.batch_decode_ring_dp(
                prim_reg_pred, sizes, max_step_size=sizes.max(),
                lam=self.poly_cfg.get('lam', 4),
                device=device, only_return_dp=True
            )

        # opt_dis = torch.gather(dp[:,0], 1, sizes.unsqueeze(1)-1)

        opt_dis_comp = torch.gather(dp[is_complete], 2, sizes[is_complete].unsqueeze(1).unsqueeze(1).repeat(1,N,1)).min(dim=1)[0]
        opt_dis_incomp = torch.gather(dp[~is_complete, 0], 1, sizes[~is_complete].unsqueeze(1)-1)
        opt_dis = torch.cat([opt_dis_comp, opt_dis_incomp])
        avg_factor = reduce_mean(opt_dis.new_tensor(len(opt_dis)))
        # losses['loss_dp'] = self.loss_poly_dp(opt_dis, torch.zeros_like(opt_dis), avg_factor=avg_factor)
        losses['loss_dp'] = (opt_dis_comp.sum() + opt_dis_incomp.sum()) / K * self.poly_cfg.get('loss_weight_dp', 0.01)

        if self.poly_cfg.get('apply_poly_iou_loss', False):
            if len(match_idxes) > 0:
                if points_coords is None:
                    # sample points and point_targets
                    sampled_coords = []
                    # point_targets = []
                    for i in match_idxes:
                        pred_ring = dp_points[i]
                        # sampled_coord = get_point_coords_around_ring(pred_ring, W)
                        temp = prim_reg_pred[i][(prim_reg_pred[i] >= 0).all(dim=-1)]
                        sampled_coord = get_point_coords_around_ring_v2(
                            temp.detach(), W, num_samples=1024,
                            max_offsets=self.poly_cfg.get('max_sample_offsets', 5)
                        )
                        # point_target = mask_targets[i, sampled_coord[:,1], sampled_coord[:,0]]

                        sampled_coords.append(sampled_coord)
                        # point_targets.append(point_target)
                    sampled_coords = torch.stack(sampled_coords, dim=0) if len(sampled_coords) > 0 else torch.zeros(0, 1024, 2, device=prim_reg_pred.device)
                    point_targets = point_sample(mask_targets[match_idxes].float().unsqueeze(1), sampled_coords / W, align_corners=False).squeeze(1)

                else:
                    pdb.set_trace()
                    sampled_coords = points_coords[match_idxes] * W
                    point_targets = point_targets[match_idxes]

                loss_poly_iou = prim_reg_pred[:0].sum()
                wn_list = []
                gt_wn_list = []
                # for i, (pred_ring, sampled_coords) in enumerate(zip(dp_points, points_coords)):
                for i, idx in enumerate(match_idxes):
                    pred_ring = dp_points[idx]
                    # gt_ring = torch.tensor(gt_jsons[idx]['coordinates'][0], device=pred_ring.device)[:-1]
                    wn = polygon_utils.cal_winding_number(pred_ring, sampled_coords[i], c=1)
                    # gt_wn = polygon_utils.cal_winding_number(gt_ring, sampled_coords[i], c=1) > 0.5
                    # gt_wn = polygon_utils.cal_winding_number(gt_ring, sampled_coords[i], c=1)

                    wn_list.append(wn)
                    # gt_wn_list.append(gt_wn)

                """
                vis_data = {
                    'polygons_and_points': dict(
                        polygons=[shapely.geometry.Polygon(dp_points[x].tolist()) for x in match_idxes],
                        points=[x.detach().cpu().numpy() for x in sampled_coords],
                        point_labels=[(x > 0.5).long().cpu().numpy() for x in wn_list]
                    )
                }
                polygon_utils.vis_data_wandb(vis_data)
                """

                wns = torch.stack(wn_list, dim=0)
                # gt_wns = torch.cat(point_targets, dim=0).unsqueeze(1)
                # gt_wns = torch.stack(gt_wn_list, dim=0)
                # ((torch.cat(gt_wn_list) > 0.5) == gt_wns[:,0]).sum()
                loss_poly_iou = self.loss_dice_wn(wns, point_targets)
                # loss_poly_iou = self.loss_dice_wn(wns, gt_wns)
                # loss_poly_iou = (wns - gt_wns).abs().mean()
                losses['loss_poly_iou'] = loss_poly_iou
            else:
                losses['loss_poly_iou'] = prim_reg_pred[:0].sum()

        # Polygon regression
        A = prim_reg_pred.reshape(-1, 2)
        B = prim_reg_targets.view(-1, 2)


        if self.poly_cfg.get('reg_targets_type', 'vertice') == 'contour':
            mask = (poly_pred >= 0).all(dim=-1).view(-1)
            avg_factor = reduce_mean(A.new_tensor(mask.sum().item() * 2))
            loss_poly_reg = self.loss_poly_reg(A[mask], B[mask], avg_factor=avg_factor)

        elif self.poly_cfg.get('reg_targets_type', 'vertice') == 'vertice':
            mask = (prim_reg_targets >= 0).all(dim=-1).view(-1)
            avg_factor = reduce_mean(A.new_tensor(mask.sum().item() * 2))
            loss_poly_reg = self.loss_poly_reg(A[mask], B[mask], avg_factor=avg_factor)
        else:
            raise ValueError()

        losses['loss_poly_reg'] = loss_poly_reg

        if self.poly_cfg.get('apply_cls', False):
            mask = (sampled_segments >= 0).all(dim=-1).view(-1)
            A = prim_cls_pred.reshape(-1, 2)
            B = prim_cls_targets.view(-1)
            loss_poly_cls = self.loss_poly_cls(A[mask], B[mask])
            losses['loss_poly_cls'] = loss_poly_cls

        if self.poly_cfg.get('apply_right_angle_loss', False):
            loss_right_ang = prim_reg_pred[:0].sum()


            """
            diffs = []
            eps = 1e-6
            for i, idx in enumerate(match_idxes):
                if len(dp_points[idx]) >= 3:
                    v = dp_points[idx]
                    u = torch.roll(v, shifts=[1], dims=[0])
                    w = torch.roll(v, shifts=[-1], dims=[0])

                    vec_1 = u - v
                    vec_2 = w - v
                    unit_vec_1 = vec_1 / (torch.norm(vec_1, dim=1, keepdim=True) + eps)
                    unit_vec_2 = vec_2 / (torch.norm(vec_2, dim=1, keepdim=True) + eps)
                    dot_product = torch.sum(unit_vec_1 * unit_vec_2, dim=1)
                    angle = torch.acos(dot_product.clamp(-1+eps, 1-eps))
                    target_angle = torch.tensor([torch.pi / 2, torch.pi], device=angle.device).unsqueeze(0)

                    diff = (angle.unsqueeze(1) - target_angle).abs().min(dim=1)[0]
                    diffs.append(diff)

            if len(diffs) > 0:
                diffs = torch.cat(diffs)
                loss_right_ang = self.loss_poly_right_ang(diffs, torch.zeros_like(diffs))
            """

            angles = []
            gt_angles = []
            eps = 1e-6
            num_base_angles = self.poly_cfg.get('num_base_angles', 16)
            base_angles = tanmlh_utils.generate_angles(num_base_angles).to(device)

            for i, idx in enumerate(match_idxes):
                if len(dp_points[idx]) >= 3:
                    angle = tanmlh_utils.get_angles(dp_points[idx])
                    gt_angle = tanmlh_utils.get_angles(torch.tensor(gt_jsons[idx]['coordinates'][0], device=device))

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
                # loss_right_ang = self.loss_poly_right_ang(diffs, torch.zeros_like(diffs))

            losses['loss_poly_right_ang'] = loss_right_ang

        if self.poly_cfg.get('apply_angle_loss', False):
            loss_ang = prim_reg_pred[:0].sum()
            diffs = []
            for i in range(K):
                cur_inds = seg_inds[i]
                cur_mask = matched_masks[i]
                cur_pred = prim_reg_pred[i][cur_inds]

                cur_target = prim_reg_targets[i][cur_inds]
                cur_angle_mask = torch.zeros_like(cur_mask, device=cur_pred.device)
                cur_angle_mask[1:-1] = cur_mask[:-2] & cur_mask[1:-1] & cur_mask[2:]

                pred_angle, pred_angle_mask = polygon_utils.calculate_polygon_angles(cur_pred)
                target_angle, target_angle_mask = polygon_utils.calculate_polygon_angles(cur_target)

                cur_mask = cur_angle_mask & pred_angle_mask & target_angle_mask
                # self.loss_poly_ang(pred_angle[cur_mask], target_angle[cur_mask])
                if cur_mask.any():
                    max_diff = (pred_angle[cur_mask] - target_angle[cur_mask]).abs().max()
                    diffs.append(max_diff)

            if len(diffs) > 0:
                diffs = torch.stack(diffs)
                avg_factor = reduce_mean(diffs.new_tensor(len(diffs)))
                # loss_ang = self.loss_poly_ang(diffs, torch.zeros_like(diffs), avg_factor=avg_factor)
                # loss_ang = torch.stack(diffs).mean() * self.loss_poly_ang.loss_weight
                loss_ang = diffs.mean() * self.loss_poly_ang.loss_weight

            losses['loss_poly_ang'] = loss_ang

        if self.poly_cfg.get('apply_poly_dis_loss', False):
            if len(match_idxes) > 0:

                loss_poly_dis = prim_reg_pred[:0].sum()
                pred_sampled = []
                gt_sampled = []

                for i, idx in enumerate(match_idxes):
                    pred_ring = dp_points[idx]
                    gt_ring = torch.tensor(gt_jsons[idx]['coordinates'][0], device=pred_ring.device)[:-1]

                    interval = self.poly_cfg.get('step_size', 4)
                    segment_lengths = torch.sqrt(((pred_ring - torch.roll(pred_ring, shifts=[-1], dims=[0]))**2).sum(dim=1))
                    perimeter = segment_lengths.sum().item()
                    num_bins = round(perimeter / interval)

                    A = polygon_utils.interpolate_ring(
                        pred_ring, num_bins=num_bins,
                        num_min_bins=self.poly_cfg.get('num_min_bins', 16),
                        drop_last=False
                    )
                    B = polygon_utils.interpolate_ring(
                        gt_ring, num_bins=num_bins,
                        num_min_bins=self.poly_cfg.get('num_min_bins', 16),
                        drop_last=False
                    )
                    B_align = polygon_utils.align_rings_by_roll(A, B, True)
                    pred_sampled.append(A)
                    gt_sampled.append(B_align)

                pred_sampled = torch.cat(pred_sampled)
                gt_sampled = torch.cat(gt_sampled)
                loss_poly_dis = self.loss_poly_reg(pred_sampled, gt_sampled)
                # loss_poly_iou = self.loss_dice_wn(wns, gt_wns)
                # pdb.set_trace()
                # loss_poly_iou = (wns - gt_wns).abs().mean()
                losses['loss_poly_dis'] = loss_poly_dis
            else:
                losses['loss_poly_dis'] = prim_reg_pred[:0].sum()


        return losses

    def forward(self, poly_pred, W, mask_feat=None, query_feat=None, batch_idxes=None):

        results = dict()

        K, N, _ = poly_pred.shape
        C = self.feat_channels

        norm_poly_pred = (poly_pred / W - 0.5) * 2
        poly_valid_mask = (poly_pred >= 0).all(dim=-1)
        poly_feat = self.poly_embed(norm_poly_pred).view(K, N, C)

        if mask_feat is not None:
            point_feat_list = []
            for i, cur_mask_feat in enumerate(mask_feat):
                cur_norm_poly_pred = norm_poly_pred[batch_idxes == i].unsqueeze(0)

                point_feat = F.grid_sample(
                    cur_mask_feat[None], cur_norm_poly_pred, align_corners=True
                )
                point_feat = point_feat.permute(0,2,3,1).squeeze(0)
                point_feat_list.append(point_feat)

            point_feat = torch.cat(point_feat_list, dim=0)
            poly_feat += point_feat

            if self.poly_cfg.get('use_decoded_feat_in_poly_feat', False):
                poly_feat += query_feat.detach().view(K, 1, C)

        poly_pos_embed = self.positional_encoding(poly_feat.new_zeros(K, N, 1))
        poly_pos_embed = poly_pos_embed.view(K, C, N).permute(0,2,1)
        # poly_pos_embed += ((torch.arange(N, device=poly_pred.device) / N - 0.5) * 2).view(1,-1,1)

        query_feat = poly_feat
        query_embed = poly_pos_embed

        prim_pred_reg_list = []
        if self.poly_cfg.get('apply_cls', False):
            prim_pred_cls_list = []
        for i in range(self.num_decoder_layers):
            layer = self.decoder.layers[i]
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

            if i == self.num_decoder_layers - 1:

                prim_pred_reg = self.poly_reg_head(query_feat).view(K, N, -1)
                prim_pred_reg_list.append(prim_pred_reg)

                if self.poly_cfg.get('apply_cls', False):
                    prim_pred_cls = self.poly_cls_head(query_feat).view(K, N, -1)
                    prim_pred_cls_list.append(prim_pred_cls)

        prim_pred_reg = prim_pred_reg_list[-1]
        if self.poly_cfg.get('apply_cls', False):
            prim_pred_cls = prim_pred_cls_list[-1]

        prim_pred_reg = poly_pred + prim_pred_reg * self.poly_cfg.get('max_offsets', 10)
        prim_pred_reg = torch.clamp(prim_pred_reg, 0, W)
        prim_pred_reg[(poly_pred < 0).all(dim=-1)] = -1

        results['prim_reg_pred'] = prim_pred_reg
        if self.poly_cfg.get('apply_cls', False):
            results['prim_cls_pred'] = prim_pred_cls

        return results

    def _get_poly_targets_single(self, poly_pred, poly_gt_json, sampled_segments):

        targets = {}

        N = self.poly_cfg.get('num_inter_points', 96)
        max_align_dis = self.poly_cfg.get('max_align_dis', 1e8)

        prim_reg_targets = torch.zeros(N, 2) - 1
        prim_cls_targets = torch.zeros(N, dtype=torch.long)
        prim_ref_targets = torch.zeros(N, 2) - 1

        K = (sampled_segments >= 0).all(dim=-1).sum()

        poly_gt_torch = torch.tensor(poly_gt_json['coordinates'][0]).float() # use the exterior
        if self.poly_cfg.get('add_gt_middle', False):
            poly_gt_torch = polygon_utils.add_middle_points(poly_gt_torch)

        if K == 0 or (poly_gt_torch == 0).all():
            targets['prim_cls_targets'] = prim_cls_targets
            targets['prim_reg_targets'] = prim_reg_targets
            return targets

        gt_instances = InstanceData(
            labels=torch.zeros(len(poly_gt_torch[:-1]), dtype=torch.long),
            points=poly_gt_torch[:-1]
        ) # (num_classes, N)

        pred_instances = InstanceData(points=sampled_segments[:K])

        assign_result = self.assigner.assign(
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
        prim_cls_targets[seg_inds[valid_mask]] = 1

        targets['prim_cls_targets'] = prim_cls_targets
        targets['prim_reg_targets'] = prim_reg_targets
        targets['seg_inds'] = seg_inds
        targets['matched_mask'] = valid_mask

        return targets


    def predict(self, poly_pred_jsons, W, mask_features=None, batch_idxes=None, device='cpu',
                return_format='coco'):

        N = self.poly_cfg.get('num_inter_points', 96)
        num_max_rings = self.poly_cfg.get('num_max_rings', 5000)

        pred_results = {}

        sampled_segs, seg_sizes, poly2segs_idxes, segs2poly_idxes = polygon_utils.sample_segments_from_json(
            poly_pred_jsons, interval=self.poly_cfg.get('step_size'),
            seg_len=N, stride=self.poly_cfg.get('stride_size', 64),
            num_min_bins=self.poly_cfg.get('num_min_bins', 8),
            num_bins=self.poly_cfg.get('num_bins', None),
        )
        sampled_segs = sampled_segs.astype(np.float32)


        if len(sampled_segs) > 0:
            # query_idx = topk_mask.view(-1).nonzero().view(-1)[poly2mask_idxes[segs2poly_idxes[:,0]]]
            # query_feat = query_feat_list[-1].view(B*Q, -1)[query_idx]
            poly_pred = torch.from_numpy(sampled_segs).to(device).float()
            # norm_poly_pred = (poly_pred / (W * 4) - 0.5) * 2

            # batch_idxes = torch.zeros(len(poly_pred), dtype=torch.long)
            # batch_idxes = poly2mask_idxes[segs2poly_idxes[:,0]] // max_per_image

            poly_pred_list = poly_pred.split(num_max_rings)
            # mask_features_list = mask_features.split(num_max_rings)
            # batch_idxes_list = batch_idxes[segs2poly_idxes[:,0]].split(num_max_rings)
            segs2poly_idxes_list = torch.tensor(segs2poly_idxes[:,0]).split(num_max_rings)

            prim_reg_pred_list = []
            for i, (poly_pred, segs2poly_idxes) in enumerate(zip(poly_pred_list, segs2poly_idxes_list)):
                dp_pred_results = self.forward(
                    poly_pred, W, mask_feat=mask_features,
                    batch_idxes=batch_idxes[segs2poly_idxes] if batch_idxes is not None else None
                )
                prim_reg_pred = dp_pred_results['prim_reg_pred']
                prim_reg_pred_list.append(prim_reg_pred)

            prim_reg_pred = torch.cat(prim_reg_pred_list)

            if self.poly_cfg.get('apply_cls', False):
                prim_cls_pred = dp_pred_results['prim_cls_pred'].cpu()
            else:
                prim_cls_pred = torch.zeros_like(prim_reg_pred).cpu()


            rings, poly2ring_idxes, others = polygon_utils.assemble_segments(
                prim_reg_pred.cpu(), poly2segs_idxes, seg_sizes,
                length=self.poly_cfg.get('num_inter_points', 96),
                stride=self.poly_cfg.get('stride_size', 64),
                prim_cls_pred=prim_cls_pred,
                sampled_rings=sampled_segs
            )

            pred_results['pred_rings'] = rings
            pred_results['sampled_rings'] = others['sampled_rings']

            rings = [ring.to(poly_pred.device) for ring in rings]
            # rings = [torch.cat([ring, ring[:1]], dim=0) for ring in rings]

            simp_rings = polygon_utils.simplify_rings_dp(
                rings, lam=self.poly_cfg.get('lam', 4), device=device,
                ref_rings=sampled_rings if self.poly_cfg.get('use_ref_rings', False) else None,
                drop_last=False, max_step_size=self.poly_cfg.get('max_step_size', 50)
            )

            simp_rings = [x[:-1] for x in simp_rings]

            simp_polygons = polygon_utils.assemble_rings(
                simp_rings, poly2ring_idxes, format=return_format
            )

            pred_results['simp_polygons'] = simp_polygons

        else:
            pred_results['simp_polygons'] = []

        return pred_results


