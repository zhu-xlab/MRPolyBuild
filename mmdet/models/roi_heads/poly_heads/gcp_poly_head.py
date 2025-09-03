# Copyright (c) OpenMMLab. All rights reserved.
import time
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
from mmdet.structures.mask import PolygonMasks
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
import mmdet.utils.tanmlh_polygon_utils as polygon_utils
from mmdet.models.layers import Mask2FormerTransformerDecoder, SinePositionalEncoding, PolyFormerTransformerDecoder
from mmdet.models.utils import get_point_coords_around_ring, get_point_coords_around_ring_v2
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean, InstanceList, tanmlh_utils

@MODELS.register_module()
class GCPPolyHead(nn.Module):

    def __init__(self, poly_cfg, in_feat_channels=2, decoder=None, feat_channels=256,
                 loss_poly_reg=None, loss_poly_right_ang=None,
                 loss_poly_ang=None):
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

            self.poly_embed = nn.Linear(2, feat_channels)
            self.poly_feat_embed = nn.Linear(in_feat_channels, feat_channels)
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='PointL1Cost', weight=1.),
                ],
                solver='lapsolver'
                # solver='scipy'
            )
            self.assigner = TASK_UTILS.build(assigner)
            self.feat_channels = feat_channels

            if loss_poly_reg is not None:
                self.loss_poly_reg = MODELS.build(loss_poly_reg)

            if loss_poly_right_ang is not None and self.poly_cfg.get('apply_right_angle_loss', False):
                self.loss_poly_right_ang = MODELS.build(loss_poly_right_ang)

            if loss_poly_ang is not None and self.poly_cfg.get('apply_angle_loss', False):
                self.loss_poly_ang = MODELS.build(loss_poly_ang)

    def loss(self, pred_jsons, gt_jsons, W, device='cpu', **kwargs):

        t0 = time.time()
        assert len(pred_jsons) == len(gt_jsons)
        N = self.poly_cfg.get('num_inter_points', 96)
        K = len(pred_jsons)
        num_iter = self.poly_cfg.get('num_iter', 1)

        if self.poly_cfg.get('align_pred_gt', False) is True:
            gt_jsons = polygon_utils.align_poly_json_pairs(pred_jsons, gt_jsons)

        t1 = time.time()

        if K == 0:
            ### set a dummy pair of polygons to avoid no loss outputs
            naive_poly = {'type': 'Polygon', 'coordinates': [[[0.,0.], [1.,0.], [1.,1.], [0.,1.], [0.,0.]]]}
            pred_jsons = [naive_poly]
            gt_jsons = [naive_poly]
            kwargs['batch_idxes'] = torch.zeros(1, dtype=torch.long)
            K = 1

            # dummy_loss = self.poly_embed.parameters().__next__()[:0].sum()
            # losses = dict(
            #     loss_dp=dummy_loss,
            #     loss_poly_reg=dummy_loss,
            # )
            # if self.poly_cfg.get('apply_right_angle_loss', False):
            #     losses['loss_poly_right_ang'] = dummy_loss

            # if self.poly_cfg.get('apply_angle_loss', False):
            #     losses['loss_poly_ang'] = dummy_loss

            # return losses

        sampled_rings, _, _ = polygon_utils.sample_rings_from_json(
            pred_jsons, interval=self.poly_cfg.get('step_size'), only_exterior=True,
            num_min_bins=self.poly_cfg.get('num_min_bins', 8),
            num_bins=self.poly_cfg.get('num_bins', None),
            sample_type=self.poly_cfg.get('sample_type', 'interpolate')
        )
        t2 = time.time()

        sampled_segments, is_complete = polygon_utils.sample_segments_from_rings(sampled_rings, self.poly_cfg.get('num_inter_points'))

        t3 = time.time()

        prim_reg_targets = torch.zeros(K, N, 2, device=device)
        prim_cls_targets = torch.zeros(K, N, dtype=torch.long, device=device)

        sampled_segments = sampled_segments.to(device)

        prim_reg_pred = sampled_segments

        poly_feat = self.get_init_poly_feat(prim_reg_pred, W, **kwargs)
        prim_reg_pred = self.forward(prim_reg_pred, poly_feat, W)

        t4 = time.time()

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
                # seg_mask = (sampled_segments[i] >= 0).all(dim=-1)
                # pred_poly = shapely.geometry.Polygon(sampled_segments[i][seg_mask].tolist())
                # gt_poly = shapely.geometry.Polygon(gt_jsons[i]['coordinates'][0])
                # iou = polygon_utils.polygon_iou(pred_poly, gt_poly)
                # if iou > self.poly_cfg.get('align_iou_thre', 0.5):
                match_idxes.append(i)

        match_idxes = torch.tensor(match_idxes)
        t5 = time.time()

        sizes = (prim_reg_pred >= 0).all(dim=-1).sum(dim=1)

        # decoded_rings = polygon_utils.batch_decode_ring_dp(prim_reg_pred, sizes, max_step_size=64, lam=4, device=prim_reg_pred.device)
        if self.poly_cfg.get('apply_right_angle_loss', False):
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
        t6 = time.time()

        opt_dis_comp = torch.gather(dp[is_complete], 2, sizes[is_complete].unsqueeze(1).unsqueeze(1).repeat(1,N,1)).min(dim=1)[0]
        opt_dis_incomp = torch.gather(dp[~is_complete, 0], 1, sizes[~is_complete].unsqueeze(1)-1)
        opt_dis = torch.cat([opt_dis_comp, opt_dis_incomp])
        avg_factor = reduce_mean(opt_dis.new_tensor(len(opt_dis)))
        losses['loss_dp'] = (opt_dis_comp.sum() + opt_dis_incomp.sum()) / K * self.poly_cfg.get('loss_weight_dp', 0.01)

        t7 = time.time()

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

        if self.poly_cfg.get('apply_right_angle_loss', False):
            loss_right_ang = prim_reg_pred[:0].sum()

            angles = []
            gt_angles = []
            eps = 1e-5
            num_base_angles = self.poly_cfg.get('num_base_angles', 16)
            base_angles = tanmlh_utils.generate_angles(num_base_angles).to(device)

            # pred_points = dp_points
            pred_points = [x[(x >= 0).all(dim=-1)] for x in prim_reg_pred]

            for i, idx in enumerate(match_idxes):
                if len(pred_points[idx]) >= 3:
                    angle = tanmlh_utils.get_angles(pred_points[idx])
                    gt_angle = tanmlh_utils.get_angles(torch.tensor(gt_jsons[idx]['coordinates'][0], device=device)[:-1])

                    angles.append(angle)
                    gt_angles.append(gt_angle)

            if len(angles) > 0:
                # angles = torch.cat(angles)
                sum_min_ang_dis_list = []

                """
                gt_ang_idx_list = []
                for angle in gt_angles:
                    diff = angle.view(-1,1,1) - base_angles.unsqueeze(0)
                    cos_sim = torch.cos(diff)  # 直接使用角度差cos值
                    gt_ang_idx = cos_sim.abs().max(dim=-1)[0].sum(dim=0).argmax()

                    # group_score = cos_sim.abs().sum(dim=0)  # 按组求和
                    # gt_ang_idx = group_score.max(dim=1)[0].argmax()  # 选择最相似组


                    # d1 = (diff.abs() % (torch.pi * 2))
                    # d2 = 2 * torch.pi - d1
                    # min_ang_dis = torch.where(d1 < d2, d1, d2)
                    # gt_ang_idx = min_ang_dis.min(dim=-1)[0].sum(dim=0).argmin()

                    gt_ang_idx_list.append(gt_ang_idx)

                for j, angle in enumerate(angles):
                    cur_base_angle = base_angles[gt_ang_idx_list[j]]
                    cos_sim = torch.cos(angle.unsqueeze(1) - cur_base_angle.unsqueeze(0))
                    sum_min_ang_dis = cos_sim.abs().max(dim=1)[0].mean()

                    # diff = angle.view(-1,1,1) - base_angles.unsqueeze(0)
                    # d1 = (diff.abs() % (torch.pi * 2))
                    # d2 = 2 * torch.pi - (diff.abs() % (torch.pi * 2))
                    # min_ang_dis = torch.where(d1 < d2, d1, d2)
                    # min_ang_dis[:, gt_ang_idx_list[j]].min(dim=-1)[0]

                    # sum_min_ang_dis = min_ang_dis[:, gt_ang_idx_list[j]].min(dim=-1)[0].mean()
                    # sum_min_ang_dis = min_ang_dis.min(dim=-1)[0].sum(dim=0).min()

                    sum_min_ang_dis_list.append(sum_min_ang_dis)
                """

                for angle, gt_angle in zip(angles, gt_angles):
                    diff_gt = angle.view(-1,1,1) - base_angles.unsqueeze(0)
                    cos_sim_gt = torch.cos(diff_gt)  # 直接使用角度差cos值
                    gt_ang_idx = cos_sim_gt.abs().max(dim=-1)[0].sum(dim=0).argmax()

                    cur_base_angle = base_angles[gt_ang_idx]
                    cos_sim_pred = torch.cos(angle.unsqueeze(1) - cur_base_angle.unsqueeze(0))
                    sum_min_ang_dis = cos_sim_pred.abs().max(dim=1)[0].mean()
                    sum_min_ang_dis_list.append(sum_min_ang_dis)

                sum_min_ang_dis = torch.stack(sum_min_ang_dis_list)
                loss_right_ang = (1 - sum_min_ang_dis.mean()) * self.loss_poly_right_ang.loss_weight
                # loss_right_ang = self.loss_poly_right_ang(sum_min_ang_dis, torch.zeros_like(sum_min_ang_dis))
                # loss_right_ang = self.loss_poly_right_ang(diffs, torch.zeros_like(diffs))

            losses['loss_poly_right_ang'] = loss_right_ang

        t8 = time.time()

        if self.poly_cfg.get('apply_right_angle_loss_v2', False):
            loss_right_ang = prim_reg_pred[:0].sum()
            pred_points = [x[(x >= 0).all(dim=-1)] for x in prim_reg_pred]
            for i, idx in enumerate(match_idxes):
                if len(pred_points[idx]) >= 3:
                    angle1 = tanmlh_utils.get_angles(pred_points[idx])
                    angle2 = tanmlh_utils.get_angles_v2(pred_points[idx])
                    pdb.set_trace()

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

        # print(f'GCP loss time: {t1-t0} {t2-t1} {t3-t2} {t4-t3} {t5-t4} {t6-t5} {t7-t6} {t8-t7}')

        return losses

    @staticmethod
    def vectorized_normalization(poly_pred):
        """
        Vectorized normalization of polygon coordinates

        Args:
            poly_pred: Tensor of shape (K, N, 2) representing K polygons with N points
        
        Returns:
            Normalized polygon tensor of shape (K, N, 2)
        """
        # Step 1: Create validity mask (points with both coordinates >= 0)
        poly_valid_mask = (poly_pred >= 0).all(dim=-1)  # (K, N)

        # Step 2: Compute min and max values for each polygon
        # Create mask for valid points only
        valid_mask = poly_valid_mask.unsqueeze(-1)  # (K, N, 1)

        # For min computation: replace invalid points with a large value
        large_value = 1e9
        min_input = torch.where(valid_mask, poly_pred, large_value)
        minv = min_input.min(dim=1)[0]  # (K, 2)

        # For max computation: replace invalid points with a small value
        small_value = -1e9
        max_input = torch.where(valid_mask, poly_pred, small_value)
        maxv = max_input.max(dim=1)[0]  # (K, 2)

        # Step 3: Compute max side length for each polygon
        ranges = maxv - minv  # (K, 2)
        max_w = ranges.max(dim=1)[0]  # (K,)

        # Handle polygons with no valid points
        has_valid = poly_valid_mask.any(dim=1)  # (K,)
        safe_max_w = torch.where(has_valid, max_w, torch.ones_like(max_w))
        safe_minv = torch.where(has_valid.unsqueeze(1), minv, torch.zeros_like(minv))

        # Step 4: Normalize coordinates
        # Center and scale
        centered = (poly_pred - safe_minv.unsqueeze(1)) / safe_max_w.unsqueeze(-1).unsqueeze(-1)

        # Shift to [-1, 1] range
        normalized = (centered - 0.5) * 2

        # Set invalid points to -2
        normalized = torch.where(valid_mask, normalized, -2.0)

        return normalized

    def get_init_poly_feat(self, poly_pred, W, mask_feat=None, batch_idxes=None, **kwargs):
        K, N, _ = poly_pred.shape
        C = self.feat_channels

        if self.poly_cfg.get('disable_mask_feat', False):
            mask_feat = torch.zeros_like(mask_feat)

        norm_poly_pred = (poly_pred / W - 0.5) * 2
        point_feat_list = []
        b, c, h, w = mask_feat.shape
        for i, cur_mask_feat in enumerate(mask_feat):
            cur_norm_poly_pred = norm_poly_pred[batch_idxes == i].unsqueeze(0)
            _, cur_K, cur_N, _ = cur_norm_poly_pred.shape

            if self.poly_cfg.get('unfold_cfg', {}) != {}:
                unfold_cfg = self.poly_cfg['unfold_cfg']
                kernel_size = unfold_cfg.get('kernel_size', 7)
                cur_norm_poly_pred = polygon_utils.sample_neighborhood_points(
                    cur_norm_poly_pred, kernel_size, 1
                ).view(1, cur_K, cur_N * kernel_size ** 2, 2)

                # unfolded = F.unfold(mask_feat, **unfold_cfg)
                # unfolded = unfolded.view(b, c*kernel_size**2, h, w)

            t2 = time.time()

            point_feat = F.grid_sample(
                cur_mask_feat[None], cur_norm_poly_pred.to(cur_mask_feat.device),
                align_corners=True
            )
            point_feat = point_feat.permute(0,2,3,1).squeeze(0).to(poly_pred.device)

            if self.poly_cfg.get('unfold_cfg', {}) != {}:
                point_feat = point_feat.reshape(cur_K, cur_N, kernel_size**2 * c)

            point_feat_list.append(point_feat)
            t3 = time.time()

        poly_feat = torch.cat(point_feat_list, dim=0)

        return poly_feat


    def forward(self, poly_pred, poly_feat, W):

        results = dict()
        K, N, _ = poly_feat.shape
        C = self.feat_channels

        centerized_poly_pred = self.vectorized_normalization(poly_pred)

        poly_feat = self.poly_feat_embed(poly_feat)
        poly_feat += self.poly_embed(centerized_poly_pred).view(K, N, C)

        poly_pos_embed = self.positional_encoding(poly_feat.new_zeros(K, N, 1))
        poly_pos_embed = poly_pos_embed.view(K, C, N).permute(0,2,1)
        # poly_pos_embed += ((torch.arange(N, device=poly_pred.device) / N - 0.5) * 2).view(1,-1,1)

        query_feat = poly_feat
        query_embed = poly_pos_embed

        prim_pred_reg_list = []
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

        prim_pred_reg = prim_pred_reg_list[-1]
        prim_pred_reg = poly_pred + prim_pred_reg * self.poly_cfg.get('max_offsets', 10)
        prim_pred_reg = torch.clamp(prim_pred_reg, 0, W)
        prim_pred_reg[(poly_pred < 0).all(dim=-1)] = -1

        return prim_pred_reg

    def _get_poly_targets_single(self, poly_pred, poly_gt_json, sampled_segments,
                                 assign_type='assigner'):

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

        if assign_type == 'assigner':

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


    def to_numba_nested_list(self, data):
        from numba.typed import List
        """
        递归地将 Python 的嵌套 list（任意深度）转换为 numba.typed.List。
        底层必须是原生可支持的类型（如 int、float），不支持 dict/对象等。
        """
        if isinstance(data, list):
            out = List()
            for item in data:
                out.append(self.to_numba_nested_list(item))
            return out
        else:
            # 基本类型（int, float, bool…）直接返回
            return data

    def predict_sample_segments(self, imgs, batch_data_samples):

        seg_probs = batch_data_samples[0].seg_probs
        pred_polys = batch_data_samples[0].pred_polys
        scores = batch_data_samples[0].scores
        B, _, H, W = seg_probs.shape
        num_max_rings = self.poly_cfg.get('num_max_rings', 5000)
        batch_idxes = batch_data_samples[0].get(
            'batch_idxes', torch.zeros(len(pred_polys), dtype=torch.long)
        )
        mask_feat_type = self.poly_cfg.get('mask_feat_type', 'img_prob')

        N = self.poly_cfg.get('num_inter_points', 96)
        t0 = time.time()

        up_imgs = F.interpolate(imgs.cpu(), (H, W))

        if mask_feat_type == 'img_prob':
            mask_feats = torch.cat([seg_probs, up_imgs], dim=1)
        elif mask_feat_type == 'prob':
            mask_feats = seg_probs

        sampled_segs, seg_sizes, poly2segs_idxes, segs2poly_idxes = polygon_utils.sample_segments_from_json(
            pred_polys, interval=self.poly_cfg.get('step_size'),
            seg_len=N, stride=self.poly_cfg.get('stride_size', 64),
            num_min_bins=self.poly_cfg.get('num_min_bins', 8),
            num_bins=self.poly_cfg.get('num_bins', None),
        )
        sampled_segs = sampled_segs.astype(np.float32)

        poly_feat_list = []
        poly_pred_list = []
        if len(sampled_segs) > 0:

            poly_pred = torch.from_numpy(sampled_segs).float()
            poly_pred_list = poly_pred.split(num_max_rings)
            segs2poly_idxes_list = torch.tensor(segs2poly_idxes[:,0]).split(num_max_rings)

            for i, (poly_pred, segs2poly_idxes) in enumerate(zip(poly_pred_list, segs2poly_idxes_list)):
                poly_feat = self.get_init_poly_feat(poly_pred, W, mask_feats, batch_idxes[segs2poly_idxes])
                # if poly_feat.sum().isnan():
                #     pdb.set_trace()
                poly_feat_list.append(poly_feat)

        batch_data_samples[0].sampled_segs = sampled_segs
        batch_data_samples[0].poly2segs_idxes = poly2segs_idxes
        batch_data_samples[0].seg_sizes = seg_sizes
        batch_data_samples[0].segs2poly_idxes = segs2poly_idxes
        batch_data_samples[0].poly_feat_list = poly_feat_list
        batch_data_samples[0].poly_pred_list = poly_pred_list

        return batch_data_samples

    def predict_gcp(self, imgs, batch_data_samples):

        seg_probs = batch_data_samples[0].seg_probs
        poly_pred_list = batch_data_samples[0].poly_pred_list
        poly_feat_list = batch_data_samples[0].poly_feat_list

        B, _, H, W = seg_probs.shape

        prim_reg_pred_list = []
        for poly_pred, poly_feat in zip(poly_pred_list, poly_feat_list):
            poly_pred = poly_pred.to(imgs.device)
            poly_feat = poly_feat.to(imgs.device)

            prim_reg_pred = self.forward(poly_pred, poly_feat, W)
            prim_reg_pred_list.append(prim_reg_pred)

        if len(prim_reg_pred_list) > 0:
            prim_reg_pred = torch.cat(prim_reg_pred_list, dim=0)
            batch_data_samples[0].prim_reg_pred = prim_reg_pred

        return batch_data_samples

    def predict_assemble_segments(self, imgs, batch_data_samples):
        poly2segs_idxes = batch_data_samples[0].poly2segs_idxes
        seg_sizes = batch_data_samples[0].seg_sizes
        sampled_segs = batch_data_samples[0].sampled_segs

        if len(sampled_segs) > 0:

            prim_reg_pred = batch_data_samples[0].prim_reg_pred
            poly2segs_idxes = self.to_numba_nested_list(poly2segs_idxes)
            seg_sizes = self.to_numba_nested_list(seg_sizes)
            rings, poly2ring_idxes = polygon_utils.assemble_segments_cpp(
                prim_reg_pred.cpu().numpy(), poly2segs_idxes, seg_sizes,
                length=self.poly_cfg.get('num_inter_points', 96),
                stride=self.poly_cfg.get('stride_size', 64),
            )

            rings = [torch.tensor(ring).to(prim_reg_pred.device) for ring in rings]
            batch_data_samples[0].rings = rings
            batch_data_samples[0].poly2ring_idxes = poly2ring_idxes

        return batch_data_samples

    def predict_dp(self, imgs, batch_data_samples, save_instances=True):

        rings = batch_data_samples[0].get('rings', None)
        scores = batch_data_samples[0].scores
        seg_probs = batch_data_samples[0].seg_probs
        B, _, H, W = seg_probs.shape

        if rings is not None:

            poly2ring_idxes = batch_data_samples[0].poly2ring_idxes
            return_format = self.poly_cfg.get('poly_return_format', 'json')

            simp_rings = polygon_utils.simplify_rings_dp(
                rings, lam=self.poly_cfg.get('lam', 4), device=rings[0].device,
                drop_last=False, max_step_size=self.poly_cfg.get('max_step_size', 50)
            )

            simp_rings = [x[:-1] for x in simp_rings]

            simp_polygons = polygon_utils.assemble_rings(
                simp_rings, poly2ring_idxes, format=return_format
            )

            # batch_data_samples[0].simp_polygons = simp_polygons
        else:
            simp_polygons = []

        if save_instances:
            seg_instances = InstanceData(segmentations=simp_polygons, scores=scores)
            seg_instances.polygon_masks = PolygonMasks.from_json(simp_polygons, H, W)
            seg_instances.bboxes = torch.tensor(seg_instances.polygon_masks.get_bounds())
            seg_instances.labels = torch.zeros(len(seg_instances), dtype=torch.long)

            batch_data_samples[0].pred_instances = seg_instances

        return batch_data_samples

    def predict(self, imgs, batch_data_samples, device='cpu'):

        t0 = time.time()

        batch_data_samples = self.predict_sample_segments(imgs, batch_data_samples)

        t1 = time.time()

        batch_data_samples = self.predict_gcp(imgs, batch_data_samples)

        t2 = time.time()

        batch_data_samples = self.predict_assemble_segments(imgs, batch_data_samples)

        t3 = time.time()

        batch_data_samples = self.predict_dp(imgs, batch_data_samples)

        t4 = time.time()

        print(f'{t1-t0} {t2-t1} {t3-t2} {t4-t3}')

        return batch_data_samples


