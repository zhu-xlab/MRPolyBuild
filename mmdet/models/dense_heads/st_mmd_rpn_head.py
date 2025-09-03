# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import (cat_boxes, empty_box_as, get_box_tensor,
                                   get_box_wh, scale_boxes)
from mmdet.structures.bbox.mmrotate_transforms import obb2xyxy
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, MultiConfig, OptInstanceList
from .anchor_head import AnchorHead
from ..utils import (filter_scores_and_topk, select_single_mlvl, unpack_gt_instances)
from ..utils import images_to_levels, multi_apply, unmap
from ..layers import SinePositionalEncoding

@MODELS.register_module()
class STMMDRPNHead(AnchorHead):
    """Implementation of RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the background
            category. Defaults to 1.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or \
            list[dict]): Initialization config dict.
        num_convs (int): Number of convolution layers in the head.
            Defaults to 1.
    """  # noqa: W605

    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 init_cfg: MultiConfig = dict(
                     type='Normal', layer='Conv2d', std=0.01),
                 num_convs: int = 1,
                 st_cfg: dict = {},
                 domain_net: Optional[ConfigDict] = None,
                 loss_dice: Optional[ConfigDict] = None,
                 **kwargs) -> None:
        self.num_convs = num_convs
        assert num_classes == 1
        self.st_cfg = st_cfg

        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)

        self.loss_dice = MODELS.build(loss_dice)
        self.local_iter = torch.nn.Parameter(torch.zeros(1))
        self.local_iter.requires_grad = False

        self.domain_net = None
        if domain_net is not None:
            self.domain_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Step 1: Global Average Pooling
                nn.Flatten(),  # Step 2: Flatten to shape (B, C)
                nn.Linear(domain_net['in_channels'], domain_net['out_channels']),  # Step 3: Linear layer to get shape (B, C2)
            )

            self.domain_net_tch = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Step 1: Global Average Pooling
                nn.Flatten(),  # Step 2: Flatten to shape (B, C)
                nn.Linear(domain_net['in_channels'], domain_net['out_channels']),  # Step 3: Linear layer to get shape (B, C2)
            )
            if st_cfg.get('domain_net_embedding', False):
                self.domain_net_embedding = torch.nn.Parameter(torch.zeros(domain_net['out_channels']))

            if st_cfg.get('domain_pos_cfg', None) is not None:
                pos_embed = SinePositionalEncoding(128)
                self.domain_pos_embedding = torch.nn.Parameter(pos_embed(torch.zeros(1, 1024, 1024))[0])

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False))
            self.rpn_conv_stu = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv_stu = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)

        reg_dim = self.bbox_coder.encode_size

        self.rpn_cls_stu = nn.Conv2d(self.feat_channels, self.num_base_priors * self.cls_out_channels, 1)
        self.rpn_reg_stu = nn.Conv2d(self.feat_channels, self.num_base_priors * reg_dim, 1)

        self.rpn_conv_tch = copy.deepcopy(self.rpn_conv_stu)
        self.rpn_cls_tch = nn.Conv2d(self.feat_channels, self.num_base_priors * self.cls_out_channels, 1)
        self.rpn_reg_tch = nn.Conv2d(self.feat_channels, self.num_base_priors * reg_dim, 1)
        self.dropout = torch.nn.Dropout(p=self.st_cfg.get('drop_rate', 0.3))

        if self.st_cfg.get('do_memory_bank', False):
            memory_size = self.st_cfg.get('memory_size', 65536)
            self.pos_memory = torch.nn.Parameter(torch.zeros(memory_size))
            self.neg_memory = torch.nn.Parameter(torch.zeros(memory_size))
            self.pos_memory_idx = torch.nn.Parameter(torch.zeros(1))
            self.neg_memory_idx = torch.nn.Parameter(torch.zeros(1))
            self.pos_memory.requires_grad = False
            self.neg_memory.requires_grad = False
            self.pos_memory_idx.requires_grad = False
            self.neg_memory_idx.requires_grad = False

    def _init_stu_tch_weights(self):
        param_list = ['conv', 'reg', 'cls']

        for param_name in param_list:

            mcp_stu = list(eval(f'self.rpn_{param_name}_stu').parameters())
            mcp_tch = list(eval(f'self.rpn_{param_name}_tch').parameters())

            for i in range(0, len(mcp_stu)):
                if not mcp_stu[i].data.shape:  # scalar tensor
                    # mcp_stu[i].data = mp[i].data.clone()
                    mcp_tch[i].data = mcp_stu[i].data.clone()
                    if self.st_cfg.get('init_cls', True) and param_name == 'cls':
                        mcp_stu[i].data.fill_(0)
                        mcp_tch[i].data.fill_(0)
                else:
                    # mcp_stu[i].data[:] = mp[i].data[:].clone()
                    mcp_tch[i].data[:] = mcp_stu[i].data[:].clone()

                    if self.st_cfg.get('init_cls', True) and param_name == 'cls':
                        torch.nn.init.normal_(mcp_stu[i].data, 0, 0.01)
                        torch.nn.init.normal_(mcp_tch[i].data, 0, 0.01)

    def _update_tch(self, iter):
        alpha = self.st_cfg.get('alpha', 0.999)
        alpha_teacher = min(1 - 1 / (iter + 1), alpha)
        if self.domain_net is not None:
            for ema_param, param in zip(self.domain_net.parameters(), self.domain_net_tch.parameters()):
                if not param.data.shape:  # scalar tensor
                    ema_param.data = \
                        alpha_teacher * ema_param.data + \
                        (1 - alpha_teacher) * param.data
                else:
                    ema_param.data[:] = \
                        alpha_teacher * ema_param[:].data[:] + \
                        (1 - alpha_teacher) * param[:].data[:]

        else:
            for param_name in ['conv', 'cls', 'reg']:
                for ema_param, param in zip(eval(f'self.rpn_{param_name}_tch').parameters(),
                                            eval(f'self.rpn_{param_name}_stu').parameters()):
                    if not param.data.shape:  # scalar tensor
                        ema_param.data = \
                            alpha_teacher * ema_param.data + \
                            (1 - alpha_teacher) * param.data
                    else:
                        ema_param.data[:] = \
                            alpha_teacher * ema_param[:].data[:] + \
                            (1 - alpha_teacher) * param[:].data[:]

    def forward_single(self, ori_x: Tensor):
        """Forward feature map of a single scale level."""
        x_stu = self.dropout(ori_x)
        x_stu = self.rpn_conv_stu(x_stu)
        x_stu = F.relu(x_stu, inplace=False)

        rpn_cls_score_stu = self.rpn_cls_stu(x_stu)
        rpn_bbox_pred_stu = self.rpn_reg_stu(x_stu)

        with torch.no_grad():
            x_tch = self.rpn_conv_tch(ori_x)
            x_tch = F.relu(x_tch, inplace=False)

            rpn_cls_score_tch = self.rpn_cls_tch(x_tch)
            rpn_bbox_pred_tch = self.rpn_reg_tch(x_tch)

        return rpn_cls_score_stu, rpn_bbox_pred_stu, rpn_cls_score_tch, rpn_bbox_pred_tch
        # return rpn_cls_score_stu, rpn_bbox_pred_stu

    def loss_by_feat(
            self,
            stu_cls_scores: List[Tensor],
            stu_bbox_preds: List[Tensor],
            tch_cls_scores: List[Tensor],
            tch_bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:

        featmap_sizes = [featmap.size()[-2:] for featmap in stu_cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        if int(self.local_iter.data.item()) == 0:
            self._init_stu_tch_weights()
        else:
            self._update_tch(int(self.local_iter.data.item()))

        self.local_iter.data += 1

        device = stu_cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_dice = multi_apply(
            self.loss_by_feat_single,
            stu_cls_scores,
            stu_bbox_preds,
            tch_cls_scores,
            tch_bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor,
        )

        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox, loss_rpn_dice=losses_dice)

    def loss_by_feat_single(
        self, stu_cls_score: Tensor, stu_bbox_pred: Tensor,
        tch_cls_score: Tensor, tch_bbox_pred: Tensor,
        anchors: Tensor, labels: Tensor,
        label_weights: Tensor, bbox_targets: Tensor,
        bbox_weights: Tensor, avg_factor: int
    ) -> tuple:

        B, num_anchors, H, W = stu_cls_score.shape
        if not self.st_cfg.get('do_memory_bank', False):
            ignore_thre = self.st_cfg.get('st_ignore_thr', 0.8)
        else:
            ignore_thre = 1.0
            warm_up_iter = self.st_cfg.get('warm_up_iter', 1000)

            if self.local_iter.data.item() > warm_up_iter:
                # pos_mean = self.pos_memory.mean()
                # neg_mean = self.neg_memory.mean()
                # if pos_mean > neg_mean:
                #     std_ratio = self.st_cfg.get('pos_memory_std_ratio', 0.0)
                #     ignore_thre = pos_mean + self.pos_memory.std() * std_ratio

                pos_mean = self.pos_memory[0]
                pos_std = self.pos_memory[1]
                neg_mean = self.neg_memory[0]
                neg_std = self.neg_memory[1]
                if pos_mean > neg_mean:
                    std_ratio = self.st_cfg.get('pos_memory_std_ratio', 0.0)
                    ignore_thre = pos_mean + pos_std * std_ratio

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(B, H, W, num_anchors, self.bbox_coder.encode_size)
            stu_bbox_pred = stu_bbox_pred.permute(0,2,3,1).view(B, H, W, num_anchors, self.bbox_coder.encode_size)
            tch_bbox_pred = tch_bbox_pred.permute(0,2,3,1).view(B, H, W, num_anchors, self.bbox_coder.encode_size)

            stu_bbox_pred = self.bbox_coder.decode(anchors, stu_bbox_pred)
            tch_bbox_pred = self.bbox_coder.decode(anchors, tch_bbox_pred)

            stu_bbox_pred = get_box_tensor(stu_bbox_pred)
            tch_bbox_pred = get_box_tensor(tch_bbox_pred)
        else:
            stu_bbox_pred = stu_bbox_pred.permute(0,2,3,1).view(B, H, W, num_anchors, self.bbox_coder.encode_size)
            tch_bbox_pred = tch_bbox_pred.permute(0,2,3,1).view(B, H, W, num_anchors, self.bbox_coder.encode_size)

        tch_cls_score = tch_cls_score.permute(0,2,3,1)
        bbox_weights = bbox_weights.view(B, H, W, num_anchors, self.bbox_coder.encode_size)
        neg_bbox_mask = (bbox_weights == 0).all(dim=-1)
        pseudo_bbox_mask = tch_cls_score.sigmoid() >= ignore_thre
        pseudo_label_mask = (pseudo_bbox_mask & neg_bbox_mask).permute(0,3,1,2).reshape(-1)

        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        label_weights[pseudo_label_mask] = 0
        stu_cls_score = stu_cls_score.permute(0,2,3,1).reshape(-1, self.cls_out_channels)

        loss_cls = self.loss_cls(
            stu_cls_score, labels, label_weights, avg_factor=avg_factor)

        loss_dice = self.loss_dice(stu_cls_score, labels.view(-1,1), label_weights, avg_factor=avg_factor)

        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)

        stu_bbox_pred = stu_bbox_pred.reshape(-1, self.bbox_coder.encode_size)

        loss_bbox = self.loss_bbox(
            stu_bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)

        # print(bbox_weights.any(dim=1).sum().item())

        # pos_mean = stu_cls_score[labels == 0].sigmoid().mean()
        # pos_std = stu_cls_score[labels == 0].sigmoid().std()
        # neg_mean = stu_cls_score[labels == 1].sigmoid().mean()
        # neg_std = stu_cls_score[labels == 1].sigmoid().std()

        if self.st_cfg.get('do_memory_bank', False):
            if (labels == 0).sum() > 1:
                pos_scores = tch_cls_score.reshape(-1)[labels == 0].sigmoid()
                pos_mean_std = torch.stack([pos_scores.mean(), pos_scores.std()])
                self.update_memory(self.pos_memory, self.pos_memory_idx, pos_mean_std)

            if (labels == 1).sum() > 1:
                neg_scores = tch_cls_score.reshape(-1)[labels == 1].sigmoid()
                neg_mean_std = torch.stack([neg_scores.mean(), neg_scores.std()])
                self.update_memory(self.neg_memory, self.neg_memory_idx, neg_mean_std)

            # self.update_memory(self.pos_memory, self.pos_memory_idx, tch_cls_score.reshape(-1)[labels == 0].sigmoid())
            # self.update_memory(self.neg_memory, self.neg_memory_idx, tch_cls_score.reshape(-1)[labels == 1].sigmoid())
            # print(f'pos: {self.pos_memory.mean()} neg: {self.neg_memory.mean()}')

        return loss_cls, loss_bbox, loss_dice

    def predict_by_feat(self,
                        stu_cls_scores: List[Tensor],
                        stu_bbox_preds: List[Tensor],
                        tch_cls_scores: List[Tensor],
                        tch_bbox_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:

        assert len(stu_cls_scores) == len(stu_bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(stu_cls_scores) == len(score_factors)

        num_levels = len(stu_cls_scores)

        featmap_sizes = [stu_cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=stu_cls_scores[0].dtype,
            device=stu_cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                stu_cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                stu_bbox_preds, img_id, detach=True)

            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)

        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Be compatible with
                BaseDenseHead. Not used in RPNHead.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (ConfigDict, optional): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        level_ids = []
        for level_idx, (cls_score, bbox_pred, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            reg_dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, reg_dim)
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0] since mmdet v2.0
                # BG cat_id: 1
                scores = cls_score.softmax(-1)[:, :-1]

            scores = torch.squeeze(scores)
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                bbox_pred = bbox_pred[topk_inds, :]
                priors = priors[topk_inds]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)

            # use level id to implement the separate level nms
            level_ids.append(
                scores.new_full((scores.size(0), ), level_idx, dtype=torch.long)
            )

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.level_ids = torch.cat(level_ids)

        return self._bbox_post_process(
            results=results, cfg=cfg, rescale=rescale, img_meta=img_meta)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        assert with_nms, '`with_nms` must be True in RPNHead'
        if results.bboxes.shape[-1] == 4:
            if rescale:
                assert img_meta.get('scale_factor') is not None
                scale_factor = [1 / s for s in img_meta['scale_factor']]
                results.bboxes = scale_boxes(results.bboxes, scale_factor)

            # filter small size bboxes
            if cfg.get('min_bbox_size', -1) >= 0:
                w, h = get_box_wh(results.bboxes)
                valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
                if not valid_mask.all():
                    results = results[valid_mask]

        elif results.bboxes.shape[-1] == 5:
            if rescale:
                assert img_meta.get('scale_factor') is not None
                scale_factor = [1 / s for s in img_meta['scale_factor']]
                results.bboxes = scale_boxes(results.bboxes, scale_factor)

            if cfg.get('min_bbox_size', -1) >= 0:
                w = results.bboxes[:, 2]
                h = results.bboxes[:, 3]
                valid_mask = (w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size)
                if not valid_mask.all():
                    results = results[valid_mask]

        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            if results.bboxes.shape[-1] == 5:
                bboxes = obb2xyxy(bboxes, cfg.get('rbb_version', 'oc'))

            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.level_ids, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
            # TODO: This would unreasonably show the 0th class label
            #  in visualization
            results.labels = results.scores.new_zeros(
                len(results), dtype=torch.long)
            del results.level_ids
        else:
            # To avoid some potential error
            results_ = InstanceData()
            results_.bboxes = empty_box_as(results.bboxes)
            results_.scores = results.scores.new_zeros(0)
            results_.labels = results.scores.new_zeros(0)
            results = results_

        if cfg.get('score_thr', -1) > 0:
            score_thr = cfg['score_thr']
            keep_masks = results.scores > score_thr
            results = results[keep_masks]

        return results

    def update_memory(self, memory, memory_idx, data, num_sample=512):
        if len(data) == 0:
            return

        alpha = self.st_cfg.get('alpha_memory', 0.0)

        if len(data) > num_sample:
            sample_idxes = np.random.permutation(len(data))[:num_sample]
            data = data[sample_idxes]

        written_idxes = (torch.arange(len(data)) + memory_idx.data.item()) % len(memory)
        memory[written_idxes.long()] = memory[written_idxes.long()] * alpha + (1 - alpha) * data

        memory_idx += len(data)
        memory_idx = memory_idx % len(data)

    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            proposal_cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs

        if self.st_cfg.get('up_feat_level_1', False):
            _, _, h, w = x[0].shape
            new_x0 = F.interpolate(x[0], (h * 2, w * 2))
            x = [new_x0, *x[1:]]

        if self.st_cfg.get('up_feat_levels', None) is not None:
            new_xs = []
            for level in self.st_cfg['up_feat_levels']:
                _, _, h, w = x[level].shape
                new_x = F.interpolate(x[level], (h * 2, w * 2))
                new_xs.append(new_x)

            x = new_xs

        outs = self(x, batch_img_metas)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg)

        return losses, predictions

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        if self.st_cfg.get('up_feat_level_1', False):
            _, _, h, w = x[0].shape
            new_x0 = F.interpolate(x[0], (h * 2, w * 2))
            x = [new_x0, *x[1:]]

        if self.st_cfg.get('up_feat_levels', None) is not None:
            new_xs = []
            for level in self.st_cfg['up_feat_levels']:
                _, _, h, w = x[level].shape
                new_x = F.interpolate(x[level], (h * 2, w * 2))
                new_xs.append(new_x)

            x = new_xs

        outs = self(x, batch_img_metas)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def forward(self, x: Tuple[Tensor], batch_img_metas=None) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        if self.domain_net is not None:
            if self.st_cfg.get('domain_pos_cfg', None) is not None:
                domain_pos_cfg = self.st_cfg['domain_pos_cfg']
                tile_names = [meta['img_path'].split('/')[-1].split('_')[0] for meta in batch_img_metas]
                tile_x = torch.tensor([float(name.split('-')[0]) for name in tile_names])
                tile_y = torch.tensor([float(name.split('-')[1]) for name in tile_names])
                tile_idxes = torch.stack([tile_x, tile_y], dim=1)
                tile_idxes = tile_idxes / domain_pos_cfg['tile_max_size'] * domain_pos_cfg['grid_size']
                tile_idxes = tile_idxes.long()
                pos_embeddings = []
                B = len(batch_img_metas)
                for i in range(B):
                    pos_embeddings.append(self.domain_pos_embedding[:, tile_idxes[i,0], tile_idxes[i,1]])

                pos_embeddings = torch.stack(pos_embeddings)
                x[-1] = x[-1] + pos_embeddings.view(B,-1,1,1)


            weights_stu = self.domain_net(x[-1])
            weights_tch = self.domain_net_tch(x[-1])

            return multi_apply(self.forward_single_dynamic, x, [weights_stu] * len(x), [weights_tch] * len(x))

            """
            cls_score = []
            bbox_pred = []
            x_cls = x_cls.split(num_roi_per_img)
            x_reg = x_reg.split(num_roi_per_img)
            for i in range(B):
                # x_cls = torch.stack(x_cls.split(num_roi_per_img))
                # x_reg = torch.stack(x_reg.split(num_roi_per_img))
                # x_cls = x_cls.view(B, num_roi_per_img, -1)
                # x_reg = x_reg.view(B, num_roi_per_img, -1)
                cur_cls_score = torch.einsum('nc,ca->na', x_cls[i], dynamic_fc_cls[i])
                cur_bbox_pred = torch.einsum('nc,ca->na', x_reg[i], dynamic_fc_reg[i])
                cls_score.append(cur_cls_score)
                bbox_pred.append(cur_bbox_pred)

            cls_score = torch.cat(cls_score)
            bbox_pred = torch.cat(bbox_pred)
            """

        return multi_apply(self.forward_single, x)

    def forward_single_dynamic(self, ori_x: Tensor, ori_domain_weights_stu, ori_domain_weights_tch):
        """Forward feature map of a single scale level."""
        B = ori_x.shape[0]
        # num_roi_per_img = len(x) // B
        if self.st_cfg.get('domain_net_embedding', False):
            domain_weights_stu = ori_domain_weights_stu + self.domain_net_embedding.unsqueeze(0)
            domain_weights_tch = ori_domain_weights_tch + self.domain_net_embedding.unsqueeze(0)
        else:
            domain_weights_stu = ori_domain_weights_stu
            domain_weights_tch = ori_domain_weights_tch


        weights_cls_stu = domain_weights_stu.view(B, -1, 3*(1+4))[:,:,:3]
        weights_reg_stu = domain_weights_stu.view(B, -1, 3*(1+4))[:,:,3:]

        x = self.rpn_conv_stu(ori_x)
        noisy_x = self.dropout(x)

        x = F.relu(x, inplace=False)
        noisy_x = F.relu(noisy_x, inplace=False)

        rpn_cls_score_stu = torch.einsum('bchw,bcn->bnhw', noisy_x, weights_cls_stu)
        # rpn_bbox_pred_stu = torch.einsum('bchw,bcn->bnhw', noisy_x, weights_reg_stu)
        rpn_bbox_pred_stu = self.rpn_reg_stu(noisy_x)

        with torch.no_grad():
            weights_cls_tch = domain_weights_tch.view(B, -1, 3*(1+4))[:,:,:3]
            weights_reg_tch = domain_weights_tch.view(B, -1, 3*(1+4))[:,:,3:]
            rpn_cls_score_tch = torch.einsum('bchw,bcn->bnhw', x, weights_cls_tch)
            # rpn_bbox_pred_tch = torch.einsum('bchw,bcn->bnhw', x, weights_reg_tch)
            rpn_bbox_pred_tch = self.rpn_reg_tch(x)

        return rpn_cls_score_stu, rpn_bbox_pred_stu, rpn_cls_score_tch, rpn_bbox_pred_tch
