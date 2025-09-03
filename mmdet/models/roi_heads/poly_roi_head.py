# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple
import pdb
import shapely
import time

import torch
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi, rbbox2roi, obb2hbb, bbox_cxcywh_to_xyxy
from mmdet.utils import ConfigType, InstanceList
import mmdet.utils.tanmlh_polygon_utils as polygon_utils

from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .base_roi_head import BaseRoIHead


@MODELS.register_module()
class PolyRoIHead(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, poly_cfg={}, poly_head=None, **kwargs):
        self.poly_cfg = poly_cfg

        super().__init__(**kwargs)
        self.poly_head = None
        if poly_head is not None:
            self.poly_head = MODELS.build(poly_head)

    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
            self.bbox_sampler = TASK_UTILS.build(
                self.train_cfg.sampler, default_args=dict(context=self))

    def init_bbox_head(self, bbox_roi_extractor: ConfigType,
                       bbox_head: ConfigType) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)

    def init_mask_head(self, mask_roi_extractor: ConfigType,
                       mask_head: ConfigType) -> None:
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict or ConfigDict): Config of mask roi
                extractor.
            mask_head (dict or ConfigDict): Config of mask in mask head.
        """
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = MODELS.build(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

        self.mask_head = MODELS.build(mask_head)

    # TODO: Need to refactor later
    def forward(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList = None) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
            the meta information of each image and corresponding
            annotations.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        results = ()
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            results = results + (bbox_results['cls_score'],
                                 bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            results = results + (mask_results['mask_preds'], )

        return results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)

        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []

        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            cur_gt_instances = batch_gt_instances[i]
            if rpn_results.priors.shape[1] == 5:
                rpn_results.priors = obb2hbb(rpn_results.priors)
                rpn_results.priors = bbox_cxcywh_to_xyxy(rpn_results.priors[:,:-1])

                cur_gt_instances = batch_gt_instances[i].clone()
                cur_gt_instances.bboxes = obb2hbb(batch_gt_instances[i].bboxes)[:,:-1]
                cur_gt_instances.bboxes = bbox_cxcywh_to_xyxy(cur_gt_instances.bboxes)

            # if ((batch_gt_instances[i].bboxes[:,2] - batch_gt_instances[i].bboxes[:,0]) <= 0).any():
            #     pdb.set_trace()
            # if ((batch_gt_instances[i].bboxes[:,3] - batch_gt_instances[i].bboxes[:,1]) <= 0).any():
            #     pdb.set_trace()

            assign_result = self.bbox_assigner.assign(
                rpn_results, cur_gt_instances,
                batch_gt_instances_ignore[i])

            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                cur_gt_instances,
                feats=[lvl_feat[i][None] for lvl_feat in x])

            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            # mask_results = self.mask_loss(x, sampling_results,
            #                               bbox_results['bbox_feats'],
            #                               batch_gt_instances)
            mask_results = self.mask_loss(x, sampling_results,
                                          None, batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        if self.poly_head is not None and self.train_cfg.get('train_poly_head', False) is True:
            poly_results = self.poly_loss(
                sampling_results,
                mask_results['pos_rois'][:,1:],
                mask_results,
                batch_data_samples
            )
            losses.update(poly_results)

        return losses

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        B = x[0].shape[0]

        num_roi_per_img = [(rois[:,0].long() == i).sum() for i in range(B)]

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = self.bbox_head(bbox_feats, x, num_roi_per_img)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        return bbox_results

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            tch_cls_score=bbox_results['tch_cls_score'],
            tch_bbox_pred=bbox_results['tch_bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg,
            x=x
        )

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats: Tensor,
                  batch_gt_instances: InstanceList) -> dict:
        """Perform forward propagation and loss calculation of the mask head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            bbox_feats (Tensor): Extract bbox RoI features.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
                - `mask_targets` (Tensor): Mask target of each positive\
                    proposals in the image.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """

        if not self.share_roi_extractor:
            # pos_rois = rbbox2roi([res.pos_priors for res in sampling_results])
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)

        mask_results.update(
            loss_mask=mask_loss_and_target['loss_mask'], pos_rois=pos_rois,
            mask_targets=mask_loss_and_target['mask_targets']
        )
        return mask_results

    def poly_loss(self, sampling_results, bbox_preds, mask_results, batch_data_samples):
        batch_gt_instances = [x.gt_instances for x in batch_data_samples]

        sample_iou_thr = self.poly_head.poly_cfg.get('sample_iou_thr', 0.5)
        num_max_sample = self.poly_head.poly_cfg.get('num_max_sample', 100)
        mask_preds = mask_results['mask_preds']
        mask_feats = mask_results['mask_feats']
        pos_rois = mask_results['pos_rois']
        mask_targets = mask_results['mask_targets'].bool()
        bin_mask_preds = mask_preds[:,0] > 0.5
        intersection = mask_targets & bin_mask_preds
        union = mask_targets | bin_mask_preds
        mask_ious = intersection.sum(dim=[1,2]) / union.sum(dim=[1,2])
        valid_idxes = (mask_ious > sample_iou_thr).nonzero()[:,0]
        mask_shape = mask_preds.shape[2:]

        if len(valid_idxes) > num_max_sample:
            sample_idxes = torch.randperm(len(valid_idxes))[:num_max_sample]
            valid_idxes = valid_idxes[sample_idxes]

        if len(valid_idxes) > 0:

            img_shape = batch_data_samples[0].img_shape
            batch_idxes = torch.arange(len(valid_idxes), device=mask_preds.device)

            bboxes = bbox_preds
            gt_poly_jsons = []
            for i, sampling_result in enumerate(sampling_results):
                cur_gt_jsons = batch_gt_instances[i].masks.to_json()
                gt_poly_jsons.extend([cur_gt_jsons[x.item()] for x in sampling_result.pos_assigned_gt_inds])

            scale_w = mask_shape[1] / (bboxes[:,2] - bboxes[:,0])
            scale_h = mask_shape[0] / (bboxes[:,3] - bboxes[:,1])
            offsets = (-bboxes[:,:2]).detach().cpu().numpy()
            scaled_gt_jsons = [polygon_utils.transform_polygon(
                json, offsets=offsets[i].tolist(), scales=(scale_w[i].item(), scale_h[i].item())
            ) for i, json in enumerate(gt_poly_jsons)]

            sampled_mask_feats = mask_feats[valid_idxes]
            sampled_mask_preds = mask_preds[valid_idxes]
            sampled_mask_targets = mask_targets[valid_idxes]

            pred_jsons = polygon_utils.polygonize_mask((sampled_mask_preds > 0.5)[:,0].to(torch.int), scale=1, mode='concat_mask')

            sampled_gt_jsons = [scaled_gt_jsons[x] for x in valid_idxes]

            sampled_gt_jsons = polygon_utils.polygonize_mask(sampled_mask_targets.to(torch.int), scale=1, mode='concat_mask')
            sampled_gt_shps = [shapely.geometry.shape(x) for x in sampled_gt_jsons]
            simple_gt_jsons = [shapely.geometry.mapping(x.simplify(tolerance=1.0)) for x in sampled_gt_shps]

            losses_poly = self.poly_head.loss(
                pred_jsons,
                simple_gt_jsons,
                mask_shape[0],
                mask_feat=sampled_mask_feats,
                batch_idxes=batch_idxes,
                device=mask_preds.device
            )

        else:
            # losses_poly = dict(
            #     loss_poly_reg = mask_preds[:0].sum(),
            #     loss_dp = mask_preds[:0].sum(),
            #     loss_poly_right_ang = mask_preds[:0].sum()
            # )

            dummy_json = dict(type='Polygon', coordinates=[[[0,0], [1,0], [1,1], [0,1], [0,0]]])
            losses_poly = self.poly_head.loss([dummy_json], [dummy_json], mask_shape[0], device=mask_preds.device)

        return losses_poly


    def _mask_forward(self,
                      x: Tuple[Tensor],
                      rois: Tensor = None,
                      pos_inds: Optional[Tensor] = None,
                      bbox_feats: Optional[Tensor] = None) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            pos_inds (Tensor, optional): Indices of positive samples.
                Defaults to None.
            bbox_feats (Tensor): Extract bbox RoI features. Defaults to None.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
        """
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds = self.mask_head(mask_feats)
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats)

        return mask_results

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale,
            x=x
        )

        return result_list

    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]

        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary
            )
            if self.test_cfg.get('block_mask_predict', False):
                for results in results_list:
                    del results['masks']

            feat_size = self.mask_roi_extractor.roi_layers[0].output_size
            feat_channels = self.mask_roi_extractor.out_channels
            mask_size = (self.train_cfg['mask_size'], self.train_cfg['mask_size'])

            empty_mask_feats = torch.zeros(0, feat_channels, *feat_size, device=x[0].device)
            empty_mask_preds = torch.zeros(0, 1, *mask_size, device=x[0].device)

            for results in results_list:
                results['mask_feats'] = empty_mask_feats
                results['mask_preds'] = empty_mask_preds

            return results_list

        mask_results = self._mask_forward(x, mask_rois)
        mask_preds = mask_results['mask_preds']
        mask_feats = mask_results['mask_feats']

        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
        mask_feats = mask_feats.split(num_mask_rois_per_img, 0)

        if not self.test_cfg.get('block_mask_predict', False):
            # TODO: Handle the case where rescale is false
            results_list = self.mask_head.predict_by_feat(
                mask_preds=mask_preds,
                results_list=results_list,
                batch_img_metas=batch_img_metas,
                rcnn_test_cfg=self.test_cfg,
                rescale=rescale)

        for results, mask_feat, mask_pred in zip(results_list, mask_feats, mask_preds):
            results['mask_feats'] = mask_feat
            results['mask_preds'] = mask_pred

        return results_list

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (N, C, H, W).
            rpn_results_list (list[:obj:`InstanceData`]): list of region
                proposals.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results to
                the original image. Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # assert self.with_bbox, 'Bbox head must be implemented.'

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        # TODO: nms_op in mmcv need be enhanced, the bbox result may get
        #  difference when not rescale in bbox_head

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.

        if self.with_bbox:
            bbox_rescale = rescale if not self.with_mask else False
            results_list = self.predict_bbox(
                x,
                batch_img_metas,
                rpn_results_list,
                rcnn_test_cfg=self.test_cfg,
                rescale=bbox_rescale)

        else:
            results_list = rpn_results_list

        if self.with_mask and not self.test_cfg.get('disable_mask_head', False):
            results_list = self.predict_mask(
                x, batch_img_metas, results_list, rescale=rescale)

        if self.poly_head is not None:
            mask_preds = [x['mask_preds'][:,0].sigmoid() >= 0.5 for x in results_list]
            mask_feats = [x['mask_feats'] for x in results_list]
            mask_shape = mask_preds[0].shape[1:]

            num_instance_per_img = [len(x) for x in mask_preds]
            concat_mask_preds = torch.cat(mask_preds)
            mask_feats = torch.cat([x['mask_feats'] for x in results_list])

            if len(concat_mask_preds) == 0:
                for results in results_list:
                    results.segmentations = []

                return results_list

            poly_jsons = polygon_utils.polygonize_mask(concat_mask_preds, scale=1., mode='concat_mask', return_multi_polygon=True)
            # poly_idxes = torch.tensor(poly_idxes, device=x[0].device)

            unfold_poly_jsons, geom2poly_idxes, poly2geom_idxes = polygon_utils.unfold_poly_jsons(poly_jsons)

            poly_results = self.poly_head.predict(
                unfold_poly_jsons, mask_shape[1], mask_features=mask_feats,
                batch_idxes=torch.tensor(poly2geom_idxes, device=x[0].device),
                device=x[0].device, return_format='json'
            )
            unfold_poly_jsons = poly_results['simp_polygons']

            poly_jsons = polygon_utils.fold_poly_jsons(unfold_poly_jsons, geom2poly_idxes)

            poly_jsons_list = []
            cur_num = 0
            for i, num in enumerate(num_instance_per_img):
                if num > 0:
                    # _, H, W = results_list[i].masks.shape
                    H, W = batch_img_metas[i]['img_shape']

                    _, _, h, w = results_list[i].mask_preds.shape
                    cur_poly_jsons = poly_jsons[cur_num:cur_num + num]
                    cur_poly_jsons = polygon_utils.paste_poly_json(
                        cur_poly_jsons, results_list[i].bboxes.cpu(), h, w, H, W
                    )
                    poly_jsons_list.append(cur_poly_jsons)
                    cur_num += num
                else:
                    poly_jsons_list.append([])

            for poly_jsons, results in zip(poly_jsons_list, results_list):
                results.segmentations = poly_jsons

        return results_list
