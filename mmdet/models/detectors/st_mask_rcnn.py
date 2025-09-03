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
from mmdet.structures import SampleList
from mmdet.utils import tanmlh_utils
from mmdet.structures.det_data_sample import DetDataSample
from mmdet.structures.mask import PolygonMasks
from .two_stage import TwoStageDetector
from typing import List, Tuple, Union
import pdb
import numpy as np
from tqdm import tqdm
from mmdet.utils import tanmlh_polygon_utils as polygon_utils

@MODELS.register_module()
class STMaskRCNN(TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone: ConfigDict,
                 rpn_head: ConfigDict,
                 roi_head: ConfigDict,
                 train_cfg: ConfigDict,
                 test_cfg: ConfigDict,
                 seg_head: ConfigDict = None,
                 neck: OptConfigType = None,
                 frozen_parameters: ConfigDict = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:

        self.frozen_parameters = frozen_parameters
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.seg_head = None
        if seg_head is not None:
            self.seg_head = MODELS.build(seg_head)

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
        inf_cfg = self.test_cfg.inf_cfg

        batch_size, _, h_img, w_img = batch_inputs.size()
        h_stride, w_stride = inf_cfg.stride
        h_crop, w_crop = inf_cfg.crop_size
        h_crop_up, w_crop_up = inf_cfg.crop_up_size
        out_h, out_w = inf_cfg.out_size if inf_cfg.out_size is not None else (h_img, w_img)
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

        for j in range(len(splits) - 1):
        # for j in tqdm(range(len(splits) - 1)):
            t0 = time.time()
            cur_crop_boxes = torch.tensor(
                np.stack(selected_crop_boxes[splits[j]:splits[j+1]]), device=batch_inputs.device
            )
            # is_border_boxes = (cur_crop_boxes == 0).any(dim=1) | (cur_crop_boxes == out_h).any(dim=1)

            cur_img = torch.cat(selected_crop_imgs[splits[j]:splits[j+1]])
            cur_img = F.interpolate(cur_img, size=(h_crop_up, w_crop_up), mode='bilinear')

            # x = self.extract_feat(cur_img)
            # ori_x = self.backbone(batch_inputs)
            ori_x = self.backbone(cur_img)
            if self.with_neck:
                x = self.neck(ori_x)

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

            t1 = time.time()

            # If there are no pre-defined proposals, use RPN to get proposals
            rpn_results_list = self.rpn_head.predict(x, pseudo_data_samples, rescale=False)
            t2 = time.time()

            results_list = self.roi_head.predict(
                x, rpn_results_list, pseudo_data_samples, rescale=rescale)

            t3 = time.time()

            for results in results_list:
                if 'segmentations' in results:
                    if type(results.segmentations) == PolygonMasks:
                        results.segmentations = results.segmentations.to_coco()

            filter_border_width = inf_cfg.get('filter_border_width', 0)
            if filter_border_width > 0:
                filter_results_list = []
                for i, result in enumerate(results_list):
                    filter_results_list.append(
                        tanmlh_utils.filter_border_instance(
                            result, filter_border_width, h_crop_up
                        )
                    )

                results_list = filter_results_list

            scale = h_crop_up / h_crop
            mask_shape = (round((scale * out_h)), round((scale * out_w)))
            # mask_shape = (pad_size_h, pad_size_w)

            merged_instance = tanmlh_utils.mosaic_instance_data(
                results_list,
                (cur_crop_boxes[:,:2] * scale - pad_offset).to(torch.int),
            )
            # merged_instance = tanmlh_utils.instance_nms(merged_instance, self.test_cfg.get('nms_cfg', {}))
            # if len(merged_instance) > 0:
            #     pdb.set_trace()

            new_results_list.append(merged_instance)
            if eval_proposal:
                merged_proposals = tanmlh_utils.mosaic_instance_data(
                    rpn_results_list,
                    (cur_crop_boxes[:,:2] * scale - pad_offset).to(torch.int),
                )
                new_proposals_list.append(merged_proposals)

            t4 = time.time()

            if self.seg_head is not None:

                seg_feats = ori_x
                if self.test_cfg.get('seg_head', {}).get('up_feat_levels', None) is not None:
                    seg_feats = []
                    for level in self.test_cfg['seg_head']['up_feat_levels']:
                        _, _, h, w = ori[level].shape
                        new_x = F.interpolate(ori[level], (h * 2, w * 2))
                        seg_feats.append(new_x)

                pseudo_meta_infos = [data_sample.metainfo for data_sample in pseudo_data_samples]
                pred_sem_seg = self.seg_head.predict(seg_feats, pseudo_meta_infos, None)
                sem_seg_list = [
                    InstanceData(
                        sem_seg=cur_sem_seg[None], offsets=offset[None]
                    ) for cur_sem_seg, offset in zip(pred_sem_seg, cur_crop_boxes[:,:2] * scale - pad_offset)
                ]

                # merged_pred_sem_seg = tanmlh_utils.mosaic_instance_data(
                #     sem_seg_list,
                #     (cur_crop_boxes[:,:2] * scale - pad_offset).to(torch.int),
                # )
                merged_sem_seg_list.extend(sem_seg_list)

            t5 = time.time()

            # print(f'time elapsed: {t1-t0} {t2-t1} {t3-t2} {t4-t3} {t5-t4}')
            # if j == 10:
            #     break

        lens = [len(x) for x in new_results_list]
        if sum(lens) > 0:
            results = {}

            concat_instance = new_results_list[0].cat(new_results_list)
            if self.seg_head is not None and inf_cfg.get('sem_seg_type', None) != 'det':
                offsets = torch.cat([x.offsets for x in merged_sem_seg_list])
                merged_sem_seg = tanmlh_utils.mosaic_instance_data(
                    merged_sem_seg_list,
                    offsets.to(torch.int), mask_shape=(out_h, out_w)
                )
                mask = merged_sem_seg.sem_seg != 0
                pred_sem_seg = F.softmax(merged_sem_seg.sem_seg, dim=1)[0,1:]
                pred_sem_seg = pred_sem_seg * mask[:,1]

                sem_seg_thr = inf_cfg.get('sem_seg_thr', 0.3)
                pixel_data = PixelData(sem_seg=(pred_sem_seg > sem_seg_thr).long())

            else:
                if 'masks' in concat_instance:
                    filtered_idxes = concat_instance.scores > inf_cfg.get('sem_seg_thr', 0.1)
                    if filtered_idxes.sum() > 0:
                        pred_sem_seg = concat_instance[filtered_idxes].masks.max(dim=0)[0].long().unsqueeze(0)
                    else:
                        pred_sem_seg = torch.zeros(1, out_h, out_w).long()

                    pixel_data = PixelData(sem_seg=pred_sem_seg)

            results = self.add_pred_to_datasample(batch_data_samples, [concat_instance])
            results[0].pred_sem_seg = pixel_data
            results[0].pred_sem_seg_prob = pred_sem_seg

            if post_processor.do_merge_with_fg:

                pred_sem_seg = results[0].pred_sem_seg.sem_seg.cpu().numpy()[0]
                pred_sem_seg_prob = results[0].pred_sem_seg_prob

                structuring_element = np.ones((7, 7))
                pred_sem_seg = binary_opening(pred_sem_seg, structure=structuring_element).astype(np.uint8)
                # pixel_data = PixelData(sem_seg=torch.tensor(pred_sem_seg)[None])
                # results[0].pred_sem_seg = pixel_data

                labeled_mask, num_components = scipy.ndimage.label(pred_sem_seg)
                flat_labels = labeled_mask.ravel()
                flat_prob = pred_sem_seg_prob.ravel().cpu().numpy()
                sum_prob = np.bincount(flat_labels, weights=flat_prob)
                pixel_counts = np.bincount(flat_labels)
                sem_seg_scores = sum_prob[1:] / pixel_counts[1:]

                sem_seg_polys, colors = polygon_utils.polygonize_mask(
                    torch.tensor(labeled_mask).to(torch.int), mode='simple_mask', scale=1.
                )

                poly_results = self.roi_head.poly_head.predict(
                    sem_seg_polys, out_w,
                    device=x[0].device, return_format='json'
                )
                sem_seg_polys = poly_results['simp_polygons']
                sem_seg_scores = torch.tensor([sem_seg_scores[x-1] for x in colors])

                fg_instances = InstanceData(segmentations=sem_seg_polys, scores=sem_seg_scores)
                fg_instances.polygon_masks = PolygonMasks.from_json(sem_seg_polys, out_h, out_w)
                fg_instances.bboxes = torch.tensor(fg_instances.polygon_masks.get_bounds())
                fg_instances.labels = concat_instance.labels.new_zeros(len(fg_instances))

                results[0].fg_instances = fg_instances


            results[0] = post_processor.process(results[0])
            # merged_sem_seg = results[0].pred_instances.polygon_masks.merge().to_tensor(dtype=torch.uint8, device='cpu')
            # pixel_data = PixelData(sem_seg=merged_sem_seg)
            # results[0].pred_sem_seg = pixel_data

            # concat_instance = tanmlh_utils.instance_nms(concat_instance, self.test_cfg.get('nms_cfg', {}))

            # results = dict(ins_results=concat_instance)

            if eval_proposal:
                concat_proposal = new_proposals_list[0].cat(new_proposals_list)
                concat_proposal = tanmlh_utils.instance_nms(concat_proposal, self.test_cfg.get('nms_cfg', {}))
                results[0].proposals = concat_proposal

            if 'segmentations' in results[0].pred_instances:
                pass
                # for result in results:
                #     poly_jsons = result.pred_instances.segmentations
                #     poly_coco = [polygon_utils.poly_json2coco(x) for x in poly_jsons]
                #     result.pred_instances.segmentations = poly_coco

        else:

            results = batch_data_samples
            results[0].pred_instances = new_results_list[0]
            if eval_proposal:
                results[0].proposals = new_proposals_list[0]

            pred_sem_seg = torch.zeros(1, out_h, out_w).long()
            pixel_data = PixelData(sem_seg=pred_sem_seg)
            results[0].pred_sem_seg = pixel_data

        return results

    def fft(self, x, is_shift=False):
        spectrum = fft.fft2(x, dim=(-2, -1))
        phase = torch.angle(spectrum)
        magnitude = torch.abs(spectrum)
        if is_shift:
            magnitude = fft.ifftshift(magnitude)
        return phase, magnitude

    def ifft(self, magnitude, phase):
        reconstructed_spectrum = magnitude * torch.exp(1j * phase)
        reconstructed_x = fft.ifft2(reconstructed_spectrum, dim=(-2, -1)).real
        return reconstructed_x

    def magnitude_mixup(self, x):
        # extract pahse and manigtude from images by DCT
        phase, magnitude = self.fft(x)

        # enhance: magnitude mixup
        batch_size = x.size(0)
        lam = torch.rand(batch_size).to(x.device).detach()\
            .unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        index = torch.randperm(batch_size)
        mixed_magnitude = lam * magnitude + (1-lam) * magnitude[index]

        # reconstruct images
        reconstructed_x = self.ifft(mixed_magnitude, phase)
        return reconstructed_x

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
        import time
        t0 = time.time()
        losses = dict()

        if self.train_cfg.get('apply_mixup', False):
            if torch.rand(1).item() < self.train_cfg.get('mixup_prob', 1.0):
                batch_inputs = self.magnitude_mixup(batch_inputs)

        x = self.backbone(batch_inputs)

        seg_data_samples = []
        if self.seg_head is not None:
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

        if self.train_cfg.get('apply_seg_poly_loss', False):
            poly_cfg = self.roi_head.poly_head.poly_cfg
            step_size = poly_cfg.get('step_size', 2)
            num_sample_points = poly_cfg.get('num_inter_points', 64)
            num_max_sample = poly_cfg.get('num_max_sample', 100)

            sem_seg_probs = F.softmax(seg_logits, dim=1)[:,1]
            sem_seg_mask = sem_seg_probs > self.train_cfg.get('sem_seg_thr', 0.5)
            sem_seg_mask = sem_seg_mask * (torch.arange(len(sem_seg_mask), device=sem_seg_mask.device).view(-1,1,1) + 1)

            sem_seg_polys, _ = polygon_utils.polygonize_mask(
                torch.cat([*sem_seg_mask]).cpu().to(torch.int), mode='simple_mask', scale=1.
            )

            if len(sem_seg_polys) == 0:
                # create a dummy one in this case
                sem_seg_polys = [dict(type='Polygon', coordinates=[[[0,0], [1,0], [1,1], [0,1], [0,0]]])]

            sem_seg_poly_masks = PolygonMasks.from_json(sem_seg_polys, out_h, out_w)
            sem_seg_poly_masks = sem_seg_poly_masks.centerize()
            sem_seg_poly_masks = sem_seg_poly_masks.random_sample(num_max_sample)
            losses_poly_seg = self.roi_head.poly_head.loss(
                sem_seg_poly_masks.to_json(),
                sem_seg_poly_masks.to_json(), step_size * num_sample_points,
                device=x[0].device
            )
            for key, loss in losses_poly_seg.items():
                losses[key+'_seg'] = loss

        t1 = time.time()

        if self.with_neck:
            x = self.neck(x)

        t2 = time.time()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        t3 = time.time()

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        t4 = time.time()
        # print(f'{t1-t0} {t2-t1} {t3-t2} {t4-t3}')

        if hasattr(self.rpn_head, 'st_cfg'):
            # and self.rpn_head.st_cfg.get('do_memory_bank', False):
            # losses['rpn_pos_mean'] = self.rpn_head.pos_memory.mean()
            # losses['rpn_pos_std'] = self.rpn_head.pos_memory.std()
            # losses['rpn_neg_mean'] = self.rpn_head.neg_memory.mean()
            # losses['rpn_neg_std'] = self.rpn_head.neg_memory.std()

            losses['rpn_pos_mean'] = self.rpn_head.pos_memory[0]
            losses['rpn_pos_std'] = self.rpn_head.pos_memory[1]
            losses['rpn_neg_mean'] = self.rpn_head.neg_memory[0]
            losses['rpn_neg_std'] = self.rpn_head.neg_memory[1]

        if self.roi_head.with_bbox and hasattr(self.roi_head.bbox_head, 'st_cfg') and self.roi_head.bbox_head.st_cfg.get('do_memory_bank', False):
            losses['roi_pos_mean'] = self.roi_head.bbox_head.pos_memory.mean()
            losses['roi_pos_std'] = self.roi_head.bbox_head.pos_memory.std()
            losses['roi_neg_mean'] = self.roi_head.bbox_head.neg_memory.mean()
            losses['roi_neg_std'] = self.roi_head.bbox_head.neg_memory.std()

        return losses

    def seg_logits2poly_jsons(self, seg_logits):
        pass
