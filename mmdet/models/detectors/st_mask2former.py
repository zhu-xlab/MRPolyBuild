# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData, PixelData

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from .maskformer import MaskFormer
from mmdet.structures import SampleList
from mmdet.utils import tanmlh_utils
from mmdet.structures.det_data_sample import DetDataSample
from typing import List, Tuple, Union
import pdb
import numpy as np
from tqdm import tqdm


@MODELS.register_module()
class STMask2Former(MaskFormer):
    r"""Implementation of `Masked-attention Mask
    Transformer for Universal Image Segmentation
    <https://arxiv.org/pdf/2112.01527>`_."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 panoptic_head: OptConfigType = None,
                 panoptic_fusion_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

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
        out_size_scale = inf_cfg.get('out_size_scale', None)
        scale = h_crop_up / h_crop
        mask_shape = (round((scale * out_h)), round((scale * out_w)))

        if out_size_scale is not None:
            out_h = round(out_h * out_size_scale)
            out_w = round(out_w * out_size_scale)

        crop_boxes = tanmlh_utils.get_crop_boxes(h_img, w_img, (h_crop, w_crop), (h_stride, w_stride))

        assert batch_size == 1
        split_batch_size = 4

        """
        preds[0, 1, 100:110, 100:110] = 1
        preds[0, 1, 200:210, 100:110] = 1
        preds[0, 1, 100:110, 200:210] = 1
        preds[0, 1, 200:210, 200:210] = 1
        return preds
        """

        for i in range(batch_size):
            selected_crop_imgs = []
            selected_crop_boxes = []
            for crop_idx, crop_box in enumerate(crop_boxes):
                start_x, start_y, end_x, end_y = crop_box
                crop_img = batch_inputs[i:i+1, :, start_y:end_y, start_x:end_x]
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
            for j in range(len(splits) - 1):
            # for j in range(len(splits) - 1):
                cur_crop_boxes = torch.tensor(
                    np.stack(selected_crop_boxes[splits[j]:splits[j+1]]), device=batch_inputs.device
                )
                cur_img = torch.cat(selected_crop_imgs[splits[j]:splits[j+1]])
                cur_img = F.interpolate(cur_img, size=(h_crop_up, w_crop_up), mode='bilinear')

                pseudo_data_samples = [DetDataSample(
                    metainfo=dict(
                        # scale_factor=(h_crop_up / h_crop, w_crop_up / w_crop),
                        scale_factor=(1,1),
                        img_shape=(h_crop_up, w_crop_up),
                        ori_shape=(h_crop, w_crop),
                        batch_input_shape=(h_crop_up, w_crop_up)
                    )
                ) for i in cur_img]

                feats = self.extract_feat(cur_img)
                if self.panoptic_head.grid_cfg is not None:

                    grid_h, grid_w = self.panoptic_head.grid_cfg['H'], self.panoptic_head.grid_cfg['W']
                    crop_boxes = tanmlh_utils.get_crop_boxes(
                        grid_h, grid_w,
                        self.panoptic_head.grid_cfg['crop_size'],
                        self.panoptic_head.grid_cfg['stride'],
                    )
                    crop_boxes = np.concatenate([crop_boxes, np.array([[0,0,grid_h,grid_w]])])

                    featmap_strides = self.panoptic_head.grid_cfg['featmap_strides']
                    crop_feats = [tanmlh_utils.crop_featmap(feats[i], crop_boxes, featmap_strides[i]) for i in range(len(feats))]
                    mask_cls_results, mask_pred_results = self.panoptic_head.predict(crop_feats, pseudo_data_samples)

                else:
                    mask_cls_results, mask_pred_results = self.panoptic_head.predict(feats, pseudo_data_samples)

                # results_list = self.panoptic_fusion_head.predict(
                #     mask_cls_results,
                #     mask_pred_results,
                #     pseudo_data_samples,
                #     rescale=rescale)

                mask_scores = F.softmax(mask_cls_results, dim=1)
                mask_idx = mask_scores[:,:,0].argmax(dim=1)
                # sem_mask_pred = torch.gather(mask_pred_results, 1, mask_idx.view(len(mask_pred_results), 1, *mask_pred_results.shape[2:]))
                sem_mask_preds = []
                for k in range(len(mask_pred_results)):
                    sem_mask_preds.append(mask_pred_results[k][mask_idx[k]])

                pred_sem_seg = (torch.stack(sem_mask_preds).sigmoid())
                if self.panoptic_head.grid_cfg is not None:
                    # pred_sem_seg = tanmlh_utils.paste_masks(pred_sem_seg[:-1], crop_boxes[:-1], grid_h, grid_w)
                    pred_sem_seg = tanmlh_utils.paste_masks(pred_sem_seg[-1:], crop_boxes[-1:], grid_h, grid_w)
                    pred_sem_seg = (pred_sem_seg > 0.3).any(dim=0).long().unsqueeze(0)
                else:
                    pred_sem_seg = (pred_sem_seg > 0.3).long()

                pixel_data = PixelData(sem_seg=pred_sem_seg)
                batch_data_samples[0].pred_sem_seg = pixel_data
                batch_data_samples[0].pred_instances = InstanceData(
                    bboxes=torch.zeros(1,4),
                    masks=pred_sem_seg.bool(),
                    scores=torch.zeros(1),
                    labels=torch.zeros(1).long()
                )

                return batch_data_samples


                ins_results_list = [x['ins_results'] for x in results_list]
                lens = [len(x) for x in ins_results_list]
                if sum(lens) > 0:
                    merged_instance = tanmlh_utils.mosaic_instance_data(
                        ins_results_list, (cur_crop_boxes[:,:2] * scale).to(torch.int), mask_shape,
                        mask_up_scale=scale
                    )
                else:
                    merged_instance = ins_results_list[0]
                    # merged_instance.masks = torch.zeros(1, *mask_shape, dtype=torch.bool)

                # merged_instance.masks = F.interpolate(
                #     merged_instance.masks.unsqueeze(1).float(),
                #     size=(out_h, out_w), mode='bilinear'
                # )[:,0].bool()
                # merged_instance.bboxes /= scale

                new_results = dict(ins_results=merged_instance)
                new_results_list.append(new_results)

            ins_results_list = [x['ins_results'] for x in new_results_list]
            concat_instance = ins_results_list[0].cat(ins_results_list)
            # concat_instance = tanmlh_utils.concat_instance_data(ins_results_list)
            results = dict(ins_results=concat_instance)
            results = self.add_pred_to_datasample(batch_data_samples, [results])

            return results

        return batch_data_samples
