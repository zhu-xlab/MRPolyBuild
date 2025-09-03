# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
import torch.nn.functional as F
from mmengine.config import ConfigDict

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .maskformer import MaskFormer
from mmdet.structures import SampleList
from mmdet.utils import tanmlh_utils
from mmdet.structures.det_data_sample import DetDataSample
from mmengine.structures import InstanceData, PixelData
from typing import List, Tuple, Union
import pdb
import numpy as np
from tqdm import tqdm


@MODELS.register_module()
class STMask2FormerV2(MaskFormer):
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
        weights = torch.tensor(tanmlh_utils.get_patch_weight(h_crop))

        num_classes = 2
        preds = torch.zeros(batch_size, num_classes, out_h, out_w)
        count_mat = torch.zeros(batch_size, num_classes, out_h, out_w)


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
                mask_cls_results, mask_pred_results, seg_pred_results = self.panoptic_head.predict(
                    feats, pseudo_data_samples)

                results_list = self.panoptic_fusion_head.predict(
                    mask_cls_results,
                    mask_pred_results,
                    pseudo_data_samples,
                    rescale=rescale)

                ins_results_list = [x['ins_results'] for x in results_list]
                merged_instance = tanmlh_utils.mosaic_instance_data(
                    ins_results_list, (cur_crop_boxes[:,:2] * scale).to(torch.int), mask_shape,
                    mask_up_scale=scale
                )


                for offset, seg_pred in zip(cur_crop_boxes[:,:2], seg_pred_results):
                    start_x, start_y = offset
                    end_x = start_x + w_crop
                    end_y = start_y + h_crop

                    temp = seg_pred[None,:].cpu() * weights.view(1,1, h_crop, w_crop)
                    preds[i:i+1, :, start_y:end_y, start_x:end_x] += temp
                    count_mat[i:i+1, :, start_y:end_y, start_x:end_x] += 1

                # merged_instance.masks = F.interpolate(
                #     merged_instance.masks.unsqueeze(1).float(),
                #     size=(out_h, out_w), mode='bilinear'
                # )[:,0].bool()
                # merged_instance.bboxes /= scale

                new_results = dict(ins_results=merged_instance)
                new_results_list.append(new_results)

            assert (count_mat > 0).all()
            ins_results_list = [x['ins_results'] for x in new_results_list]
            concat_instance = ins_results_list[0].cat(ins_results_list)
            # concat_instance = tanmlh_utils.concat_instance_data(ins_results_list)
            results = dict(ins_results=concat_instance)
            results = self.add_pred_to_datasample(batch_data_samples, [results])
            pixel_data = PixelData(sem_seg=(F.softmax(preds / count_mat, dim=1)[:,1] > 0.5).long())
            results[0].pred_sem_seg = pixel_data

            return results

        return batch_data_samples
