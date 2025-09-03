# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
import torch.nn.functional as F
from mmengine.config import ConfigDict
from tqdm import tqdm
from typing import List, Tuple, Union

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import SampleList


from mmdet.utils import tanmlh_utils
from mmdet.structures.det_data_sample import DetDataSample
from .single_stage_instance_seg import SingleStageInstanceSegmentor
import pdb
import numpy as np


@MODELS.register_module()
class STSOLOv2(SingleStageInstanceSegmentor):
    """`SOLOv2: Dynamic and Fast Instance Segmentation
    <https://arxiv.org/abs/2003.10152>`_

    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 mask_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            mask_head=mask_head,
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
                is_border_boxes = (cur_crop_boxes == 0).any(dim=1) | (cur_crop_boxes == out_h).any(dim=1)

                cur_img = torch.cat(selected_crop_imgs[splits[j]:splits[j+1]])
                cur_img = F.interpolate(cur_img, size=(h_crop_up, w_crop_up), mode='bilinear')

                """
                x = self.extract_feat(cur_img)

                pseudo_data_samples = [DetDataSample(
                    metainfo=dict(
                        # scale_factor=(h_crop_up / h_crop, w_crop_up / w_crop),
                        scale_factor=(1,1),
                        img_shape=(h_crop_up, w_crop_up),
                        ori_shape=(h_crop, w_crop),
                        batch_input_shape=(h_crop_up, w_crop_up)
                    )
                ) for i in cur_img]

                # If there are no pre-defined proposals, use RPN to get proposals
                rpn_results_list = self.rpn_head.predict(x, pseudo_data_samples, rescale=False)
                results_list = self.roi_head.predict(
                    x, rpn_results_list, pseudo_data_samples, rescale=rescale)
                """

                pseudo_data_samples = [DetDataSample(
                    metainfo=dict(
                        # scale_factor=(h_crop_up / h_crop, w_crop_up / w_crop),
                        scale_factor=(1,1),
                        img_shape=(h_crop_up, w_crop_up),
                        ori_shape=(h_crop, w_crop),
                        batch_input_shape=(h_crop_up, w_crop_up)
                    )
                ) for i in cur_img]

                x = self.extract_feat(cur_img)
                # bbox_rescale = rescale if not self.with_mask else False
                # results_list = self.bbox_head.predict(
                #     x, pseudo_data_samples, rescale=bbox_rescale)

                results_list = None
                results_list = self.mask_head.predict(
                    x, pseudo_data_samples, rescale=rescale, results_list=results_list)

                filter_border_width = inf_cfg.get('filter_border_width', 0)
                if filter_border_width > 0:
                    filter_results_list = []
                    for i, result in enumerate(results_list):
                        if is_border_boxes[i]:
                            filter_results_list.append(result)
                        else:
                            filter_results_list.append(
                                tanmlh_utils.filter_border_instance(
                                    result, filter_border_width, h_crop_up
                                )
                            )
                    results_list = filter_results_list

                scale = h_crop_up / h_crop
                mask_shape = (round((scale * out_h)), round((scale * out_w)))
                merged_instance = tanmlh_utils.mosaic_instance_data(
                    results_list, (cur_crop_boxes[:,:2] * scale).to(torch.int), mask_shape
                )
                # merged_instance.masks = F.interpolate(merged_instance.masks, size=(out_h, out_w), mode='bilinear')
                # merged_instance.masks = F.interpolate(
                #     merged_instance.masks.unsqueeze(1).float(),
                #     size=(out_h, out_w), mode='bilinear'
                # )[:,0].bool()
                # merged_instance.bboxes /= scale

                if len(merged_instance) > 0:
                    nms_instance = tanmlh_utils.instance_nms(merged_instance, self.test_cfg.get('nms_cfg', {}))
                    new_results = dict(ins_results=nms_instance)
                    new_results_list.append(new_results)

            ins_results_list = [x['ins_results'] for x in new_results_list]
            concat_instance = ins_results_list[0].cat(ins_results_list)
            # results = dict(ins_results=concat_instance)
            # batch_data_samples = self.add_pred_to_datasample(batch_data_samples, [merged_instance])
            results = self.add_pred_to_datasample(batch_data_samples, [concat_instance])

            return results

        return batch_data_samples
