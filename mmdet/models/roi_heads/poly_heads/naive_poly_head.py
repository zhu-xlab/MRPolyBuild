# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, ModuleList
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.structures.mask import mask_target, poly_target, PolygonMasks
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
import mmdet.utils.tanmlh_polygon_utils as polygon_utils
from mmdet.utils import tanmlh_utils
from mmdet.models.layers import Mask2FormerTransformerDecoder, SinePositionalEncoding

import shapely

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
#  determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


@MODELS.register_module()
class NaivePolyHead(BaseModule):

    def __init__(self,
                 poly_cfg: OptMultiConfig = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.poly_cfg = poly_cfg

    def forward(self, x: Tensor) -> Tensor:

        return x

    def predict(self, poly_jsons, mask_size,  **kwargs):
        results = {}
        poly_shps = [shapely.geometry.shape(poly_json) for poly_json in poly_jsons]
        # simp_poly_shps = [poly.simplify(tolerance=1.5, preserve_topology=True) for poly in poly_shps]
        simp_poly_shps = poly_shps
        simp_poly_jsons = [shapely.geometry.mapping(poly) for poly in simp_poly_shps]

        results['simp_polygons'] = simp_poly_jsons
        return results

    def _predict_by_feat_single(self,
                                poly_preds: Tensor,
                                bboxes: Tensor,
                                labels: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: ConfigDict,
                                rescale: bool = False,
                                activate_map: bool = False) -> Tensor:
        """Get segmentation masks from mask_preds and bboxes.

        Args:
            mask_preds (Tensor): Predicted foreground masks, has shape
                (n, num_classes, h, w).
            bboxes (Tensor): Predicted bboxes, has shape (n, 4)
            labels (Tensor): Labels of bboxes, has shape (n, )
            img_meta (dict): image information.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            activate_map (book): Whether get results with augmentations test.
                If True, the `mask_preds` will not process with sigmoid.
                Defaults to False.

        Returns:
            Tensor: Encoded masks, has shape (n, img_w, img_h)

        Example:
            >>> from mmengine.config import Config
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> self = FCNMaskHead(num_classes=C, num_convs=0)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_preds = self.forward(inputs)
            >>> # Each input is associated with some bounding box
            >>> bboxes = torch.Tensor([[1, 1, 42, 42 ]] * N)
            >>> labels = torch.randint(0, C, size=(N,))
            >>> rcnn_test_cfg = Config({'mask_thr_binary': 0, })
            >>> ori_shape = (H * 4, W * 4)
            >>> scale_factor = (1, 1)
            >>> rescale = False
            >>> img_meta = {'scale_factor': scale_factor,
            ...             'ori_shape': ori_shape}
            >>> # Encoded masks are a list for each category.
            >>> encoded_masks = self._get_seg_masks_single(
            ...     mask_preds, bboxes, labels,
            ...     img_meta, rcnn_test_cfg, rescale)
            >>> assert encoded_masks.size()[0] == N
            >>> assert encoded_masks.size()[1:] == ori_shape
        """
        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        # img_h, img_w = img_meta['ori_shape'][:2]
        img_h, img_w = img_meta['batch_input_shape'][:2]
        device = bboxes.device

        if rescale:  # in-placed rescale the bboxes
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)

        N = len(poly_preds)

        poly_masks = PolygonMasks([[x.cpu().numpy().reshape(-1)] for x in poly_preds], img_h, img_w)
        poly_masks = poly_masks.paste_by_bboxes(bboxes.cpu().numpy(), self.mask_size, self.mask_size)

        return poly_masks

