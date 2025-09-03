# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType

@MODELS.register_module()
class SimplePixelDecoder(BaseModule):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        fpn_cfg = dict(
            type='FPN',
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=len(in_channels)
        )
        self.fpn = MODELS.build(fpn_cfg)
        self.mask_feature = Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        x = self.fpn(inputs)
        mask_features = self.mask_feature(x[-1])

        return mask_features, x
