# Copyright (c) OpenMMLab. All rights reserved.
import torch
import logging
from torch import Tensor
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.logging import print_log
from mmengine.structures import InstanceData, PixelData

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import SampleList
from mmdet.utils import tanmlh_utils
from mmdet.structures.det_data_sample import DetDataSample
from .st_mask_rcnn import STMaskRCNN
from typing import List, Tuple, Union
import pdb
import numpy as np
from tqdm import tqdm



@MODELS.register_module()
class STMaskRCNNPolygonizer(STMaskRCNN):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self, polygon_head=None, **kwargs) -> None:
        super().__init__(**kwargs)

