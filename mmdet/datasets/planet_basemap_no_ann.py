# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union
import numpy as np

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset

import pdb
from .base_semseg_dataset import BaseSegDataset


@DATASETS.register_module()
class PlanetBasemapNoAnnDataset(BaseSegDataset):
    """Dataset for iSAID instance segmentation.

    iSAID: A Large-scale Dataset for Instance Segmentation
    in Aerial Images.

    For more detail, please refer to "projects/iSAID/README.md"
    """

    METAINFO = dict(
        classes=('background', 'building'),
        palette=[(0, 0, 255), (255, 0, 0)])
