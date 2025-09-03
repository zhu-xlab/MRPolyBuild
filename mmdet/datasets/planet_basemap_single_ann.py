# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import json
import os.path as osp
from typing import List, Union
import numpy as np
import shapely
import mmdet.utils.tanmlh_polygon_utils as polygon_utils

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset

import pdb
from .coco import CocoDataset


@DATASETS.register_module()
class PlanetBasemapSingleAnnDataset(CocoDataset):
    """Dataset for iSAID instance segmentation.

    iSAID: A Large-scale Dataset for Instance Segmentation
    in Aerial Images.

    For more detail, please refer to "projects/iSAID/README.md"
    """

    METAINFO = dict(
        # classes=('building'),
        # palette=[(0, 0, 255)])
        classes=('background', 'building'),
        palette=[(0, 0, 255), (255, 0, 0)])

    def __init__(self, ann_dir, bbox_buffer=-1, min_bbox_w=-1, drop_rate=0.0, ann_cfg={}, **kwargs):
        self.min_bbox_w = min_bbox_w
        self.drop_rate = drop_rate
        self.ann_dir=ann_dir
        self.ann_cfg = ann_cfg
        self.bbox_buffer = bbox_buffer

        super().__init__(**kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None

        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if 'continent' in img_info:
            data_info['continent'] = img_info['continent']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        if self.drop_rate > 0:
            num_chosen = int(len(ann_info) * (1 - self.drop_rate))
            chosen_idxes = np.random.permutation(len(ann_info))[:num_chosen]
            ann_info = [ann_info[x] for x in chosen_idxes]

        instances = self.parse_ann_info(img_info, ann_info)
        data_info['instances'] = instances

        return data_info

    def parse_ann_info(self, img_info, ann_info):
        ann_type = self.ann_cfg.get('ann_type', 'coco')
        instances = []
        if ann_type == 'coco':
            for i, ann in enumerate(ann_info):
                instance = {}

                if ann.get('ignore', False):
                    continue
                x1, y1, w, h = ann['bbox']
                if self.min_bbox_w > 0:
                    if w <= self.min_bbox_w:
                        x1 -= (self.min_bbox_w - w) / 2
                        w = self.min_bbox_w

                    if h <= self.min_bbox_w:
                        y1 -= (self.min_bbox_w - h) / 2
                        h = self.min_bbox_w

                inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
                inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))

                if inter_w * inter_h == 0:
                    continue
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue
                if ann['category_id'] not in self.cat_ids:
                    continue
                bbox = [x1, y1, x1 + w, y1 + h]

                if ann.get('iscrowd', False):
                    instance['ignore_flag'] = 1
                else:
                    instance['ignore_flag'] = 0
                instance['bbox'] = bbox

                if 'rotated_bbox' in ann:
                    x1, y1, w, h, a = ann['rotated_bbox']
                    w = max(w, self.min_bbox_w)
                    h = max(h, self.min_bbox_w)
                    instance['rotated_bbox'] = [x1,y1,w,h,a]

                instance['bbox_label'] = self.cat2label[ann['category_id']]

                if ann.get('segmentation', None):
                    instance['mask'] = ann['segmentation']

                instances.append(instance)
        elif ann_type == 'json':
            for i, ann in enumerate(ann_info):
                instance = {}

                try:
                    polygon = shapely.geometry.shape(ann['geometry'])

                except Exception as e:
                    print(f'Annotation {ann} has invalid content!')
                    print(f'Error information: {e}')
                    continue

                if self.ann_cfg.get('used_source', None) is not None:
                    if not ann['properties'].get('source', 'none') in self.ann_cfg['used_source']:
                        continue

                if not polygon.is_valid:
                    continue

                if polygon.area < 1:
                    continue

                bbox = list(polygon.bounds)
                x1, y1, x2, y2 = bbox

                if self.bbox_buffer > 0:
                    x1 -= self.bbox_buffer / 2
                    x2 += self.bbox_buffer / 2
                    y1 -= self.bbox_buffer / 2
                    y2 += self.bbox_buffer / 2

                w = x2 - x1
                h = y2 - y1
                if self.min_bbox_w > 0:
                    if w <= self.min_bbox_w:
                        x1 -= (self.min_bbox_w - w) / 2
                        x2 += (self.min_bbox_w - w) / 2

                    if h <= self.min_bbox_w:
                        y1 -= (self.min_bbox_w - h) / 2
                        y2 += (self.min_bbox_w - h) / 2

                bbox = (x1, y1, x2, y2)

                bbox_label = self.cat2label[ann['properties'].get('category_id', 100)]
                segmentation = polygon_utils.poly_json2coco(ann['geometry'])
                # points = np.array(segmentation[0]).reshape(-1,2)
                # x_min, y_min = np.min(points, axis=0)
                # x_max, y_max = np.max(points, axis=0)
                # bbox = [x_min, y_min, x_max, y_max]

                instance['bbox'] = bbox
                instance['bbox_label'] = bbox_label
                instance['mask'] = segmentation
                instance['ignore_flag'] = 0

                instances.append(instance)

        else:
            raise ValueError(f'annotation type {ann_type} is not supported.')

        return instances

    def prepare_data(self, idx):
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        img_id = data_info['img_id']
        img_name = data_info['img_path'].split('/')[-1].split('.')[0]
        ann_path_pattern = self.ann_cfg.get('ann_path_pattern', '{img_id}.json')
        ann_name = ann_path_pattern.format_map(dict(img_id=img_id, img_name=img_name))
        ann_path = os.path.join(self.data_root, self.ann_dir, ann_name)

        try:
            ann_info = json.load(open(ann_path))
            if self.ann_cfg.get('ann_type', 'coco') == 'coco':
                ann_info = ann_info['annotations']
            else:
                ann_info = ann_info['features']
        except Exception as e:
            print(e)
            ann_info = []

        instances = self.parse_ann_info(data_info, ann_info)
        data_info['instances'] = instances

        return self.pipeline(data_info)
