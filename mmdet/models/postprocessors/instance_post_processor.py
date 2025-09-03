import torch
import numpy as np
import pdb
import shapely
from shapely.geometry import shape, box, mapping
import scipy

from mmdet.registry import MODELS
from mmdet.utils import tanmlh_utils
from mmdet.utils import tanmlh_polygon_utils as polygon_utils
from mmdet.structures.mask import PolygonMasks
from mmengine.structures import InstanceData, PixelData
from mmdet.structures.bbox import bbox_overlaps, obb2xyxy


@MODELS.register_module()
class InstancePostProcessor:
    def __init__(
        self, out_size, nms_cfg={}, do_crop_to_boundary=True,
        do_filter_large=True, do_remove_small_holes=True, small_hole_area=64, max_area=900, crop_box=None, do_merge_with_fg=False,
        out_cfg={}
    ):
        self.out_size = out_size
        self.nms_cfg = nms_cfg
        self.do_crop_to_boundary = do_crop_to_boundary
        self.do_filter_large = do_filter_large
        self.max_area = max_area
        self.crop_box = crop_box
        self.do_merge_with_fg = do_merge_with_fg
        self.do_remove_small_holes = do_remove_small_holes
        self.small_hole_area = small_hole_area


    def process(self, pred_results):
        assert 'pred_instances' in pred_results
        pred_instances = pred_results.pred_instances

        if self.do_filter_large:
            pred_instances = self.filter_large_instance(pred_instances, self.max_area)

        if self.do_crop_to_boundary:
            pred_instances = self.crop_to_boundary(pred_instances, self.crop_box)

        pred_instances = tanmlh_utils.instance_nms(pred_instances, self.nms_cfg)

        if self.do_merge_with_fg:
            fg_instances = pred_results.fg_instances
            pred_instances = self.merge_det_fg_instance(pred_instances, fg_instances)

        if self.do_remove_small_holes:
            pred_instances = self.remove_small_holes(pred_instances, self.small_hole_area)

        pred_results.pred_instances = pred_instances
        return pred_results

    def merge_det_fg_instance(self, det_instances, fg_instances, min_fg_area=64, iou_thr=0.5):
        if 'polygon_masks' in det_instances:
            pass
        elif 'segmentations' in det_instances:
            pred_polys = det_instances['segmentations']
            pred_polys = PolygonMasks.from_json(pred_polys, self.out_size[0], self.out_size[1])
            det_instances.polygon_masks = pred_polys
        else:
            raise ValueError('polygons must exist in the instances!')


        det_instances = det_instances[det_instances.polygon_masks.get_valid_idxes()]

        if 'polygon_masks' in fg_instances:
            pass
        elif 'segmentations' in fg_instances:
            fg_polys = fg_instances['segmentations']
            fg_polys = PolygonMasks.from_json(fg_polys, self.out_size[0], self.out_size[1])
            fg_instances.polygon_masks = fg_polys
        else:
            raise ValueError('polygons must exist in the instances!')

        fg_instances = fg_instances[fg_instances.polygon_masks.get_large_idxes(min_area=min_fg_area)]
        fg_instances = fg_instances[fg_instances.polygon_masks.get_valid_idxes()]

        iou_mat = tanmlh_utils.poly_overlaps(fg_instances.polygon_masks, det_instances.polygon_masks, iou_type='half_iou')
        new_fg_idxes = (iou_mat.sum(axis=1) < iou_thr).nonzero()[0]
        keep_pred_idxes = (iou_mat[new_fg_idxes] < 1e-8).all(axis=0).nonzero()[0]

        merged_instances = det_instances.cat([det_instances[keep_pred_idxes], fg_instances[new_fg_idxes]])
        # bbox_iou = bbox_overlaps(torch.tensor(fg_instances[new_fg_idxes].polygon_masks.get_bounds()),
        #                          torch.tensor(det_instances.polygon_masks.get_bounds()))
        # merged_instances2 = tanmlh_utils.instance_nms(merged_instances, self.nms_cfg)
        # if len(merged_instances) != len(merged_instances2):
        #     iou_mat2 = tanmlh_utils.poly_overlaps(fg_instances[new_fg_idxes].polygon_masks,
        #                                           det_instances.polygon_masks, iou_type='half_iou',
        #                                           debug=True)
        #     pdb.set_trace()

        return merged_instances

    def drop_invalid_polygons(self, pred_instances):

        pred_polys = None
        if 'polygon_masks' in pred_instances:
            pred_polys = pred_instances.polygon_masks

        elif 'segmentations' in pred_instances:
            pred_polys = pred_instances.segmentations
            pred_polys = PolygonMasks.from_json(pred_polys, self.out_size[0], self.out_size[1])
            pred_instances.polygon_masks = pred_polys
        else:
            raise ValueError('polygons must exist in the instances!')

        assert pred_polys is not None, 'segmentations or polygon_masks must exists in the instances!'

        keep_idxes = pred_polys.get_valid_idxes()
        if len(keep_idxes) > 0:
            pred_instances = pred_instances[keep_idxes]
            pred_instances.polygon_masks = pred_polys[keep_idxes]
        else:
            pred_instances = pred_instances[:0]
            pred_instances.polygon_masks = pred_polys[:0]

        return pred_instances

    def filter_large_instance(self, pred_instances, max_area=900):
        pred_polys = None
        if 'polygon_masks' in pred_instances:
            pred_polys = pred_instances.polygon_masks
        elif 'segmentations' in pred_instances:
            pred_polys = pred_instances.segmentations
            pred_polys = PolygonMasks.from_json(pred_polys, self.out_size[0], self.out_size[1])
            pred_instances.polygon_masks = pred_polys
        else:
            raise ValueError('polygons must exist in the instances!')

        keep_idxes = pred_polys.get_small_idxes(max_area)
        pred_instances = pred_instances[keep_idxes]
        pred_instances.polygon_masks = pred_polys[keep_idxes]

        return pred_instances

    def remove_small_holes(self, pred_instances, min_area=64):
        pred_polys = None
        if 'polygon_masks' in pred_instances:
            pred_polys = pred_instances.polygon_masks
        elif 'segmentations' in pred_instances:
            pred_polys = pred_instances.segmentations
            pred_polys = PolygonMasks.from_json(pred_polys, self.out_size[0], self.out_size[1])
            pred_instances.polygon_masks = pred_polys
        else:
            raise ValueError('polygons must exist in the instances!')

        new_polygon_masks = pred_instances.polygon_masks.remove_small_holes(min_area)
        pred_instances.polygon_masks = new_polygon_masks
        pred_instances.segmentations = new_polygon_masks.to_json()

        return pred_instances

    def crop_to_boundary(self, pred_instances, crop_box, min_area=16):
        pred_instances = self.drop_invalid_polygons(pred_instances)

        if len(pred_instances) == 0:
            return pred_instances

        polygon_masks = pred_instances.polygon_masks
        polygon_masks = polygon_masks.fast_crop_shapely(np.array(crop_box))
        pred_instances.polygon_masks = polygon_masks

        # drop invalid polygons
        valid_idxes = pred_instances.polygon_masks.get_valid_idxes()
        pred_instances = pred_instances[valid_idxes]

        # drop small polygons
        valid_idxes = pred_instances.polygon_masks.get_large_idxes(min_area)
        pred_instances = pred_instances[valid_idxes]

        if 'segmentations' in pred_instances:
            pred_instances.segmentations = pred_instances.polygon_masks.to_json()

        return pred_instances
