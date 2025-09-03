# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple
import time

import shapely
import scipy
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numba
import numba.cuda
from numba import njit
from numba.typed import Dict, List

from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, ModuleList
from mmengine.structures import InstanceData, PixelData
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.mask import mask_target, poly_target, PolygonMasks
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox_overlaps
from mmdet.structures.det_data_sample import DetDataSample
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
import mmdet.utils.tanmlh_polygon_utils as polygon_utils
from mmdet.utils import tanmlh_utils
from mmdet.models.layers import Mask2FormerTransformerDecoder, SinePositionalEncoding

import numpy as np

# ------------------------------------------------------------------
# Numba-accelerated union-find functions.
# ------------------------------------------------------------------

@njit
def find_numba(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

@njit
def union_numba(parent, size, center, max_prob, x, y):
    rx = find_numba(parent, x)
    ry = find_numba(parent, y)
    if rx == ry:
        return rx
    if max_prob[rx] >= max_prob[ry]:
        parent[ry] = rx
        new_size = size[rx] + size[ry]
        center[rx, 0] = (center[rx, 0] * size[rx] + center[ry, 0] * size[ry]) / new_size
        center[rx, 1] = (center[rx, 1] * size[rx] + center[ry, 1] * size[ry]) / new_size
        size[rx] = new_size
        if max_prob[ry] > max_prob[rx]:
            max_prob[rx] = max_prob[ry]
        return rx
    else:
        parent[rx] = ry
        new_size = size[ry] + size[rx]
        center[ry, 0] = (center[ry, 0] * size[ry] + center[rx, 0] * size[rx]) / new_size
        center[ry, 1] = (center[ry, 1] * size[ry] + center[rx, 1] * size[rx]) / new_size
        size[ry] = new_size
        if max_prob[rx] > max_prob[ry]:
            max_prob[ry] = max_prob[rx]
        return ry

# @njit
@njit(parallel=True)
def cluster_by_probs_core(idxes, probs, grid, sorted_ids, diff_thr, conn_thr):
    N = idxes.shape[0]
    parent = np.empty(N, dtype=np.int64)
    for i in range(N):
        parent[i] = i
    size = np.ones(N, dtype=np.int64)
    center = idxes.astype(np.float64).copy()
    max_prob = probs.copy()
    visited = np.zeros(N, dtype=np.bool_)
    # New variables: touched and valid
    touched = np.zeros(N, dtype=np.bool_)  # Track clusters involved in multi-merges
    valid = np.ones(N, dtype=np.bool_)      # True if cluster is never part of multi-merge

    neigh_r = np.array([-1, 1, 0, 0], dtype=np.int64)
    neigh_c = np.array([0, 0, -1, 1], dtype=np.int64)

    # neigh_r = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
    # neigh_c = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)

    for k in range(sorted_ids.shape[0]):
        pid = sorted_ids[k]
        row = idxes[pid, 0]
        col = idxes[pid, 1]
        cur_prob = probs[pid]
        rep_list = np.empty(4, dtype=np.int64)
        rep_count = 0

        # if k % 1000000 == 0:
        #     # time.sleep(0.001) # release GIL lock for not blocking the main thread
        #     temp = 0
        #     for __ in range(1000000):
        #         temp += 1

        # Check 4-connected neighbors
        for j in range(4):
            r = row + neigh_r[j]
            c = col + neigh_c[j]
            if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
                continue
            nb = grid[r][c]
            if nb == -1:
                continue
            if visited[nb]:
                rep = find_numba(parent, nb)
                local_found = False
                for t in range(rep_count):
                    if rep_list[t] == rep:
                        local_found = True
                        break
                if not local_found:
                    rep_list[rep_count] = rep
                    rep_count += 1

        if rep_count == 0:
            # No neighbors: valid remains True
            pass

        elif rep_count == 1:
            # Single neighbor: check if target cluster is touched
            rep = rep_list[0]
            parent[pid] = rep
            new_size = size[rep] + 1
            center[rep, 0] = (center[rep, 0] * size[rep] + row) / new_size
            center[rep, 1] = (center[rep, 1] * size[rep] + col) / new_size
            size[rep] = new_size
            if cur_prob > max_prob[rep]:
                max_prob[rep] = cur_prob
            # Set valid based on target cluster's touched status
            valid[pid] = not touched[rep]
        else:
            # Multiple neighbors: mark clusters as touched
            min_max = max_prob[rep_list[0]]
            for t in range(1, rep_count):
                if max_prob[rep_list[t]] < min_max:
                    min_max = max_prob[rep_list[t]]

            # max_max = max_prob[rep_list[0]]
            # for t in range(1, rep_count):
            #     if max_prob[rep_list[t]] > max_max:
            #         max_max = max_prob[rep_list[t]]
            # if (max_max - cur_prob) < diff_thr or cur_prob > conn_thr:
            if (min_max - cur_prob) < diff_thr:
                # Merge all clusters
                rep = rep_list[0]
                for t in range(1, rep_count):
                    rep = union_numba(parent, size, center, max_prob, rep, rep_list[t])
                parent[pid] = rep
                new_size = size[rep] + 1
                center[rep, 0] = (center[rep, 0] * size[rep] + row) / new_size
                center[rep, 1] = (center[rep, 1] * size[rep] + col) / new_size
                size[rep] = new_size
                if cur_prob > max_prob[rep]:
                    max_prob[rep] = cur_prob
                # Mark as touched and invalid
                # touched[rep] = True
                # valid[pid] = False
            else:
                # Merge into closest cluster
                best_rep = rep_list[0]
                best_dist = (center[best_rep, 0] - row) ** 2 + (center[best_rep, 1] - col) ** 2
                for t in range(1, rep_count):
                    rdist = (center[rep_list[t], 0] - row) ** 2 + (center[rep_list[t], 1] - col) ** 2
                    if rdist < best_dist:
                        best_rep = rep_list[t]
                        best_dist = rdist
                parent[pid] = best_rep
                new_size = size[best_rep] + 1
                center[best_rep, 0] = (center[best_rep, 0] * size[best_rep] + row) / new_size
                center[best_rep, 1] = (center[best_rep, 1] * size[best_rep] + col) / new_size
                size[best_rep] = new_size
                if cur_prob > max_prob[best_rep]:
                    max_prob[best_rep] = cur_prob
                # Mark as touched and invalid
                touched[best_rep] = True
                valid[pid] = False

        visited[pid] = True

    # Final path compression
    for i in range(N):
        parent[i] = find_numba(parent, i)

    return parent, valid


@MODELS.register_module()
class ClusterSeg2InsHead(BaseModule):

    def __init__(self,
                 poly_cfg: OptMultiConfig = {},
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.poly_cfg = poly_cfg

    def loss(self, **kwargs):
        return None

    @staticmethod
    def get_connected_components(labeled_mask):
        """
        Extract row-col index arrays for each connected component in a labeled mask.

        Parameters:
            labeled_mask (np.ndarray): Output from scipy.ndimage.label. Each connected component has a unique label > 0.

        Returns:
            List[np.ndarray]: A list of arrays of shape (N, 2), each containing (row, col) indices of one component.
        """
        slices = scipy.ndimage.find_objects(labeled_mask)
        components = []

        for i, slc in enumerate(slices):
            if slc is None:
                continue  # skip empty slices
            region_mask = (labeled_mask[slc] == (i + 1))
            local_coords = np.argwhere(region_mask)
            row_offset = slc[0].start
            col_offset = slc[1].start
            global_coords = local_coords + np.array([row_offset, col_offset])
            components.append(global_coords)

        return components, slices

    def predict(self, imgs, seg_probs, data_samples):

        import time
        t0 = time.time()
        C, H, W = imgs.shape

        sem_seg_thr = self.poly_cfg.get('sem_seg_thr', 0.5)
        diff_thr = self.poly_cfg.get('diff_thr', 0.1)
        conn_thr = self.poly_cfg.get('conn_thr', 0.8)
        low_thr = self.poly_cfg.get('low_thr', -1)
        cluster_mode = self.poly_cfg.get('cluster_mode', 'early_stop')


        seg_mask = (seg_probs > sem_seg_thr).long()
        # labeled_mask, num_components = scipy.ndimage.label(seg_mask.cpu().numpy())
        # ins_mask = np.zeros_like(labeled_mask)
        ins_mask = np.zeros((seg_mask.shape[0], seg_mask.shape[1]), dtype=np.int64)

        fg_idxes = seg_mask.nonzero()
        fg_probs = seg_probs[fg_idxes[:,0], fg_idxes[:,1]]

        if len(fg_idxes) ==  0:
            return [], torch.zeros(0)

        # cluster_idxes_list = self.cluster_by_probs(fg_idxes.cpu().numpy(), fg_probs.cpu().numpy(),
        #                                            diff_thr, conn_thr, cluster_mode)
        fg_idxes = fg_idxes.cpu().numpy()
        cluster_idxes, valid = self.cluster_by_probs(fg_idxes, fg_probs.cpu().numpy(),
                                                     diff_thr, conn_thr, cluster_mode)
        if low_thr > 0:
            valid |= (fg_probs > low_thr).numpy()

        if cluster_mode == 'early_stop':
            ins_mask[fg_idxes[valid,0], fg_idxes[valid,1]] = cluster_idxes[valid] + 1
        else:
            ins_mask[fg_idxes[:,0], fg_idxes[:,1]] = cluster_idxes + 1

        t1 = time.time()

        _, slices = self.get_connected_components(ins_mask)

        pred_polys, colors = polygon_utils.polygonize_sliced_masks(ins_mask, slices)

        t2 = time.time()

        # pred_polys, colors = polygon_utils.polygonize_mask(
        #     torch.tensor(ins_mask).to(torch.int), mode='simple_mask', scale=1.
        # )

        flat_labels = ins_mask.ravel()
        flat_prob = seg_probs.ravel().cpu().numpy()
        sum_prob = np.bincount(flat_labels, weights=flat_prob)
        pixel_counts = np.bincount(flat_labels)
        scores = sum_prob[1:] / pixel_counts[1:]
        scores = torch.tensor([scores[x-1] for x in colors])

        t3 = time.time()

        # print(f'{ins_mask.max()} {len(pred_polys)} {labeled_mask.max()}')
        # print(f'Cluster time: {t1-t0} {t2-t1} {t3-t2}')

        return pred_polys, scores



    @staticmethod
    def cluster_by_probs(idxes, probs, diff_thr=0.1, conn_thr=0.6, cluster_mode='early_stop'):
        """
        idxes: numpy array of shape (N,2) of pixel coordinates (row, col).
        probs: numpy array of shape (N,1) of pixel probabilities.
        diff_thr: merging threshold.
        
        Returns a list of clusters, where each cluster is a numpy array of pixel coordinates.
        """

        N = idxes.shape[0]
        max_row = int(np.max(idxes[:, 0])) + 1
        max_col = int(np.max(idxes[:, 1])) + 1


        grid = np.full((max_row, max_col), -1, dtype=np.int64)
        grid[idxes[:, 0], idxes[:, 1]] = np.arange(len(idxes))

        sorted_ids = np.argsort(-probs)

        parent, valid = cluster_by_probs_core(idxes, probs, grid, sorted_ids, diff_thr, conn_thr)

        unique_vals, inverse = np.unique(parent, return_inverse=True)
        cluster_idxes = inverse.astype(np.int64)

        return cluster_idxes, valid
