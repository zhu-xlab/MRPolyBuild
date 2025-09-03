# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from functools import partial
from pathlib import Path
import pdb
import os
from tqdm import tqdm
import rasterio
from rasterio.transform import Affine

import numpy as np
import torch
from typing import Callable, Dict, List, Optional, Sequence, Union

from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmdet.registry import MODELS
import mmdet.utils.tanmlh_polygon_utils as polygon_utils
from mmdet.utils.planet_inferencer import InferencePipeline

try:
    from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def load_checkpoint(model,
                    filename: str,
                    map_location: Union[str, Callable] = 'cpu',
                    strict: bool = False,
                    revise_keys: list = [(r'^module.', '')]):
    """Load checkpoint from given ``filename``.

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        map_location (str or callable): A string or a callable function to
            specifying how to remap storage locations.
            Defaults to 'cpu'.
        strict (bool): strict (bool): Whether to allow different params for
            the model and checkpoint.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Defaults to strip
            the prefix 'module.' by [(r'^module\\.', '')].
    """
    checkpoint = _load_checkpoint(filename, map_location=map_location)
    checkpoint = _load_checkpoint_to_model(
        model, checkpoint, strict, revise_keys=revise_keys)

    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=1e9,
        help='num images of calculate model flops')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference(args, logger):
    if digit_version(torch.__version__) < digit_version('1.12'):
        logger.warning(
            'Some config files, such as configs/yolact and configs/detectors,'
            'may have compatibility issues with torch.jit when torch<1.12. '
            'If you want to calculate flops for these models, '
            'please make sure your pytorch version is >=1.12.')

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg.test_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name
    save_cfg = cfg.get('save_cfg', {})

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    result = {}
    cfg.test_dataloader['sampler']['shuffle'] = True
    data_loader = Runner.build_dataloader(cfg.test_dataloader)
    model = MODELS.build(cfg.model)
    if 'load_from' in cfg:
        load_checkpoint(model, cfg.load_from)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    for idx, data_batch in enumerate(tqdm(data_loader, desc='inferencing...')):
        if idx == args.num_images:
            break
        data = model.data_preprocessor(data_batch)
        with torch.no_grad():
            # results = model.predict(data['inputs'], data['data_samples'])
            imgs = data['inputs']
            batch_data_samples = data['data_samples']
            img_path = data['data_samples'][0].metainfo['img_path']
            tif_name = img_path.split('/')[-1].split('.')[0]
            if save_cfg.get('prob_tif_pattern', None) is not None:
                prob_tif_path = save_cfg['prob_tif_pattern'].format(tif_name)
                if not os.path.exists(prob_tif_path):
                    continue

                results = batch_data_samples
                with rasterio.open(prob_tif_path) as src:
                    seg_probs = torch.tensor(src.read(3))
                    seg_probs = seg_probs[None, None].repeat(1,2,1,1)
                    seg_probs[seg_probs.isnan()] = 0
                    results[0].seg_probs = seg_probs
                    save_cfg['out_poly_scale'] = 1
                    transform = src.transform
                    crs = src.crs
            else:
                results = model.predict_sem_seg(imgs, batch_data_samples) # GPU
                results = model.predict_mosaic_sem_seg(imgs, batch_data_samples) # CPU

            results = model.seg_poly_head.predict_seg2ins(imgs, results) # CPU
            # results = model.seg_poly_head.predict_prepare_instances(imgs, results)

            results = model.seg_poly_head.poly_head.predict_sample_segments(imgs, results) # CPU
            results = model.seg_poly_head.poly_head.predict_gcp(imgs, results) # GPU
            results = model.seg_poly_head.poly_head.predict_assemble_segments(imgs, results) # CPU
            results = model.seg_poly_head.poly_head.predict_dp(imgs, results) # GPU

        if save_cfg.get('save_results', False):
            poly_jsons = results[0].pred_instances['segmentations']
            file_name = results[0].metainfo['img_path'].split('/')[-1].split('.')[0]
            height = results[0].seg_probs.shape[2]  # 图像高度
            width = results[0].seg_probs.shape[3]   # 图像宽度
            # transform = results[0].metainfo['tif_meta']['transform']

            out_dir = save_cfg['out_dir']
            out_geojson_dir = os.path.join(out_dir, 'geojson')
            out_tif_dir = os.path.join(out_dir, 'tif')
            os.makedirs(out_geojson_dir, exist_ok=True)
            os.makedirs(out_tif_dir, exist_ok=True)

            out_geojson_path = os.path.join(out_geojson_dir, file_name + '.geojson')
            out_tif_path = os.path.join(out_tif_dir, file_name + '.tif')

            out_scale = save_cfg.get('out_poly_scale', 1.0)

            os.makedirs(out_dir, exist_ok=True)

            # crs = results[0].metainfo['tif_meta']['crs']
            scores = results[0].scores

            polygon_utils.save_polygons(
                poly_jsons, transform, crs, out_geojson_path, out_scale,
                properties=dict(scores=results[0].scores)
            )


            original_transform = results[0].metainfo['tif_meta']['transform']

            pixel_size_x = original_transform.a / 8
            pixel_size_y = original_transform.e / 8

            adjusted_transform = Affine(
                pixel_size_x,            # 新的x方向分辨率
                0,                       # 无旋转
                original_transform.c,    # 左上角x坐标保持不变
                0,                       # 无旋转
                pixel_size_y,            # 新的y方向分辨率
                original_transform.f      # 左上角y坐标保持不变
            )

            if adjusted_transform.e > 0:
                # 确保y方向分辨率为负值（图像从上到下）
                adjusted_transform = Affine(
                    adjusted_transform.a,
                    adjusted_transform.b,
                    adjusted_transform.c,
                    adjusted_transform.d,
                    -adjusted_transform.e,
                    adjusted_transform.f
                )
            seg_probs = results[0].seg_probs[0,1] # (H, W)
            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,  # 单波段
                'dtype': rasterio.float32,
                'crs': crs,
                'transform': adjusted_transform,
                'compress': 'lzw',  # 压缩
                'nodata': 0         # 无数据值
            }

            with rasterio.open(out_tif_path, 'w', **profile) as dst:
                dst.write(seg_probs, 1)

    del data_loader

    return result


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)

if __name__ == '__main__':
    main()
