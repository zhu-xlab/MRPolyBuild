# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import pdb
import torch

from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmdet.models.utils import mask2ndarray
from mmdet.registry import DATASETS, VISUALIZERS
from mmdet.structures.bbox import BaseBoxes
import torch.fft as fft

def fft_fun(x, is_shift=False):
    spectrum = fft.fft2(x, dim=(-2, -1))
    phase = torch.angle(spectrum)
    magnitude = torch.abs(spectrum)
    if is_shift:
        magnitude = fft.ifftshift(magnitude)
    return phase, magnitude

def ifft(magnitude, phase):
    reconstructed_spectrum = magnitude * torch.exp(1j * phase)
    reconstructed_x = fft.ifft2(reconstructed_spectrum, dim=(-2, -1)).real
    return reconstructed_x

def magnitude_mixup(x):
    # extract pahse and manigtude from images by DCT
    phase, magnitude = fft_fun(x)

    # enhance: magnitude mixup
    batch_size = x.size(0)
    lam = torch.rand(batch_size).to(x.device).detach()\
        .unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    # index = torch.randperm(batch_size)
    index = (torch.arange(batch_size) + 1) % batch_size
    mixed_magnitude = lam * magnitude + (1-lam) * magnitude[index]

    # reconstruct images
    reconstructed_x = ifft(mixed_magnitude, phase)
    return reconstructed_x


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
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


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    imgs = []
    data_samples = []
    base_names = []
    out_files = []
    cnt = 0
    progress_bar = ProgressBar(len(dataset))
    for item in dataset:
        img = item['inputs'].permute(1, 2, 0).numpy()
        data_sample = item['data_samples'].numpy()
        gt_instances = data_sample.gt_instances
        img_path = osp.basename(item['data_samples'].img_path)

        out_file = osp.join(
            args.output_dir,
            osp.basename(img_path)) if args.output_dir is not None else None

        img = img[..., [2, 1, 0]]  # bgr to rgb

        imgs.append(torch.tensor(img))
        cnt += 1
        if cnt >= 100:
            break

        gt_bboxes = gt_instances.get('bboxes', None)
        if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
            gt_instances.bboxes = gt_bboxes.tensor
        gt_masks = gt_instances.get('masks', None)
        if gt_masks is not None:
            masks = mask2ndarray(gt_masks)
            gt_instances.masks = masks.astype(bool)

        # data_sample.gt_instances = gt_instances
        """
        visualizer.add_datasample(
            osp.basename(img_path),
            img,
            data_sample,
            draw_pred=False,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file
        )
        """
        base_names.append(osp.basename(img_path))
        data_samples.append(data_sample)
        out_files.append(out_file)
        progress_bar.update()

    imgs = torch.stack(imgs).permute(0,3,1,2)
    mixed_imgs = magnitude_mixup(imgs)

    for i, (img, mixed_img, basename, data_sample) in enumerate(zip(imgs, mixed_imgs, base_names, data_samples)):
        img = img.permute(1,2,0).numpy()
        mixed_img = mixed_img.permute(1,2,0).numpy()

        visualizer.add_datasample(
            'ori_img',
            img,
            # data_sample,
            None,
            draw_pred=False,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file,
            step=i
        )

        visualizer.add_datasample(
            'mixed_img',
            mixed_img,
            # data_sample,
            None,
            draw_pred=False,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file,
            step=i
        )


if __name__ == '__main__':
    main()
