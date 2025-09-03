# dataset settings
dataset_type = 'PlanetBasemapNoAnnDataset'
data_root = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v3/test_lcz'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
backend_args = None
crop_size = (512, 512)

batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
    batch_augments=batch_augments
)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False, with_poly_json=False),
    dict(type='LoadTIFMetaInfo'),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'tif_meta'))
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    # persistent_workers=True,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        backend_args=backend_args,
        img_suffix='',
        ann_file=f'{data_root}/list.txt',
        # data_prefix=dict(img_path='img'),
    )
)
