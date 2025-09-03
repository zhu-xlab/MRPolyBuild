# dataset settings
dataset_type = 'PlanetBasemapDataset'
data_root = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
backend_args = None
crop_size = (1024, 1024)

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


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False, with_poly_json=False),
    dict(type='LoadTIFMetaInfo'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='Rotate90', prob=0.75),
    dict(type='LoadSegFromPolygonMasks'),
    # dict(type='CropFeaturesToBounds'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    # dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_semantic_seg', 'eroded_gt_semantic_seg'], cpu_keys=['features']),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'tif_meta')),
    dict(type='PreLoadShapely'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False, poly2mask=False, with_poly_json=False),
    dict(type='LoadTIFMetaInfo'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=False, with_mask=True, poly2mask=False, with_poly_json=False, with_label=False),
    # dict(type='RandomCrop', crop_size=crop_size),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'tif_meta', 'continent'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    # persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='PlanetBasemapSingleAnnDataset',
        lazy_init=True,
        data_root = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v2',
        ann_dir='coco_json_files/train_global_quartely_2023q2/ann',
        pipeline=train_pipeline,
        backend_args=backend_args,
        ann_file = 'coco_json_files/train_global_quartely_2023q2/coco.json',
        data_prefix=dict(img=''),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    # persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        backend_args=backend_args,
        ann_file = 'coco_ann_full/small_test_global_quartely_2023q2.json',
        data_prefix=dict(img=''),
        test_mode=True,
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    # persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        backend_args=backend_args,
        # ann_file = 'coco_ann/merged_v2_global_quartely_2023q2.json',
        ann_file = 'coco_ann_full/small_test_global_quartely_2023q2.json',
        # ann_file = 'coco_ann/small-global_quartely_2023q2.json',
        data_prefix=dict(img=''),
        test_mode=True,
        min_bbox_w=2
    )
)
