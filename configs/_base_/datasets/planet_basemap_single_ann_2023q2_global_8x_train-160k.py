# dataset settings
dataset_type='PlanetBasemapSingleAnnDataset'
data_root = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v3'
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
    dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='Rotate90', prob=0.75),
    dict(type='LoadSegFromPolygonMasks'),
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
    dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=False, with_mask=True, poly2mask=False, with_poly_json=False, with_label=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'tif_meta', 'continent'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        lazy_init=True,
        data_root = data_root,
        ann_dir='train/merged_ann',
        pipeline=train_pipeline,
        backend_args=backend_args,
        # ann_file = 'coco_json_files/train_global_quartely_2023q2/coco.json',
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
        lazy_init=True,
        data_root = data_root,
        # data_root = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v3/test_6k',
        ann_file = 'test_6k/coco/coco_no_ann.json',
        ann_dir='test_6k/osm/geojson',
        pipeline=test_pipeline,
        backend_args=backend_args,
        data_prefix=dict(img=''),
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
        lazy_init=True,
        data_root = data_root,
        ann_file = 'test_6k/coco/coco_no_ann.json',
        ann_dir='test_6k/osm/geojson',
        pipeline=test_pipeline,
        backend_args=backend_args,
        data_prefix=dict(img=''),
    )
)
