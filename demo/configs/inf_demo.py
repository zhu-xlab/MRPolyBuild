_base_ = '../../configs/planet_basemap/gcp_ins-v2_8x_right-ang-v2_seg-based-det_convnext-v2-b_320k_planet_basemap_global.py'

custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)

load_from = 'checkpoints/gcp_ins-v2_8x_right-ang-v2_seg-based-det_convnext-v2-b_320k_planet_basemap_global/iter_320000.pth'

test_data_root = 'demo/data'
product_name = 'gcp_ins-v2_8x_right-ang-v2_320k_demo'

# Official MRPolyBuild 8x inference settings:
model = dict(
    test_cfg=dict(
        inf_cfg=dict(
            mode='slide',
            crop_size=(256, 256),
            stride=(192, 192),
            crop_up_size=(2048, 2048),
            out_size=None,
            out_size_scale=8.,
            filter_border_width=0,
            sem_seg_type='sem_seg',
            sem_seg_thr=0.4,
            eval_proposal=False,
            split_batch_size=2,
        ),
        post_cfg=dict(
            type='InstancePostProcessor',
            out_size=(2048, 2048),
            do_crop_to_boundary=True,
            max_area=1600,
            crop_box=(0, 0, 2048, 2048),
            do_merge_with_fg=True,
            do_filter_large=False,
            nms_cfg=dict(nms_type='polygon', iou_thr=0.8),
        ),
    ))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadTIFMetaInfo'),
    dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'tif_meta')),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='PlanetBasemapNoAnnDataset',
        data_root=test_data_root,
        ann_file='all.txt',
        img_suffix='',
        pipeline=test_pipeline,
        backend_args=None,
    ))

test_evaluator = []

save_cfg = dict(
    save_results=True,
    out_dir='demo/output/' + product_name,
    out_poly_scale=1 / 8.,
)
