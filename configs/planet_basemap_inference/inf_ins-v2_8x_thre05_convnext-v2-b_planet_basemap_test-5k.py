_base_ = [
    '../_base_/datasets/planet_basemap_single_ann_2023q2_global_8x.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
load_from = 'work_dirs/gcp_ins-v2_8x_right-ang_seg-based-det_convnext-v2-b_320k_planet_basemap_global/iter_320000.pth'

model = dict(
    type='SegBasedDetector',
    data_preprocessor = dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255,
    ),
    frozen_parameters=[
        'backbone', 'seg_head',
    ],
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        # TODO: verify stochastic depth rate {0.1, 0.2, 0.3, 0.4}
        drop_path_rate=0.4,
        layer_scale_init_value=0.,  # disable layer scale when using GRN
        gap_before_final_norm=False,
        use_grn=True,  # V2 uses GRN
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth',
            prefix='backbone.')
    ),
    seg_head=dict(
        type='mmseg.UPerHead',
        # in_channels=[256, 512, 1024, 2048],
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        # norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    seg_poly_head=dict(
        type='SegPolyHead',
        poly_cfg=dict(
            poly_iou_thr=0.3,
            max_offsets=20,
            train_poly_head=True,
            sem_seg_thr=0.5,
            num_max_sample=200,
            train_seg2ins_head=False,
            use_roi_mask_feat=False
        ),
        seg2ins_head=dict(
            type='ClusterSeg2InsHead',
            poly_cfg=dict(
                sem_seg_thr=0.5,
                diff_thr=0.05,
                cluster_mode='late_stop'
            )
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        train_poly_head=False,
    ),
    test_cfg=dict(
        inf_cfg=dict(
            mode='slide', crop_size=(2048, 2048), stride=(2048, 2048),
            crop_up_size=(2048, 2048),
            out_size=None, out_size_scale=1.,
            filter_border_width = 0,
            sem_seg_type='sem_seg',
            sem_seg_thr=0.5,
            eval_proposal=False,
            split_batch_size=2,
        ),
        post_cfg = dict(
            type='InstancePostProcessor',
            out_size=(2048, 2048),
            do_crop_to_boundary=True,
            max_area = 1600,
            # crop_box = (32 * 4, 32 * 4, (4096 + 32) * 4, (4096 + 32) * 4),
            crop_box = (0,0,2048, 2048),
            do_merge_with_fg=True,
            do_filter_large=False,
            nms_cfg=dict(
                nms_type='polygon',
                iou_thr=0.8
            ),
        )
    ))

test_evaluator = [
]
test_cfg = dict(type='TestLoop')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

save_cfg=dict(
    save_results=True,
    out_dir='./work_dirs/basemap_pred_results/inf_ins_v2_8x_thre05_planet-test-5k',
    out_poly_scale=1/8.,
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        max_keep_ckpts=10,
        interval=8000),
    ema=dict(
        type='EMAHook', momentum=0.01, interval=1
    ),
    # visualizer=dict(type='WandbVisualizer', wandb_cfg=wandb_cfg, name='wandb_vis')
    visualization=dict(type='TanmlhVisualizationHook', draw=True, interval=5, score_thr=0.1)
)

vis_backends = [
    dict(
        type='WandbVisBackend', save_dir='./wandb/',
        init_kwargs=dict(
            project = 'planet_basemap',
            entity = 'tum-tanmlh',
            name = 'gcp_ins-v2_8x_right-ang_seg-based-det_convnext-v2-b_320k_planet_basemap_global',
            resume = 'never',
            dir = './work_dirs/',
            allow_val_change=True
        ),
    )
]
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TanmlhVisualizer', vis_backends=vis_backends, name='visualizer'
)
# find_unused_parameters=True


test_dataloader = dict(
    num_workers=4,
    dataset=dict(
        ann_file = 'coco_ann_global/test_continent_global_quartely_2023q2.json',
    )
)
