_base_ = [
    '../_base_/datasets/planet_basemap_single_ann_2023q2_global_8x.py', '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
# load_from = 'work_dirs/st-mask-rcnn_merged_min-bbox-2_iou-thr-03_r50_100e_planet_basemap_sample-europe/epoch_80.pth'

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
    frozen_parameters=[],
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
            poly_iou_thr=0.5,
            train_poly_head=False,
            sem_seg_thr=0.5,
            num_max_sample=200,
            train_seg2ins_head=False,
        ),
        seg2ins_head=dict(
            type='ClusterSeg2InsHead',
            poly_cfg=dict(
                sem_seg_thr=0.5,
                diff_thr=0.1
            )
        )
    ),
    # model training and testing settings
    train_cfg=dict(
        train_poly_head=False,
        # seg_head=dict(
        #     up_feat_levels = [0,1,2,3]
        # ),
    ),
    test_cfg=dict(
        # seg_head=dict(
        #     up_feat_levels = [0,1,2,3]
        # ),
        inf_cfg=dict(
            mode='slide', crop_size=(2048, 2048), stride=(2048, 2048),
            crop_up_size=(2048, 2048),
            out_size=None, out_size_scale=1.,
            filter_border_width = 0,
            sem_seg_type='sem_seg',
            sem_seg_thr=0.5,
            eval_proposal=False
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
            out_cfg=dict(
                save_results=True,
                out_dir='./work_dirs/basemap_pred_results/st-mark-rcnn-v2_convnext-v2-b',
                out_poly_scale=1/4.,
            )
        )
    ))

val_evaluator = [
    dict(
        type='PlanetMetric',
        # ann_file = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_sample_europe2/coco_ann/upscale-global_quartely_2023q2.json',
        # ann_file = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2/coco_ann_full/upscale_test_global_quartely_2023q2.json',
        # ann_file = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2/coco_ann_full/upscale_merged_filtered_test_dp_global_quartely_2023q2.json',
        ann_file = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2/coco_ann_global/upscale_test_continent_global_quartely_2023q2.json',
        # ann_file = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2/coco_ann_full/upscale_merged_test_continent_global_quartely_2023q2.json'
        # metric=['bbox'],
        # metric=['map_fast', 'proposal_fast', 'bbox_fast'],
        # metric=['poly_ap_fast', 'map_fast', 'bbox_fast'],
        metric=['poly_ap_fast'],
        # split_meta_key='continent',
        backend_args={{_base_.backend_args}},
        out_cfg=dict(
            save_results=True,
            out_dir='./work_dirs/basemap_pred_results/st-mark-rcnn-v2_convnext-v2-b',
            out_size=(256, 256)
        ),
        min_bbox_size=0,
        iou_thrs=[0.3, 0.35, 0.4, 0.45, 0.5],
        proposal_nums=[128, 256, 512, 1024, 2048]
    )
]
test_evaluator = val_evaluator

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

max_epochs=50
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
    #     end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=800000, val_interval=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        save_last=True,
        max_keep_ckpts=10,
        interval=1),
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
            name = 'ins-cluster_seg-based-det_8x_multi-source_convnext-v2-b_50e_planet_basemap_global',
            resume = 'never',
            dir = './work_dirs/',
            allow_val_change=True
        ),
    )
]
# vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TanmlhVisualizer', vis_backends=vis_backends, name='visualizer'
)


# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=2)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        ann_dir = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v2/merged_ann_v2',
        ann_cfg=dict(
            ann_path_pattern = '{img_name}.json',
            ann_type='json',
            # used_source=['osm'],
        ),
        min_bbox_w=2,
    )
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        # ann_file = 'coco_ann_full/small_merged_filtered_test_dp_global_quartely_2023q2.json',
        # ann_file = 'coco_ann_global/small_test_continent_global_quartely_2023q2.json',
        # ann_file = 'coco_ann_full/small_merged_filtered_test_dp_global_quartely_2023q2.json',
        # ann_file = 'coco_ann_global/small_merged_test_continent_global_quartely_2023q2.json',
        ann_file = 'coco_ann_global/small_test_continent_global_quartely_2023q2.json',
        min_bbox_w=2
    )
)
test_dataloader = dict(
    dataset=dict(
        # ann_file = 'coco_ann_full/filtered_test_global_quartely_2023q2.json',
        # ann_file = 'coco_ann_full/small_merged_filtered_test_dp_global_quartely_2023q2.json',
        # ann_file = 'coco_ann_full/small_merged_filtered_test_dp_global_quartely_2023q2.json',
        # ann_file = 'coco_ann_global/small_merged_test_continent_global_quartely_2023q2.json',
        # ann_file = 'coco_ann_global/test_continent_global_quartely_2023q2.json',
        # ann_file = 'coco_ann_global/small_test_continent_global_quartely_2023q2.json',
        # ann_file = 'coco_ann_full/small_merged_filtered_test_dp_global_quartely_2023q2.json',
        ann_file = 'coco_ann_global/small_test_continent_global_quartely_2023q2.json',
        min_bbox_w=2
    )
)
