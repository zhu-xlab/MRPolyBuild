_base_ = [
    '../_base_/datasets/planet_basemap_single_ann_2023q2_global_512x512.py', '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
# load_from = 'work_dirs/st-mask-rcnn_merged_min-bbox-2_iou-thr-03_r50_100e_planet_basemap_sample-europe/epoch_80.pth'
model = dict(
    type='SegMaskRCNN',
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
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5),
    seg_head=dict(
        type='mmseg.UPerHead',
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
    rpn_head=dict(
        type='UpRPNHead',
        custom_cfg=dict(
            up_feat_levels = [0,1,2,3,4]
        ),
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[1],
            base_sizes=[8, 16, 32, 64, 128],
            ratios=[0.5, 1.0, 2.0],
            strides=[2, 4, 8, 16, 32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=50.0),
        loss_bbox=dict(type='L1Loss', loss_weight=100.0)),
    roi_head=dict(
        type='PolyRoIHead',
        poly_cfg=dict(
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        poly_head=dict(
            type='NaivePolyHead'
        )
    ),
    # model training and testing settings
    train_cfg=dict(
        train_poly_head=False,
        rpn=dict(
            assigner=dict(
                type='HierarchicalAssigner',
                img_size=(1024, 1024),
                grid_size=(256, 256),
                base_assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
            ),
            sampler=dict(
                type='RandomSampler',
                # num=4096 * 8,
                num=512*512,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2048,
            max_per_img=1024,
            nms=dict(type='nms', iou_threshold=0.5),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.3,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=1.,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        # seg_head=dict(
        #     up_feat_levels = [0,1,2,3]
        # ),
        rpn=dict(
            nms_pre=4096,
            max_per_img=2048,
            nms=dict(type='nms', iou_threshold=0.5),
            min_bbox_size=0,
            score_thr=0.2
        ),
        rcnn=dict(
            # not used
            score_thr=0.1,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=2048,
            mask_thr_binary=0.5,
            block_mask_predict=True
        ),
        inf_cfg=dict(
            mode='slide', crop_size=(1024, 1024), stride=(1024, 1024),
            crop_up_size=(1024, 1024),
            out_size=None, out_size_scale=1.,
            filter_border_width = 0,
            sem_seg_type='sem_seg',
            sem_seg_thr=0.5,
            eval_proposal=False
        ),
        post_cfg = dict(
            type='InstancePostProcessor',
            out_size=(1024, 1024),
            do_crop_to_boundary=True,
            max_area = 1600,
            # crop_box = (32 * 4, 32 * 4, (4096 + 32) * 4, (4096 + 32) * 4),
            crop_box = (0,0,1024, 1024),
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
        # ann_file = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2/coco_ann_global/upscale_test_continent_global_quartely_2023q2.json',
        # ann_file = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2/coco_ann_full/upscale_merged_test_continent_global_quartely_2023q2.json'
        ann_file = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2/coco_ann_full/upscale_merged_filtered_test_dp_global_quartely_2023q2.json',
        # metric=['bbox'],
        # metric=['map_fast', 'proposal_fast', 'bbox_fast'],
        # metric=['poly_ap_fast', 'map_fast', 'bbox_fast'],
        metric=['poly_ap_fast'],
        # split_meta_key='continent',
        backend_args={{_base_.backend_args}},
        out_cfg=dict(
            save_results=False,
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
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=False,
        milestones=[40],
        gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=200)
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
    visualization=dict(type='TanmlhVisualizationHook', draw=True, interval=5, score_thr=0.1)
)

vis_backends = [
    dict(
        type='WandbVisBackend', save_dir='./wandb/',
        init_kwargs=dict(
            project = 'planet_basemap',
            entity = 'tum-tanmlh',
            name = 'seg-mask-rcnn_convnext-v2-b_50e_planet_basemap_global',
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
        bbox_buffer=0.5
    )
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file = 'coco_ann_full/small_merged_filtered_test_dp_global_quartely_2023q2.json',
        min_bbox_w=2
    )
)
test_dataloader = dict(
    dataset=dict(
        ann_file = 'coco_ann_full/small_merged_filtered_test_dp_global_quartely_2023q2.json',
        min_bbox_w=2
    )
)
