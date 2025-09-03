_base_ = [
    '../_base_/datasets/planet_basemap_2023q2_sample-europe-merged.py', '../_base_/default_runtime.py',
]

# load_from = 'work_dirs/st-mask-rcnn_min-bbox-4_iou-thr-03_r50_100e_planet_basemap_sample-europe/epoch_80.pth'
load_from = 'work_dirs/st-mask-rcnn_st-rpn_r50_100e_planet_basemap_sample-europe/epoch_300.pth'
model = dict(
    type='STMaskRCNN',
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
        'backbone',
        'neck',
        'rpn_head',
        'roi_head.bbox_head',
        'roi_head.mask_head'
    ],
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='STRPNHead',
        st_cfg=dict(
            st_ignore_thr=0.8,
            drop_rate=0.3,
            alpha=0.999,
            init_cls=True,
        ),
        reg_decoded_bbox=False,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2],
            base_sizes=[4, 8, 16, 32, 64],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='STRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=18, sampling_ratio=0),
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
            type='DPPolygonizeHead',
            feat_channels=256,
            poly_cfg=dict(
                sample_iou_thr=0.7,
                num_max_sample=100,
                num_inter_points=64,
                num_primitive_queries=64,
                apply_prim_pred=True,
                step_size=2,
                polygonized_scale=4.,
                max_offsets=5,
                use_coords_in_poly_feat=True,
                use_decoded_feat_in_poly_feat=False,
                use_point_feat_in_poly_feat=True,
                point_as_prim=True,
                pred_angle=False,
                prim_cls_thre=0.1,
                num_cls_channels=2,
                stride_size=64,
                use_ind_offset=True,
                # poly_decode_type='dp',
                poly_decode_type='none',
                reg_targets_type='vertice',
                return_poly_json=False,
                use_gt_jsons=False,
                mask_cls_thre=0.0,
                lam=0,
                map_features=True,
                max_align_dis=4,
                align_iou_thre=0.5,
                num_min_bins=32,
                proj_gt=False,
                loss_weight_dp=0.01,
                max_match_dis=4,
                use_ref_rings=False,
                apply_poly_iou_loss=False,
                sample_points=True,
                max_step_size=128,
                polygonize_mode='cv2_single_mask',
                apply_right_angle_loss=False,
                apply_angle_loss=False
            ),
            decoder=dict(  # Mask2FormerTransformerDecoder
                return_intermediate=True,
                num_layers=3,
                layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                    self_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.0,
                        batch_first=True),
                    cross_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            loss_poly_reg=dict(
                type='SmoothL1Loss',
                reduction='mean',
                loss_weight=1.
            ),
            loss_poly_dp=dict(
                type='SmoothL1Loss',
                reduction='mean',
                loss_weight=0.01
            ),
            loss_poly_right_ang = dict(
                type='SmoothL1Loss',
                reduction='mean',
                loss_weight=10.
            ))
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.3,
                neg_iou_thr=0.1,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.3,
                neg_iou_thr=0.1,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=36,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=8000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.1,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=1200,
            mask_thr_binary=0.5),
        inf_cfg=dict(
            mode='slide', crop_size=(1024, 1024), stride=(1024, 1024),
            crop_up_size=(1024, 1024),
            out_size=None, out_size_scale=1.,
            filter_border_width = 0
        ),
        nms_cfg=dict(
            nms_type='no_overlap',
            iou_thr=0.5
        )
    ))


val_evaluator = [
    dict(
        type='CocoMetric',
        # ann_file = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_sample_europe2/coco_ann/upscale-global_quartely_2023q2.json',
        ann_file = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_test_sample/with_ann/coco_ann/upscale_merged_douglas_global_quartely_2023q2.json',
        metric=['segm'],
        backend_args={{_base_.backend_args}},
        out_cfg=dict(
            save_results=False,
            out_dir='./work_dirs/basemap_pred_results/st-mark-rcnn',
            out_size=(256, 256)
        )
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

max_epochs=300
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
    #     end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[260],
        gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        save_last=True,
        max_keep_ckpts=5,
        interval=1),
    # visualizer=dict(type='WandbVisualizer', wandb_cfg=wandb_cfg, name='wandb_vis')
    visualization=dict(type='TanmlhVisualizationHook', draw=True, interval=50)
)

vis_backends = [
    dict(
        type='WandbVisBackend', save_dir='./wandb/',
        init_kwargs=dict(
            project = 'mmdetection-planet_basemap',
            entity = 'tum-tanmlh',
            name = 'gcp_mask-rcnn_st-rpn_r50_300e_planet_basemap_sample-europe',
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
auto_scale_lr = dict(enable=False, base_batch_size=16)

train_dataloader = dict(
    dataset=dict(
        ann_file = 'coco_ann/small-global_quartely_2023q2.json',
        min_bbox_w=2
    )
)
test_dataloader = dict(
    dataset=dict(
        min_bbox_w=2
    )
)
