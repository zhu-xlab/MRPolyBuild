_base_ = [
    '../_base_/datasets/planet_basemap_single_ann_2023q2_global_8x.py', '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
load_from = 'work_dirs/seg-based-det_8x_multi-source_convnext-v2-b_50e_planet_basemap_global/epoch_50.pth'

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
            sem_seg_thr=0.4,
            num_max_sample=200,
            train_seg2ins_head=False,
            use_roi_mask_feat=False
        ),
        seg2ins_head=dict(
            type='ClusterSeg2InsHead',
            poly_cfg=dict(
                sem_seg_thr=0.4,
                diff_thr=0.05,
                cluster_mode='late_stop'
            )
        ),
        poly_head=dict(
            type='GCPPolyHead',
            feat_channels=256,
            in_feat_channels=7 * 7 * 2,
            poly_cfg=dict(
                unfold_cfg=dict(
                    kernel_size=7, stride=1
                ),
                # mask_feat_type='img_prob',
                disable_mask_feat=True,
                mask_feat_type='prob',
                align_pred_gt=True,
                sample_iou_thr=0.3,
                num_max_sample=200,
                num_inter_points=64,
                step_size=16,
                polygonized_scale=4.,
                max_offsets=20,
                use_decoded_feat_in_poly_feat=False,
                num_cls_channels=2,
                stride_size=64,
                lam=4,
                max_align_dis=16,
                num_min_bins=32,
                loss_weight_dp=0.01,
                max_step_size=128,
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
                        act_cfg=dict(type='ReLU', inplace=True)
                    )),
                init_cfg=None),
            loss_poly_reg=dict(
                type='SmoothL1Loss',
                reduction='mean',
                loss_weight=1.
            ),
            loss_poly_right_ang = dict(
                type='SmoothL1Loss',
                reduction='mean',
                loss_weight=10.
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
            sem_seg_thr=0.4,
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
                save_results=False,
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

max_iters=160000
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
    #     end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=160000,
        by_epoch=False,
        milestones=[120000],
        gamma=0.1)
]

# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        max_keep_ckpts=10,
        interval=16000),
    # ema=dict(
    #     type='EMAHook', momentum=0.01, interval=1
    # ),
    # visualizer=dict(type='WandbVisualizer', wandb_cfg=wandb_cfg, name='wandb_vis')
    # visualization=dict(type='TanmlhVisualizationHook', draw=True, interval=5, score_thr=0.1)
)

vis_backends = [
    dict(
        type='WandbVisBackend', save_dir='./wandb/',
        init_kwargs=dict(
            project = 'planet_basemap',
            entity = 'tum-tanmlh',
            name = 'gcp_8x_no-pfs_late-stop_seg-based-det_convnext-v2-b_160k_planet_basemap_global',
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


# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=2)

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        ann_dir = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v2/merged_ann_v2',
        ann_cfg=dict(
            ann_path_pattern = '{img_name}.json',
            ann_type='json',
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
