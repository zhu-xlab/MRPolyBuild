_base_ = [
    '../../../configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/isaid_instance.py',
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]

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
    visualization=dict(type='TanmlhVisualizationHook', draw=True, interval=50, score_thr=0.1)
)

vis_backends = [
    dict(
        type='WandbVisBackend', save_dir='./wandb/',
        init_kwargs=dict(
            project = 'mmdetection-planet_basemap',
            entity = 'tum-tanmlh',
            name = 'mask-rcnn_r50_fpn_isaid',
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
