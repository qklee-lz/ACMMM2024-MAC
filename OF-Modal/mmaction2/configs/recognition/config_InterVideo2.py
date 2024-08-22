_base_ = ['../_base_/default_runtime.py']

# model settings

num_frames = 8
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='InternVideo2',
        img_size=224,
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48/11,
        attn_pool_num_heads=16, 
        clip_embed_dim=768,
        tubelet_size=1,
        drop_path_rate=0.3,
        num_frames=num_frames),
    cls_head=dict(
        type='TimeSformerHead',
        dropout_ratio=0.4,
        num_classes=52,
        in_channels=768,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/root/autodl-tmp/train'
data_root_val = '/root/autodl-tmp/val'
data_root_test = '/root/autodl-tmp/val'
ann_file_train = '/root/autodl-tmp/annotations/train_list_videos.txt'
ann_file_val = '/root/autodl-tmp/annotations/val_list_videos.txt'
ann_file_test = '/root/autodl-tmp/annotations/val_list_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=1,
        test_mode=True),
    # dict(type='SampleFrames', clip_len=num_frames, frame_interval=4, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop', area_range=(0.3, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='ColorJitter'),
    dict(type='RandomErasing', max_area_ratio=0.2),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=4,
        test_mode=True),
    # dict(type='SampleFrames', clip_len=num_frames, frame_interval=4, num_clips=4, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=4, test_mode=True),
    # dict(type='SampleFrames', clip_len=num_frames, frame_interval=4, num_clips=2, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
#     dict(type='Resize', scale=(-1, 224)),
#     dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_test),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=36, val_begin=16, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1)))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2.0,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=38,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=38)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=2), logger=dict(interval=100))
auto_scale_lr = dict(enable=False, base_batch_size=8)
work_dir = '/root/autodl-tmp/output/intervideoV2'
load_from = '/root/autodl-tmp/pretrained/1B_ft_ssv2_f8_mmversion.pth'


