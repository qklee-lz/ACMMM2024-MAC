_base_ = [
    '../../_base_/models/swin_tiny.py', '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        arch='small',
        drop_path_rate=0.2,
        pretrained=  # noqa: E251
        '',  # noqa: E501
        pretrained2d=False,
        weight_path = '',
    ))

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/root/autodl-tmp/all_clip'
data_root_val = '/root/autodl-tmp/all_clip'
data_root_test = '/root/autodl-tmp/all_clip'
ann_file_train = '/root/autodl-tmp/annotations/train_list_videos.txt'
ann_file_val = '/root/autodl-tmp/annotations/val_list_videos.txt'
ann_file_test = '/root/autodl-tmp/annotations/val_list_videos.txt'


file_client_args = dict(io_backend='disk')

num_frames = 32
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='RandomResizedCrop', area_range=(0.6, 1.0), aspect_ratio_range=(1/3, 1.0)),
    
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='RandomResizedCrop', area_range=(0.3, 1.0)),
    
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='ColorJitter'),
    dict(type='RandomErasing', max_area_ratio=0.2),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=2, test_mode=True),
    dict(type='DecordDecode'),
    
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=(373, 224)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224),
    
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=3, test_mode=True),
    dict(type='DecordDecode'),
    
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=(373, 224)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224),
    
    # dict(type='Resize', scale=(-1,224)),
    # dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
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
    batch_size=16,
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
    type='EpochBasedTrainLoop', max_epochs=40, val_begin=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05),
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
        end=2.5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=40,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=30)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=6), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=8)

work_dir = '/root/autodl-tmp/output/swin_small_k710_aug_i3ddoubleframe_se_crop'
load_from = "/root/autodl-tmp/pretrained/swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb_match.pth"