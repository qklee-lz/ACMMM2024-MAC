_base_ = [
    '../../_base_/models/vitclip_base.py', '../../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(patch_size=14, width=1024, layers=24, heads=16, drop_path_rate=0.2, adapter_scale=0.5, num_frames=32,
                  pretrained=None, weight_path = '/root/autodl-tmp/aim_pretrained/vit_L_clip_32frame_k400.pth'),
    cls_head=dict(in_channels=1024, num_classes=52))

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
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
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

# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.02),
#     constructor='SwinOptimWrapperConstructor',
#     paramwise_cfg=dict(
#         absolute_pos_embed=dict(decay_mult=0.),
#         relative_position_bias_table=dict(decay_mult=0.),
#         norm=dict(decay_mult=0.),
#         backbone=dict(lr_mult=0.1)))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=1e-3),
    constructor='SwinOptimWrapperConstructor',
#     paramwise_cfg=dict(
#         absolute_pos_embed=dict(decay_mult=0.),
#         relative_position_bias_table=dict(decay_mult=0.),
#         norm=dict(decay_mult=0.),
#         backbone=dict(lr_mult=1.0)))
    paramwise_cfg=dict(custom_keys={'class_embedding': dict(decay_mult=0.),
                                                 'positional_embedding': dict(decay_mult=0.),
                                                 'ln_1': dict(decay_mult=0.),
                                                 'ln_2': dict(decay_mult=0.),
                                                 'ln_pre': dict(decay_mult=0.),
                                                 'ln_post': dict(decay_mult=0.),}))


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
        T_max=40,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=40)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
work_dir = '/root/autodl-tmp//aim_large'