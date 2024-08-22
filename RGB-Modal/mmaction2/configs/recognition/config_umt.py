_base_ = ['../_base_/default_runtime.py']

# model settings
num_frames = 16
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='UMT',
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        all_frames=num_frames,),
    cls_head=dict(
        type='TimeSformerHead',
        dropout_ratio=0.5,
        num_classes=52,
        in_channels=1024,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # mean=[139.53047424, 141.15400055, 133.45072513],
        # std=[37.9471134, 46.12410568, 43.78503414],
         
        format_shape='NCTHW'))

# dataset settings
# all_clip train val new_train_list_videos train_list_videos train_val
dataset_type = 'VideoDataset'
data_root = '/root/autodl-tmp/all_clip'
data_root_val = '/root/autodl-tmp/all_clip'
data_root_test = '/root/autodl-tmp/all_clip'
ann_file_train = '/root/autodl-tmp/annotations/train_val.txt'
ann_file_val = '/root/autodl-tmp/annotations/val_list_videos.txt'
ann_file_test = '/root/autodl-tmp/annotations/test_list_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='RandomResizedCrop', area_range=(0.5, 1.0), aspect_ratio_range=(1/3, 1.0)),
    
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='RandomResizedCrop', area_range=(0.3, 1.0)),
    
    # dict(
    #     type='PytorchVideoWrapper',
    #     op='RandAugment',
    #     magnitude=7,
    #     num_layers=4),
    
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
    
    dict(type='Resize', scale=(-1, 238)),
    dict(type='CenterCrop', crop_size=(373, 224)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224),
    
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=2, test_mode=True),
    dict(type='DecordDecode'),
    
    dict(type='Resize', scale=(-1, 238)),
    dict(type='CenterCrop', crop_size=(373, 224)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224),
    
    # dict(type='Resize', scale=(224, -1)),
    # dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=10,
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
    type='EpochBasedTrainLoop', max_epochs=38, val_begin=36, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-2),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1)))

# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     optimizer=dict(type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01),
#     constructor='LearningRateDecayOptimizerConstructor',
#     paramwise_cfg={
#         'decay_rate': 0.75,
#         'decay_type': 'layer_wise',
#         'num_layers': 12,
#         'backbone': dict(lr_mult=0.1)
#     },
#     clip_grad=dict(max_norm=20, norm_type=2))

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
        T_max=39,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=39)
]

default_hooks = dict(
    checkpoint=dict(interval=6, max_keep_ckpts=1), logger=dict(interval=100))
auto_scale_lr = dict(enable=False, base_batch_size=8)

work_dir = '/root/autodl-tmp/output/umt_large_k710_trainval_crop'
load_from = '/root/autodl-tmp/pretrained/l16_ptk710_ftk710_ftk700_f16_res224_mmv.pth'



