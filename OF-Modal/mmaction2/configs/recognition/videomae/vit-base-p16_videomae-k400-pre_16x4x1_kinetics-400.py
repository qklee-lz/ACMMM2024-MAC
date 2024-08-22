_base_ = ['../../_base_/default_runtime.py']

# model settings
num_frames = 16
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='VisionTransformerTwoStream',
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=num_frames,
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='I3DHead',
        in_channels=768*2,
        num_classes=52,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob',
        loss_cls=dict(type='CrossEntropyF1LossDouble'),
        use_coarse=False,
        use_arcface=True
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        mean_flow=[0.95506871, 1.02307589, 1.04017459],
        std_flow=[7.47380747, 8.11153283, 8.16956986],         
        format_shape='NCTHW'))

# dataset settings
# all_clip train val new_train_list_videos train_list_videos train_val
dataset_type = 'VideoDataset'
data_root = '/root/data/train'
data_root_val = '/root/data/train'
data_root_test = '/root/data/train'
ann_file_train = '/root/data/annotations/train_val.txt'
ann_file_val = '/root/data/annotations/val_list_videos.txt'
ann_file_test = '/root/data/annotations/test_list_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=1),
    dict(type='DecordDecode'),
    # dict(type='RandomResizedCrop', area_range=(0.5, 1.0), aspect_ratio_range=(1/3, 1.0)),
    
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop', area_range=(0.3, 1.0)),
    
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
    
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=(373, 224)),
    # dict(type='Resize', scale=(224, 224), keep_ratio=False),
    
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=2, test_mode=True),
    dict(type='DecordDecode'),
    
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=(373, 224)),
    # dict(type='Resize', scale=(224, 224), keep_ratio=False),
    
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    
    # dict(type='Resize', scale=(224, -1)),
    # dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_pipeline_flow = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=1),
    # dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop', area_range=(0.6, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='ColorJitter'),
    dict(type='RandomErasing', max_area_ratio=0.2),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline_flow = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=2, test_mode=True),
    # dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=2, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline_flow = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=2, test_mode=True),
    # dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=2, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    # dict(type='Resize', scale=(-1, 224)),
    # dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline,
        pipeline_flow=train_pipeline_flow))
val_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        pipeline_flow=val_pipeline_flow,
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
        pipeline_flow=test_pipeline_flow,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=42, val_begin=2, val_interval=2)
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
        T_max=43,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=43)
]

default_hooks = dict(
    checkpoint=dict(interval=6, max_keep_ckpts=2), logger=dict(interval=100))
auto_scale_lr = dict(enable=False, base_batch_size=8)
work_dir = '/root/autodl-tmp/output/rgb_flow'

load_from = '/root/autodl-tmp/pretrained/vit_b_hybrid_pt_800e_k710_ft_mmv_rgbk710_flowava.pth'





