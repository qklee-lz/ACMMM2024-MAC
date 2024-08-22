# Foundation Model - VideoSwinTransformer (RGB single modal)


## Data Processing
- Use ours Croped Data
    - use Yolo/OpenCV/…… to detect people
    - use post-processing to keep crop area w/h > 0.6
    - Modify the transformation code (class CenterCrop(RandomCrop)) in the official mmaction
        - 'mmaction2/mmaction/datasets/transforms/processing.py'
        - origin official class have annotated
    - See the attached for the prepared data
        - train_val_list_videos.txt
            - Merge train_list_vidoes.txt & val_list_videos.txt
    
-  Augmentation
    - refer to 'mmaction2/configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-40e_ma52-rgb.py'
    -  32 frames (align to pretrained model)
    - dict(type='UniformSample', clip_len=num_frames, num_clips=1)
    - Other conventional data augmentation are consistent with the official mmaction


## Backbone Version
### Base（Main Used）
- Pretrained
    - [SSV2](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth)
         
### Large
- Pretrained
    - [K700](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb_20220930-f8d74db7.pth)
    
## Head 
### I3D 

### Proposed Assisting Head    
- Each Frame Prediction Assisting Head
    - fine action 52 classifcation head
    - default use (cls_head params in config)
        - loss_cls=dict(type='CrossEntropyF1LossDouble')
        - base.py (mmaction/models/heads/base-one-assisting-head.py)
        
### Desigen Loss  
- ArcFaceLoss
    - if use (cls_head params in config)
        - use_arcface=True
        -  default use
        
- CombinedF1Loss + AssistingHeadLoss (CrossEntropyF1LossDouble)
    -  default use with Frame assisting head 

- CrossEntropyF1Loss
    - CrossEntropy + CombinedF1Loss
    - default use with no assisting head
    
## Run
### Training
```shell
cd mmaction2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 tools/train.py configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-40e_ma52-rgb.py --amp --seed=0 --launcher pytorch
```
### Testing
```shell
cd mmaction2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 tools/test.py configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-40e_ma52-rgb.py {checkpoint_path} --dump submit/test_result.pickle  --launcher pytorch
```


## Results
| Model | Test F1_mean (TrainVal) |Test F1_mean (Train) | Val F1_mean | ckpt | pickle |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Swin-Base-SSV2-Frame-Assisting-I3D-Head |71.54| 69.48 | 70.84 | ModelGroup3  | mae_crop_alltrick_addval_epoch42_untest.pickle |
| Swin-Large-K700-Frame-Assisting-I3D-Head |71.01| 66.51 | 69.22 |ModelGroup4| mae_crop_alltrick_coarsehead_i3d_addval_epoch39_untest.pickle |
