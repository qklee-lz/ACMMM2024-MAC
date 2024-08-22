# Foundation Model - VideoMAE (RGB single modal)


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
    - refer to 'mmaction2/configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py'
    -  16 frames (align to pretrained model)
    - dict(type='UniformSample', clip_len=num_frames, num_clips=1)
    - Other conventional data augmentation are consistent with the official mmaction


## Backbone Version
### Base（Main Used）
- Pretrained
    - [K710](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/pretrain/videomae/vit_b_hybrid_pt_800e_k710_ft.pth)
    - [SSV2](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/pretrain/videomae/vit_b_hybrid_pt_800e_ssv2_ft.pth)
         
### Large
- Pretrained
    - [K700](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/pretrain/videomae/vit_l_hybrid_pt_800e_k700_ft.pth)
    
### Huge
- Pretrained
    - [AVA-Kinetics](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/stal/vit_h_hybrid_pt_k710_ft_ak_ft.sh)
    
## Head 
### TimeSformerHead or TSN or I3D 
- if use TimeSformerHead (mmaction official version, not recommended)
    replace vit_mae.py to vit_mae_origin.py
        - 'mmaction2/mmaction/models/backbones/vit_mae_origin.py'

### Proposed Assisting Head    
- Each Frame Prediction Assisting Head
    - fine action 52 classifcation head
    - default use (cls_head params in config)
        - loss_cls=dict(type='CrossEntropyF1LossDouble')
        - base.py (mmaction/models/heads/base-one-assisting-head.py)
        
- Coarse Assisting Head (add based Each Frame Prediction Assisting Head)
    - body 7 classifcation head
    - if use (cls_head params in config)
        - use_coarse=True 
        - loss_cls=dict(type='CrossEntropyF1LossFineACoarseDouble') 
        - base.py (mmaction/models/heads/base-two-assisting-head.py)

### Desigen Loss  
- ArcFaceLoss
    - if use (cls_head params in config)
        - use_arcface=True
        -  default use (except TimeSformerHead)
        
- CombinedF1Loss + AssistingHeadLoss
    -  default use with assisting head (CrossEntropyF1LossDouble/CrossEntropyF1LossFineACoarseDouble)

- CrossEntropyLoss
    - default use with TimeSformerHead

## Run
### Training
```shell
cd mmaction2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 toos/train.py configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py --amp --seed=0 --launcher pytorch
```
### Testing
```shell
cd mmaction2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 tools/test.py configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py {checkpoint_path} --dump submit/test_result.pickle  --launcher pytorch
```


## Results
| Model | Test F1_mean (TrainVal) |Test F1_mean (Train) | Val F1_mean | ckpt | pickle |
| :-: | :-: | :-: | :-: | :-: | :-: |
| VideoMAE-Base-SSV2-TimeSformer-Head |-| 68.94 | 68.76 | ModelGroup-xl  | mae_base_ssv2_train_crop_val_6876_fortest.pickle |
| VideoMAE-Base-K710-TimeSformer-Head |71.94| - | - | ModelGroup-xl  | mae_base_k710_trainval_crop_test_719_fortest.pickle |
| VideoMAE-Base-K710-Frame-Assisting-I3D-Head |72.81| 71.14 | 72.33 | ModelGroup1  | mae_crop_alltrick_addval_epoch42_untest.pickle |
| VideoMAE-Base-K710-Frame-Coarse-Assisting-I3D-Head |73.07| 70.98 | 72.58 |ModelGroup2| mae_crop_alltrick_coarsehead_i3d_addval_epoch39_untest.pickle |
| VideoMAE-Base-K710-Frame-Coarse-Assisting-TSN-Head |**73.13** | 71.18 | 72.7| ModelGroup2| mae_crop_alltrick_coarsehead_tsn_addval_epoch42_untest.pickle |
| VideoMAE-Large-K700-TimeSformer-Head |73.06 | - | -| ModelGroup-xl| mae_large_k700_trainval_crop_fortest.pickle |
| VideoMAE-Huge-TimeSformer-Head |72.68 | - | 71.69| ModelGroup-xl| mae_huge_k700_trainval_crop_val_7169_fortest.pickle |


