# Foundation Model - InterVideo2 (RGB single modal)


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
    - refer to 'mmaction2/configs/recognition/config_InterVideo2.py'
    -  16 frames (align to pretrained model)
    - dict(type='UniformSample', clip_len=num_frames, num_clips=1)
    - Other conventional data augmentation are consistent with the official mmaction


## Backbone Version   
### 1B (Large)
- Pretrained
    - [K700](https://huggingface.co/OpenGVLab/InternVideo2-Stage1-1B-224p-K700/blob/main/1B_ft_k710_ft_k700_f16.pth)
    
## Head 
### TimeSformerHead
- if use TimeSformerHead
    replace vit_mae.py to vit_mae_origin.py
        - 'mmaction2/mmaction/models/backbones/vit_mae_origin.py'
### Loss
- CrossEntropyLoss
    - default use with TimeSformerHead

## Run
### Training
```shell
cd mmaction2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 tools/train.py configs/recognition/config_InterVideo2.py --amp --seed=0 --launcher pytorch
```
### Testing
```shell
cd mmaction2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 tools/test.py configs/recognition/config_InterVideo2.py {checkpoint_path} --dump submit/test_result.pickle  --launcher pytorch
```


## Results
| Model | Test F1_mean (TrainVal) |Test F1_mean (Train) | Val F1_mean | ckpt | pickle |
| :-: | :-: | :-: | :-: | :-: | :-: |
| InterVideo2 | | 69.87 | 71.02 | ModelGroup-xl  | intervideo_baseline_val_7102_fortest699.pickle |
