# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model.weight_init import normal_init
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, get_str_type
from .base import AvgConsensus, BaseHead


import torch.nn.functional as F
import torch

@MODELS.register_module()
class TSNHead(BaseHead):
    """Class head for TSN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str or ConfigDict): Pooling type in spatial dimension.
            Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 spatial_type: str = 'avg',
                 consensus: ConfigType = dict(type='AvgConsensus', dim=1),
                 dropout_ratio: float = 0.4,
                 init_std: float = 0.01,
                 use_arcface=False,
                 use_coarse=False,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if get_str_type(consensus_type) == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        
        # arcface
        self.use_arcface = use_arcface
        if self.use_arcface:
            self.fc_cls = ArcFaceLoss(self.in_channels, self.num_classes)
        else: 
            self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
            
        # coarse assisting head
        self.use_coarse = use_coarse
        if self.use_coarse:
            self.fc_coarse_cls = ArcFaceLoss(self.in_channels, 7)
            # each frame prediction assisting coarse head (num_classes=7) 
            self.fc_coarse_cls_frame = FrameLevelPrediction(self.in_channels, 7)
        
        # each frame prediction assisting fine head (num_classes=52) 
        self.fc_frame = FrameLevelPrediction(self.in_channels, self.num_classes)
        
    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""

    def forward(self, x: Tensor, num_segs=8, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            Tensor: The classification scores for input samples.
        """
        frames_cls_score = self.fc_frame(x)
        if self.use_coarse:
            frames_cls_coarse_score = self.fc_coarse_cls_frame(x)
            
        x = x.permute(0, 2, 1, 3, 4)
        x = x.view(-1,x.shape[2],x.shape[3],x.shape[4])
    
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
            x = self.avg_pool(x)
            # [N * num_segs, in_channels, 1, 1]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N, num_segs, in_channels, 1, 1]
        x = self.consensus(x)
        # [N, 1, in_channels, 1, 1]
        x = x.squeeze(1)
        # [N, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N, in_channels, 1, 1]
        x = x.view(x.size(0), -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        if self.use_coarse:
            cls_coarse_score = self.fc_coarse_cls(x)
            return (cls_score, frames_cls_score), (cls_coarse_score, frames_cls_coarse_score)
        # [N, num_classes]
        return cls_score, frames_cls_score


class ArcFaceLoss(nn.Module):
    def __init__(self, feature_dim, num_classes, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.fc = nn.Linear(feature_dim, num_classes, bias=False)
        normal_init(self.fc, std=0.01)

    def forward(self, input_features):
        # Normalize the weight
        normalized_weight = F.normalize(self.fc.weight, p=2, dim=1)
        
        # Normalize the input features
        normalized_features = F.normalize(input_features, p=2, dim=1)
        
        # Calculate the cosine of the angles
        cosine = F.linear(normalized_features, normalized_weight)
        
        # Add margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        margin_cosine = torch.cos(theta + self.m)
        
        # Apply the scale: s * (cos(theta + m))
        scaled_margin_cosine = self.s * margin_cosine
        
        return scaled_margin_cosine
    
    
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)
   
    
class FrameLevelPrediction(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FrameLevelPrediction, self).__init__()
        self.se = SEBlock(input_dim)
        self.fc = nn.Linear(input_dim, num_classes)
        normal_init(self.fc, std=0.01)
    
    def forward(self, x):
        # [N, 768, 16, 7, 7] or [16, 768, 8, 14, 14]
        N, C, T, H, W = x.size()
        x = x.reshape(N * T, C, H, W)  
        x = torch.mean(x, dim=[2, 3]) 
        x = self.se(x)
        x = self.fc(x)  
        return x.view(N, T, -1)  