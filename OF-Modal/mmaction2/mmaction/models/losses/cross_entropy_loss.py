# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


from mmaction.registry import MODELS
from .base import BaseWeightedLoss


@MODELS.register_module()
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probability distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Defaults to None.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label

            if self.class_weight is not None:
                assert 'weight' not in kwargs, \
                    "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)

        return loss_cls


@MODELS.register_module()
class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Defaults to None.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        if self.class_weight is not None:
            assert 'weight' not in kwargs, "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight.to(cls_score.device)
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label,
                                                      **kwargs)
        return loss_cls


@MODELS.register_module()
class CBFocalLoss(BaseWeightedLoss):
    """Class Balanced Focal Loss. Adapted from https://github.com/abhinanda-
    punnakkal/BABEL/. This loss is used in the skeleton-based action
    recognition baseline for BABEL.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        samples_per_cls (list[int]): The number of samples per class.
            Defaults to [].
        beta (float): Hyperparameter that controls the per class loss weight.
            Defaults to 0.9999.
        gamma (float): Hyperparameter of the focal loss. Defaults to 2.0.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 samples_per_cls: List[int] = [],
                 beta: float = 0.9999,
                 gamma: float = 2.) -> None:
        super().__init__(loss_weight=loss_weight)
        self.samples_per_cls = samples_per_cls
        self.beta = beta
        self.gamma = gamma
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        self.weights = weights
        self.num_classes = len(weights)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        weights = torch.tensor(self.weights).float().to(cls_score.device)
        label_one_hot = F.one_hot(label, self.num_classes).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(label_one_hot.shape[0], 1) * label_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.num_classes)

        BCELoss = F.binary_cross_entropy_with_logits(
            input=cls_score, target=label_one_hot, reduction='none')

        modulator = 1.0
        if self.gamma:
            modulator = torch.exp(-self.gamma * label_one_hot * cls_score -
                                  self.gamma *
                                  torch.log(1 + torch.exp(-1.0 * cls_score)))

        loss = modulator * BCELoss
        weighted_loss = weights * loss

        focal_loss = torch.sum(weighted_loss)
        focal_loss /= torch.sum(label_one_hot)

        return focal_loss




class CombinedF1Loss(nn.Module):
    '''
    Macro & Micro F1Loss (only Fine Action 52 classes)
    '''
    def __init__(self, epsilon=1e-7, num_classes=52):
        super(CombinedF1Loss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).float()

        # Macro F1 -> tp, fp, fn
        tp = (inputs * targets_one_hot).sum(dim=0)
        fp = ((1 - targets_one_hot) * inputs).sum(dim=0)
        fn = (targets_one_hot * (1 - inputs)).sum(dim=0)

        precision_per_class = tp / (tp + fp + self.epsilon)
        recall_per_class = tp / (tp + fn + self.epsilon)

        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + self.epsilon)
        
        # Macro F1 Loss
        macro_f1_loss = 1 - f1_per_class.mean()

        # Micro F1 -> tp, fp, fn
        tp_total = tp.sum()
        fp_total = fp.sum()
        fn_total = fn.sum()

        precision_total = tp_total / (tp_total + fp_total + self.epsilon)
        recall_total = tp_total / (tp_total + fn_total + self.epsilon)

        # Micro F1 Loss
        micro_f1 = 2 * (precision_total * recall_total) / (precision_total + recall_total + self.epsilon)
        
        # Micro F1 Loss
        micro_f1_loss = 1 - micro_f1

        return macro_f1_loss + micro_f1_loss

# class CombinedF1Loss(nn.Module):
    '''
    Macro & Micro F1Loss (Fine Action 52 classes + Coarse Body 7 Classes)
    '''
#     def __init__(self, epsilon=1e-7, num_classes=52, coarse_classes=7):
#         super(CombinedF1Loss, self).__init__()
#         self.epsilon = epsilon
#         self.num_classes = num_classes
#         self.coarse_classes = coarse_classes

#     def forward(self, inputs, targets):
#         inputs = torch.softmax(inputs, dim=1)  # [4,52]
#         targets_one_hot = F.one_hot(targets, self.num_classes).float()

#         tp = (inputs * targets_one_hot).sum(dim=0)
#         fp = ((1 - targets_one_hot) * inputs).sum(dim=0)
#         fn = (targets_one_hot * (1 - inputs)).sum(dim=0)

#         precision_per_class = tp / (tp + fp + self.epsilon)
#         recall_per_class = tp / (tp + fn + self.epsilon)

#         f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + self.epsilon)
        
#         # Macro F1 Loss
#         macro_f1_loss = 1 - f1_per_class.mean()

#         tp_total = tp.sum()
#         fp_total = fp.sum()
#         fn_total = fn.sum()

#         precision_total = tp_total / (tp_total + fp_total + self.epsilon)
#         recall_total = tp_total / (tp_total + fn_total + self.epsilon)

#         micro_f1 = 2 * (precision_total * recall_total) / (precision_total + recall_total + self.epsilon)
        
#         # Micro F1 Loss
#         micro_f1_loss = 1 - micro_f1

#         coarse_targets = fine2coarse(targets)
#         coarse_inputs = fine2coarse(inputs.argmax(dim=1))

#         coarse_targets_one_hot = F.one_hot(coarse_targets, self.coarse_classes).float()
#         coarse_inputs_one_hot = F.one_hot(coarse_inputs, self.coarse_classes).float()

#         coarse_tp = (coarse_inputs_one_hot * coarse_targets_one_hot).sum(dim=0)
#         coarse_fp = ((1 - coarse_targets_one_hot) * coarse_inputs_one_hot).sum(dim=0)
#         coarse_fn = (coarse_targets_one_hot * (1 - coarse_inputs_one_hot)).sum(dim=0)

#         coarse_precision_per_class = coarse_tp / (coarse_tp + coarse_fp + self.epsilon)
#         coarse_recall_per_class = coarse_tp / (coarse_tp + coarse_fn + self.epsilon)

#         coarse_f1_per_class = 2 * (coarse_precision_per_class * coarse_recall_per_class) / (coarse_precision_per_class + coarse_recall_per_class + self.epsilon)
        
#         coarse_macro_f1_loss = 1 - coarse_f1_per_class.mean()

#         coarse_tp_total = coarse_tp.sum()
#         coarse_fp_total = coarse_fp.sum()
#         coarse_fn_total = coarse_fn.sum()

#         coarse_precision_total = coarse_tp_total / (coarse_tp_total + coarse_fp_total + self.epsilon)
#         coarse_recall_total = coarse_tp_total / (coarse_tp_total + coarse_fn_total + self.epsilon)

#         coarse_micro_f1 = 2 * (coarse_precision_total * coarse_recall_total) / (coarse_precision_total + coarse_recall_total + self.epsilon)
        
#         coarse_micro_f1_loss = 1 - coarse_micro_f1

#         return macro_f1_loss + micro_f1_loss + coarse_macro_f1_loss + coarse_micro_f1_loss

    
@MODELS.register_module()
class CBFocalF1Loss(BaseWeightedLoss):
    """Class Balanced Focal Loss. Adapted from https://github.com/abhinanda-
    punnakkal/BABEL/. This loss is used in the skeleton-based action
    recognition baseline for BABEL.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        samples_per_cls (list[int]): The number of samples per class.
            Defaults to [].
        beta (float): Hyperparameter that controls the per class loss weight.
            Defaults to 0.9999.
        gamma (float): Hyperparameter of the focal loss. Defaults to 2.0.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 samples_per_cls: List[int] = [],
                 beta: float = 0.9999,
                 gamma: float = 2.) -> None:
        super().__init__(loss_weight=loss_weight)
        self.samples_per_cls = samples_per_cls
        self.beta = beta
        self.gamma = gamma
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        self.weights = weights
        self.num_classes = len(weights)

        self.CombinedF1Loss = CombinedF1Loss()

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        MaMiF1_loss = self.CombinedF1Loss(cls_score, label)

        weights = torch.tensor(self.weights).float().to(cls_score.device)
        label_one_hot = F.one_hot(label, self.num_classes).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(label_one_hot.shape[0], 1) * label_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.num_classes)

        BCELoss = F.binary_cross_entropy_with_logits(
            input=cls_score, target=label_one_hot, reduction='none')

        modulator = 1.0
        if self.gamma:
            modulator = torch.exp(-self.gamma * label_one_hot * cls_score -
                                  self.gamma *
                                  torch.log(1 + torch.exp(-1.0 * cls_score)))

        loss = modulator * BCELoss
        weighted_loss = weights * loss

        focal_loss = torch.sum(weighted_loss)
        focal_loss /= torch.sum(label_one_hot)

        return focal_loss + 0.5*MaMiF1_loss


@MODELS.register_module()
class CrossEntropyF1Loss(BaseWeightedLoss):
    """
    Cross Entropy Loss. + MacroMicroF1Loss (only fine 52 action class)
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 num_classes = 52,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)
        
        self.CombinedF1Loss = CombinedF1Loss(num_classes=num_classes)


    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        MaMiF1_loss = self.CombinedF1Loss(cls_score, label)

        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label

            if self.class_weight is not None:
                assert 'weight' not in kwargs, \
                    "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)

        return loss_cls + MaMiF1_loss

@MODELS.register_module()
class CrossEntropyF1LossFineACoarseDouble(BaseWeightedLoss):
    '''
    For add Coarse Assisting Head + Each EachFrameLeveLPrediction Assisting Head
    '''
    def __init__(self,
                 loss_weight: float = 1.0,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.CrossEntropy_F1Loss_Fine = CrossEntropyF1LossDouble(num_classes=52)
        self.CrossEntropy_F1Loss_Coarse = CrossEntropyF1LossDouble(num_classes=7)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        loss_fine = self.CrossEntropy_F1Loss_Fine(cls_score[0], label)
        loss_coarse = self.CrossEntropy_F1Loss_Coarse(cls_score[1], fine2coarse(label))
        return loss_fine + 0.5*loss_coarse

@MODELS.register_module()
class CrossEntropyF1LossDouble(BaseWeightedLoss):
    '''
    For EachFrameLeveLPrediction Assisting Head
    '''
    def __init__(self,
                 num_classes=52,
                 loss_weight: float = 1.0,
                 k = 3,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.num_classes = num_classes
        self.CrossEntropy_F1Loss = CrossEntropyF1Loss(num_classes=self.num_classes)
        self.k = k

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        loss_lead = self.CrossEntropy_F1Loss(cls_score[0], label)
        loss_frame = self.CrossEntropy_F1Loss(cls_score[1].view(-1,self.num_classes), label.unsqueeze(1).repeat(1, cls_score[1].shape[1]).view(-1))
        
  # TopK     
        # top_scores, top_indices = torch.topk(cls_score[1], self.k, dim=1, largest=True)  # get topK score K frame and index
        # top_scores = top_scores.view(-1, 52)
        # loss_frame = self.CrossEntropy_F1Loss(top_scores.view(-1, 52), label.unsqueeze(1).repeat(1, self.k).view(-1))
        return loss_lead + 0.5*loss_frame
    

    
def fine2coarse(labels):
    conditions = [
        (labels <= 4),
        (labels >= 5) & (labels <= 10),
        (labels >= 11) & (labels <= 23),
        (labels >= 24) & (labels <= 31),
        (labels >= 32) & (labels <= 37),
        (labels >= 38) & (labels <= 47),
        (labels >= 48)
    ]
    
    values = torch.tensor([0, 1, 2, 3, 4, 5, 6], device=labels.device)
    
    result = torch.zeros_like(labels)
    
    for cond, val in zip(conditions, values):
        result = torch.where(cond, val, result)
    
    return result