# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmengine.model import BaseDataPreprocessor, stack_batch

from mmaction.registry import MODELS
from mmaction.utils import SampleList


@MODELS.register_module()
class ActionDataPreprocessor(BaseDataPreprocessor):
    """Data pre-processor for action recognition tasks.

    Args:
        mean (Sequence[float or int], optional): The pixel mean of channels
            of images or stacked optical flow. Defaults to None.
        std (Sequence[float or int], optional): The pixel standard deviation
            of channels of images or stacked optical flow. Defaults to None.
        to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        to_float32 (bool): Whether to convert data to float32.
            Defaults to True.
        blending (dict, optional): Config for batch blending.
            Defaults to None.
        format_shape (str): Format shape of input data.
            Defaults to ``'NCHW'``.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 mean_flow: Optional[Sequence[Union[float, int]]] = None,
                 std_flow: Optional[Sequence[Union[float, int]]] = None,
                 to_rgb: bool = False,
                 to_float32: bool = True,
                 blending: Optional[dict] = None,
                 format_shape: str = 'NCHW') -> None:
        super().__init__()
        self.to_rgb = to_rgb
        self.to_float32 = to_float32
        self.format_shape = format_shape

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            if self.format_shape == 'NCHW':
                normalizer_shape = (-1, 1, 1)
            elif self.format_shape in ['NCTHW', 'MIX2d3d']:
                normalizer_shape = (-1, 1, 1, 1)
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

            self.register_buffer(
                'mean',
                torch.tensor(mean, dtype=torch.float32).view(normalizer_shape),
                False)
            self.register_buffer(
                'std',
                torch.tensor(std, dtype=torch.float32).view(normalizer_shape),
                False)
            self.register_buffer(
                'mean_flow',
                torch.tensor(mean_flow, dtype=torch.float32).view(normalizer_shape),
                False)
            self.register_buffer(
                'std_flow',
                torch.tensor(std_flow, dtype=torch.float32).view(normalizer_shape),
                False)
        else:
            self._enable_normalize = False

        if blending is not None:
            self.blending = MODELS.build(blending)
        else:
            self.blending = None

    def forward(self,
                data: Union[dict, Tuple[dict]],
                training: bool = False) -> Union[dict, Tuple[dict]]:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict or Tuple[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict or Tuple[dict]: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        if isinstance(data, dict):
            return self.forward_onesample(data, training=training)
        elif isinstance(data, (tuple, list)):
            outputs = []
            for data_sample in data:
                output = self.forward_onesample(data_sample, training=training)
                outputs.append(output)
            return tuple(outputs)
        else:
            raise TypeError(f'Unsupported data type: {type(data)}!')

    def forward_onesample(self, data, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation on one data sample.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        # inputs, data_samples = data['inputs'], data['data_samples']
        # print(inputs)
        # print(type(inputs))
        # print(len(inputs))
        # print(inputs[0].shape)
        # input("?")
        # inputs, data_samples = self.preprocess(inputs, data_samples, training)
        # data['inputs'] = inputs
        
        inputs_rgb, inputs_flow, data_samples = data['inputs'], data['inputs_flow'], data['data_samples']
        inputs_rgb, data_samples = self.preprocess(inputs_rgb, data_samples, 'rgb', training)
        inputs_flow, _ = self.preprocess(inputs_flow, data_samples, 'flow', training)
        # print(inputs_rgb.mean(), inputs_rgb.std(), inputs_flow.mean(), inputs_flow.std())

        data["inputs"] = [inputs_rgb, inputs_flow]
        data['data_samples'] = data_samples
        return data

    def preprocess(self,
                   inputs: List[torch.Tensor],
                   data_samples: SampleList,
                   norm_data: str,
                   training: bool = False) -> Tuple:
        # --- Pad and stack --
        batch_inputs = stack_batch(inputs)

        if self.format_shape == 'MIX2d3d':
            if batch_inputs.ndim == 4:
                format_shape, view_shape = 'NCHW', (-1, 1, 1)
            else:
                format_shape, view_shape = 'NCTHW', None
        else:
            format_shape, view_shape = self.format_shape, None

        # ------ To RGB ------
        if self.to_rgb:
            if format_shape == 'NCHW':
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :]
            elif format_shape == 'NCTHW':
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :, :]
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

        # -- Normalization ---
        if norm_data == 'rgb':
            mean, std = self.mean, self.std
        else:
            mean, std = self.mean_flow, self.std_flow
        if self._enable_normalize:
            if view_shape is None:
                batch_inputs = (batch_inputs - mean) / std
            else:
                mean = mean.view(view_shape)
                std = std.view(view_shape)
                batch_inputs = (batch_inputs - mean) / std
        elif self.to_float32:
            batch_inputs = batch_inputs.to(torch.float32)

        # ----- Blending -----
        if training and self.blending is not None:
            batch_inputs, data_samples = self.blending(batch_inputs,
                                                       data_samples)

        return batch_inputs, data_samples
