# Copyright Nicolas DAVID.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
from .base import BasicTransform


class HWC_to_CHW(BasicTransform):
    def __init__(self, img_only: bool = False, mask_only: bool = False):
        super(CHW_to_HWC, self).__init__()
        self.img_only = img_only
        self.mask_only = mask_only

    @staticmethod
    def swap_axes(array: np.ndarray) -> np.ndarray:
        # swap the axes order from (rows, columns, bands) to (band, rows, columns)
        array = np.ma.transpose(array, [2, 0, 1])
        return array

    def apply_to_img(self, img: np.ndarray) -> np.ndarray:
        return HWC_to_CHW.swap_axes(img)

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        return HWC_to_CHW.swap_axes(mask)


class CHW_to_HWC(BasicTransform):
    def __init__(self, img_only: bool = False, mask_only: bool = False):
        super(CHW_to_HWC, self).__init__()
        self.img_only = img_only
        self.mask_only = mask_only

    @staticmethod
    def swap_axes(array: np.ndarray) -> np.ndarray:
        # swap the axes order from (bands, rows, columns) to (rows, columns, bands)
        array = np.ma.transpose(array, [1, 2, 0])
        return array

    def apply_to_img(self, img: np.ndarray) -> np.ndarray:
        return CHW_to_HWC.swap_axes(img)

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        return CHW_to_HWC.swap_axes(mask)


class ToTorchTensor(BasicTransform):
    def __init__(self, img_only: bool = False, mask_only: bool = False):
        super(ToTorchTensor, self).__init__()
        self.img_only = img_only
        self.mask_only = mask_only

    def apply_to_img(self, img: np.ndarray):
        return torch.from_numpy(img).type(torch.float32)

    def apply_to_mask(self, mask: np.ndarray):
        return torch.from_numpy(mask).type(torch.float32)


class TensorToArray(BasicTransform):
    def __init__(self, img_only: bool = False, mask_only: bool = False):
        super(TensorToArray, self).__init__()
        self.img_only = img_only
        self.mask_only = mask_only

    def apply_to_img(self, img) -> np.ndarray:
        return img.cpu().numpy()

    def apply_to_mask(self, mask) -> np.ndarray:
        return mask.cpu().numpy()
