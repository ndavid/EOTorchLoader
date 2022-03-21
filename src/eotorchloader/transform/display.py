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
from typing import Any, Callable, Dict, List, Sequence, Tuple, Optional
import numpy as np
from .base import BasicTransform


class ToRgbDisplay(BasicTransform):
    """
    Convert (image, mask) sample into compatible RGB format for display purpose

    For the image choose the channel to use as Red, Green, Blue for display, for the mask
    data an argmax flattening is apply first if needed and then a lut conversion is made

    Note:
        Input (image, mask) should already be in HWC order and not CHW. To convert
        from one to another format see ..

    Args:
        color_compo (str) : Name or color composition to use, if use then image channels should
          be ordered by increasing spectral wavelenght (Blue, Green , Red, Infra-Red)
        channels_display (List(Int)) : List of channel to use as gray or Red, Green, Blue band
          if len(channels_display)=1 is for gray mode and len=3 for RGB mode
        lut (np.array) : numpy array to convert class number to rgb value. If shape is (N, 3) then
          the array index is use class number if shape is (N, 4) then each is ( Class_Id, R, V, B)
        flatten_mask (bool) : use an argmax function for mask data before lut conversion.

    """

    def __init__(
        self,
        color_compo: str = None,
        channels_display: List[int] = None,
        lut: np.array = None,
        flatten_mask: bool = True,
    ):
        self.flatten = flatten_mask
        super(ToRgbDisplay, self).__init__()
        if color_compo is None and channels_display is None:
            # set rgb as first three channel
            self.color_compo = "rgb"
            self.channels_display = [0, 1, 2]

        if lut is not None:
            row_size = lut.shape[1]
            if row_size == 3:
                self.lut = lut
            elif row_size == 4:
                lut_255 = np.zeros((256, 3), dtype=np.uint8)
                lut_255[lut[:, 0]] = lut[:, 1:4]
                self.lut = lut_255
            else:
                raise ValueError(
                    f"lut shoulb be of shape [N,3) or (N,4) and not {lut.shape}"
                )
        else:
            self.lut = None

        if len(self.channels_display) != 1 and len(self.channels_display) != 3:
            raise ValueError("number of channel to display should be 1 or 3 ")

    def apply_to_img(self, img: np.ndarray) -> np.ndarray:
        img = img[:, :, self.channels_display]
        return img

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.flatten:
            mask = np.argmax(mask, axis=0)
        if not self.flatten and mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0)
        if self.lut is not None:
            # mask = self.lut[mask]
            mask = np.take(self.lut, mask, axis=0)
        return mask
