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
from .base import BasicTransform


class ToRgbDisplay(BasicTransform):
    """
    scale an input image to float image between [0, 1]
    """

    def __init__(
        self,
        color_compo: str = None,
        channels_display: List[int] = None,
        lut: np.array = None,
    ):
        """

        channels_display : list of channel to display in r,g,b order if dim 3, channel dto display if dim 1
        """

        super(ToRgbDisplay, self).__init__()
        if color_compo is None and channels_display is None:
            # set rgb as first three channel
            self.color_compo = "rgb"
            self.channels_display = [0, 1, 2]

        if lut is not None:
            self.lut = lut
        else:
            self.lut = None

        if len(self.channels_display) != 1 and len(self.channels_display) != 3:
            raise ValueError("number of channel to display should be 1 or 3 ")

    def apply_to_img(self, img: np.ndarray) -> np.ndarray:
        img = img[:, :, self.channels_display]
        return img

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = np.argmax(mask, axis=0)
        if self.lut is not None:
            # mask = self.lut[mask]
            mask = np.take(self.lut, mask, axis=0)
        return mask
