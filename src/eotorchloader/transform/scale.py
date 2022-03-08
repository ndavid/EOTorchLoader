# Copyright The PyTorch Lightning team.
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


class ScaleImageToFloat(BasicTransform):
    """
    scale an input image to float image between [0, 1]
    """

    def __init__(self, scale_factor: float = 255, clip: bool = False):
        super(ScaleImageToFloat, self).__init__()
        self.img_only = True
        self.scale_factor = scale_factor
        self.clip = clip

    def apply_to_img(self, img: np.ndarray) -> np.ndarray:
        img = np.multiply(img, 1.0 / self.scale_factor, dtype=np.float32)
        if self.clip:
            return np.clip(img, 0, 1)
        else:
            return img


class FloatImageToByte(BasicTransform):
    """
    scale an input image from [0-1] to [0-255] mainly ofr rgb display purpose
    """

    def __init__(self, clip: bool = False):
        super(FloatImageToByte, self).__init__()
        self.img_only = True
        self.scale_factor = 255
        self.clip = clip

    def apply_to_img(self, img: np.ndarray) -> np.ndarray:
        img = np.multiply(img, self.scale_factor, dtype=np.float32)
        img = img.astype(np.uint8)
        if self.clip:
            return np.clip(img, 0, 255)
        else:
            return img
