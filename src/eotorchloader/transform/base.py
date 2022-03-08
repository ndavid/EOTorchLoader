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


def format_to_dict(image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """ """
    return {"image": image, "mask": mask}


class BasicTransform:
    def __init__(self):
        self.params: Dict[Any, Any] = {}
        self.img_only: bool = False
        self.mask_only: bool = False

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        if self.img_only:
            return {"image": self.apply_to_img(image), "mask": mask}
        elif self.mask_only:
            return {"image": image, "mask": self.apply_to_mask(mask)}
        else:
            return {"image": self.apply_to_img(image), "mask": self.apply_to_mask(mask)}

    def apply_to_img(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError
