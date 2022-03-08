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
import itertools
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Optional

import numpy as np
from torch.utils.data import Dataset, DataLoader
import rasterio
import random

from ..backend.rasterio import geoimage_load_tile
from ..transform.base import format_to_dict


def get_nb_tile_from_img(img_shape: Tuple[int, int], tile_size: int) -> int:
    """ """
    nb_tile_col = img_shape[0] // tile_size
    nb_tile_row = img_shape[1] // tile_size
    return nb_tile_col * nb_tile_row


def get_img_windows_list(img_shape: Tuple[int, int], tile_size: int):
    """ """
    col_step = [col for col in range(0, img_shape[0], tile_size)]
    col_step.append(img_shape[0])
    row_step = [row for row in range(0, img_shape[1], tile_size)]
    row_step.append(img_shape[1])

    windows_list = []
    for i, j in itertools.product(
        range(0, len(col_step) - 1), range(0, len(row_step) - 1)
    ):
        windows_list.append(
            tuple(
                (
                    row_step[j],
                    col_step[i],
                    row_step[j + 1] - row_step[j],
                    col_step[i + 1] - col_step[i],
                )
            )
        )
    return windows_list


class LargeImageDataset(Dataset):
    def __init__(
        self,
        image_files,
        mask_files,
        tile_size=512,
        transforms=None,
        image_bands=None,
        mask_bands=None,
    ):
        self.image_files = image_files
        self.image_bands = image_bands
        self.tile_size = tile_size
        self.mask_files = mask_files
        self.mask_bands = mask_bands
        self.transforms = transforms
        self.format_data = format_to_dict

        self.load_array = geoimage_load_tile
        ## init tiles/windows list
        self.tiles_list = []
        for img_id, img_path in enumerate(self.image_files):
            with rasterio.open(img_path) as img_ds:
                # shape dimension is [C, W, H ]
                img_width = img_ds.width
                img_heigth = img_ds.height
                img_shape = img_ds.shape  # shape = (H, W)
                # print(f" W={img_width}, H={img_heigth}, shape ={img_shape}")

            windows_list = get_img_windows_list(img_shape, self.tile_size)
            tile_img_list = [(img_id, window) for window in windows_list]
            self.tiles_list.extend(tile_img_list)

        # shuffle list
        random.shuffle(self.tiles_list)

    def __len__(self):
        return len(self.tiles_list)

    def __getitem__(self, index):
        # get path
        idx, window = self.tiles_list[index]

        # print(window)
        # load array
        img = self.load_array(
            self.image_files[idx], band_indices=self.image_bands, window=window
        )
        # print(img.shape)

        msk = self.load_array(
            self.mask_files[idx], band_indices=self.mask_bands, window=window
        )
        # print(msk.shape)

        data = self.format_data(image=img, mask=msk)

        if self.transforms is not None:
            for t in self.transforms:
                data = t(**data)

        return data
