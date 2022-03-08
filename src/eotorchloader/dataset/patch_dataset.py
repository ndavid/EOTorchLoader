# Copyright Nicolas DAVID
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
import torch
from torch.utils.data import Dataset, DataLoader
from ..backend.rasterio import geoimage_simple_load
from ..transform.base import format_to_dict


class PatchDataset(Dataset):
    def __init__(
        self,
        image_files,
        mask_files,
        transforms=None,
        image_bands=None,
        mask_bands=None,
    ):
        self.image_files = image_files
        self.image_bands = image_bands
        self.mask_files = mask_files
        self.mask_bands = mask_bands
        self.transforms = transforms

        self.load_array = geoimage_simple_load
        self.format_data = format_to_dict

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # get path
        image_file = self.image_files[index]
        mask_file = self.mask_files[index]

        # load array
        img = self.load_array(image_file, band_indices=self.image_bands)

        msk = self.load_array(mask_file, band_indices=self.mask_bands)

        data = self.format_data(image=img, mask=msk)

        if self.transforms is not None:
            for t in self.transforms:
                data = t(**data)

        return data
