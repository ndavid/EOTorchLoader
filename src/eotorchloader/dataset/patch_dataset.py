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
        """
        Pytorch dataset to load image/mask samples from list of files.

        By default the images are fully loaded in memory before applying transform.
        If image are too big this could lead to memory issues. This dataset should
        be used with small image files (pretiled/precropped training dataset)

        Parameters
        ----------
        image_files : Path
            List of image path
        mask_files : List[str]
            list of mask/fround truth path. Must be of same size than image_files and
            with sample in the same order.
        transforms : List[BasicTransform]
            list of transform to apply to the sample (image, mask). No defaut transform
            are apply if None.
        image_bands : List[Int]
            list of bands to read/extract from image file. If none all band are read.
            follow rasterio convention and ordering begin to one
        mask_bands : List[Int]
            list of bands to read/extract from mask file. If none all band are read.
            follow rasterio convention and ordering begin to one

        """
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
