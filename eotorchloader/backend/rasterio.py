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
import rasterio
from rasterio.windows import Window


def geoimage_simple_load(img_path, band_indices):
    """
    load a geo image in numpy array of shape C * W * H

    """

    with rasterio.open(img_path) as src:
        img_array = src.read(indexes=band_indices)

    if img_array.ndim == 2:
        img_array = img_array[np.newaxis, ...]

    return img_array


def geoimage_load_tile(img_path, band_indices, window):
    """

    windows : tuple with col_off, row_off, width, height
    """

    with rasterio.open(img_path) as src:
        img_array = src.read(window=Window(*window), indexes=band_indices)

    if img_array.ndim == 2:
        img_array = img_array[np.newaxis, ...]

    return img_array
