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
import matplotlib.pyplot as plt


def view_patch(data, transforms=None):
    """
    dataset: dataset contains tile & mask
    idx : index

    Returns : plot tile & mask
    """
    if transforms is not None:
        for t in transforms:
            data = t(**data)

    raster_tile = data["image"]
    raster_gt = data["mask"]

    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    ax[0].imshow(raster_tile)
    ax[0].set_title("Raster Tile")
    ax[0].set_axis_off()

    ax[1].imshow(raster_gt)
    ax[1].set_title("Raster Gt")
    ax[1].set_axis_off()

    plt.tight_layout()
    plt.show()
    return plt


def view_batch(batch_data, transforms=None, size=None, ncols=None):

    raster_tiles = batch_data["image"]
    raster_gts = batch_data["mask"]

    batch_size = raster_tiles.shape[0]
    ncols = batch_size
    if size is not None:
        ncols = size

    figure, ax = plt.subplots(nrows=2, ncols=ncols, figsize=(20, 8))

    for idx in range(ncols):
        if transforms is not None:
            data = {"image": raster_tiles[idx], "mask": raster_gts[idx]}
            for t in transforms:
                data = t(**data)

            raster_tile = data["image"]
            raster_gt = data["mask"]
        else:
            raster_tile = raster_tiles[idx]
            raster_gt = raster_gts[idx]

        ax[0][idx].imshow(raster_tile)
        ax[0][idx].set_axis_off()

        ax[1][idx].imshow(raster_gt)
        ax[1][idx].set_axis_off()

    plt.tight_layout()
    plt.show()
    return plt
