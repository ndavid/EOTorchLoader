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
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ..dataset.scene_dataset import LargeImageDataset


class TerriaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_df,
        transforms=None,
        img_col="images",
        img_bands=None,
        mask_col="mask",
        mask_bands=None,
        group_col="fold",
        set_config=None,
        tile_size=512,
        batch_size=4,
        num_workers=None,
    ):

        super().__init__()
        self.batch_size = batch_size
        self.tile_size = tile_size
        if num_workers:
            self.num_workers = num_workers
        else:
            self.num_workers = 2

        self.transform = transforms
        self.image_bands = img_bands
        self.mask_band = mask_bands

        self.data_df = data_df
        self.img_col = img_col
        self.mask_col = mask_col
        self.group_col = group_col
        if set_config is None:
            self.set_config = {"train": ["train"], "val": ["val"], "test": ["test"]}
        else:
            self.set_config = set_config

    def prepare_data(self):
        # rasterize from vector ?
        pass

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_config = self.set_config["train"]
            train_df = self.data_df[self.data_df[self.group_col].isin(train_config)]
            image_files_train = train_df[self.img_col].values
            mask_files_train = train_df[self.mask_col].values
            self.train_dataset = LargeImageDataset(
                image_files=image_files_train,
                mask_files=mask_files_train,
                tile_size=self.tile_size,
                transforms=self.transform,
                image_bands=self.image_bands,
                mask_bands=self.mask_band,
            )

            val_config = self.set_config["val"]
            val_df = self.data_df[self.data_df[self.group_col].isin(val_config)]
            image_files_val = val_df[self.img_col].values
            mask_files_val = val_df[self.mask_col].values
            self.val_dataset = LargeImageDataset(
                image_files=image_files_val,
                mask_files=mask_files_val,
                tile_size=self.tile_size,
                transforms=self.transform,
                image_bands=self.image_bands,
                mask_bands=self.mask_band,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_config = self.set_config["test"]
            test_df = self.data_df[self.data_df[self.group_col].isin(test_config)]
            image_files_test = test_df[self.img_col].values
            mask_files_test = test_df[self.mask_col].values
            self.test_dataset = LargeImageDataset(
                image_files=image_files_test,
                mask_files=mask_files_test,
                tile_size=self.tile_size,
                transforms=self.transform,
                image_bands=self.image_bands,
                mask_bands=self.mask_band,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
