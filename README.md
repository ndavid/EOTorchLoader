EOTorchLoader: aerial and satellite imagery datamodule for pytorch
====================================================================

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PytorchLightning/pytorch-lightning/blob/master/LICENSE)


**EOTorchLoader** is an open source project and Python package
to simplify the use of Earth Observation imagery in deep learning
and in particular for pytorch and pytorch lighnting code.

   !! EARLY WORK In PROGRESS !!


Why EOTorchLoader?
------------------
**E**arth **O**bservation (EO) imageries have often a large size,
their height and width could easily be larger than 10k pixels,
possesses more spectral bands than RVB, and also come with
multi-modality : spectral imagery, SAR, height data, etc.
Due to this caracteristics, EO imagery must be pre-process in
order to be used in common deep learning framework as pytorch
and pytorch-lightning.

One of the most common pre-processing is the chipping|tiling|croping
of large EO images into small images|patchs in order to format data in
the same way than natural imageries are formatted in standard
deep learning datasets (ImageNet or Cityscape). It could be made
offline when writing the tiling results in disk or online during
the training of DL models. The latter has the advantage of not
duplicated data and to enable random cropping in large imagery during
training (vs fixed cropping).

Another distinctive features of EO imagery is that patchs/tiles of
a large EO image could not always be seen as independants as it exists
a **strong spatial correlation** in EOData.
So in order to train a DL models on EO data and to measure is
genericity this spatial correlation should be taken
in account and the split between fold or train|validation|test
should be made with distinct geographical area and not by random
splitting of tile|patch.

Finally due to their multi-modal and multi-spectral aspects, EOData
need specifics transforms for data augmentation in DL training, and
common data augmentation libray as albumentation or kornia are useful
but not sufficients for EO imagery

So EOTorchLoader aims is to bring the following features for
deep learning processing of Earth observation data :

* efficient online tiling|croping of EO data in Dataloader
* enable the use of geographic split between train/val/test or kfold
  dataset following good pratices.
* add useful transform for EO imagery.
* configurable with hydra/omegaconf.


Documentation
-------------

Learn more about EOTorchLoader in its official documentation at
https://ndavid.github.io/EOTorchLoader/


License
-------

Copyright 2022 Nicolas David

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
