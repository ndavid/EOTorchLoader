.. _getting_started:

===============
Getting started
===============

Welcome to the getting started with **EOTorchLoader** tutorial!

This tutorial assumes you have already installed EOTorchLoader.
For help with installation see our :ref:`installation tutorial<installation>`.

This tutorial will teach you all the more common way to use EOTorchLoader in
you pytorch/pytorch lighningt training code.
At the end of the tutorial you should be able to use EOTorchLoader for loading
standard EO imagery in a pytroch deep learning training loop.


Get some EO data
-----------------
We first need some Earth Observation data, in this tutorial we will use two
common benchmark dataset in remote sensing which are *small* to speed up a little
this tutorial. Theses dataset are :

* `ISPRS 2D Semantic Labeling Contest - Potsdam <https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx>`_
* `The INRIA Aerial Image Labeling Dataset <https://project.inria.fr/aerialimagelabeling/>`_

If you had not already download the data, use the following instruction :

.. tabs::

   .. tab:: ISPRS Potsdam

      the postdam data could be downloaded from the dedicated `ISRPS page <https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/>`_
      with password CjwcipT4-P8g

      .. code-block:: bash

         mkdir -p EOData/ISPRS/Potsdam
         cd EOData/ISPRS/Potsdam
         wget $(curl --silent https://seafile.projekt.uni-hannover.de/api2/f/429be50cc79d423ab6c4/ | tr -d '"')
         unzip -j Potsdam.zip "Potsdam/4_Ortho_RGBIR.zip" -d .
         unzip -j Potsdam.zip "Potsdam/5_Labels_all.zip" -d .
         unzip 4_Ortho_RGBIR.zip
         unzip 5_Labels_all.zip -d 5_Labels_all


   .. tab:: INRIA

      The data could be donwloaded from the `INIRA site <https://project.inria.fr/aerialimagelabeling/files/>`_ or directly
      from terminal

      .. code-block:: bash

         mkdir -p EOData/INRIA
         cd EOData/INRIA
         curl -k https://files.inria.fr/aerialimagelabeling/getAerial.sh | bash
         rm .7z.00*


Prepair dataframe with EO metadata
-----------------------------------

In EOTorchLoader the path of image and mask data are set outside EoTorchLoader Datapipe and Datamodule
and should be defined by the user.
The prefered way for this is to construct a Pandas Dataframe with all informations
for an (image, mask) sample on each row of the dataframe.
