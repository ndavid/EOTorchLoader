.. _installation:

==================
Installation Guide
==================


Python version support
#######################

Officially Python 3.8, 3.9 and 3.10.


Installing EOTorchLoader
########################

From PyPI
=========

TODO no packaging done yet


From conda-forge
================

TODO no packaging done yet


From GitHub
============

Option 1 (as a part of your system-wide module):
-------------------------------------------------

.. code-block:: bash

   pip install git+https://github.com/ndavid/EOTorchLoader#egg=eotorchloader


Option 2 (editable installation):
---------------------------------

To install an editable version of EOTorchLoader, it is recommended to clone the codebase directly:

.. code-block:: bash

   git clone https://github.com/ndavid/EOTorchLoader.git

This command will create a ``EOTorchLoader/`` folder in your current directory.
You can install it by running:

.. code-block:: bash

   cd EOTorchLoader/
   pip install --editable .

To uninstall the package please run:

.. code-block:: bash

   pip uninstall eotorchloader

Alternatively, simply adding the root directory of the cloned source code (e.g., ``/workspace/Documents/EOTorchLoader/src``) to your ``$PYTHONPATH``
and the codebase is ready to use.


Validating the install
########################

TODO


Dependencies
##############
