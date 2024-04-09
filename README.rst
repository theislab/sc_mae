Masked Autoencoder in Single-Cell Genomics
==========================================

This repository provides a simple setting to train a Masked Autoencoder (MAE) on single-cell genomics data with a random masking strategy. The provided code is designed to work with a smaller scale `adata` object that fits into memory.

Contents
--------

- ``data.py``: Module for loading and preprocessing single-cell genomics data.
- ``Masking.ipynb``: Jupyter notebook demonstrating the random masking strategy.
- ``models.py``: Contains the implementation of the Masked Autoencoder.
- ``train.py``: Script for training the Masked Autoencoder model.
- ``train.sh``: Bash script for executing the training process.

System Requirements
-------------------

- Python 3.10
- Dependencies listed in `requirements.txt`

Usage
-----

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/your-username/your-repository.git

2. Install the required dependencies:

.. code-block:: bash

    pip install -r requirements.txt

3. Prepare the data:

- Download the sample data from the publication mentioned in the citation section or use your own processed adata object.

4. Execute the training script:

.. code-block:: bash

    bash train.sh

Demo
----

To apply this code, follow these steps:

1. **Download Sample Data**: You can download the `adata` object from the publication mentioned in the citation section or use your own processed h5ad object.

2. **Prepare Data**: If you are using your own data, make sure it is preprocessed and compatible with the provided code. Otherwise, follow the data loading and preprocessing steps in `data.py`.

3. **Train the Model**: Execute the training script `train.py` by running `bash train.sh`. Adjust the hyperparameters and configurations as needed in the script.

Citation
--------

This repository is a part of a larger project and serves as a simplified demo. If you use this code in your research, please cite the following paper:

**Delineating the Effective Use of Self-Supervised Learning in Single-Cell Genomics**

`Link to the paper <https://doi.org/10.1101/2024.02.16.580624>`_

`Link to the repository <github.com/theislab/ssl_in_scg>`_

If you use the sample data in your research, please cite the following paper:

**COMBATdb: a database for the COVID-19 Multi-Omics Blood ATlas**

`Link to the paper <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9825482/>`_

Acknowledgments
---------------

- The sample data used in this project is sourced from the COMBATdb.

Contribution
------------

Contributions to improve this codebase are welcome. Please fork the repository and submit a pull request with your changes.

License
-------

This project is licensed under the MIT License - see `MIT License <https://opensource.org/licenses/MIT>`_.

Please refer to the main repository for more detailed information and a more elaborate analysis.

Authors
-------

sc_mae was written by `Till Richter <till.richter@helmholtz-munich.de>`_.
