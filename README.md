# Graph Neural Networks for Improved El Nino Forecasting
## Data
- Download the datasets from [this link](https://drive.google.com/drive/folders/15L2cvpAQv_c6c6gmJ8RnR2tQ_mHQR9Oz?usp=sharing)
- **Overwrite [this file](hyperparams_and_args.py) so that ``data_dir`` correctly points towards your data directory** (with subdirs SODA, GODAS, etc.).

## Environment setup
- Git clone the repo 

- Implemented using Python3 (3.7) with dependencies specified in requirements.txt, install them in a clean conda environment: <br>
    - ``conda create --name projectx python=3.7`` <br>
    - ``conda activate projectx`` <br>
    - ``pip install -r requirements.txt``
    - the [correct PyTorch version for your platform](https://pytorch.org/get-started/locally/]), e.g. ``conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch``
    - If you want to use any plotting code: ``conda install -c conda-forge cartopy``

This should be enough for Pycharm, for command line stuff you'll probably need to then also run

- ``python setup.py install``.


## Models
All reported models are saved [here](models).

## Running the experiments
- To run experiment 1 from scratch just rerun the [corresponding notebook](experiment1.ipynb).
- To run experiment 2 from scratch/use transfer learning with CMIP5 run:
    ``python transfer_learning_exp.py --horizon <#lead months> --transfer_epochs <e_1> --epochs <e_2>``
      as well as any other combination of hyperparameters (settable, as above, via ``--parameter <value>``).