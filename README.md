# Graph Neural Networks for Improved El Nino Forecasting
*Deep learning-based models have recently outperformed state-of-the-art seasonal forecasting models, such as for predicting El Nino-Southern Oscillation (ENSO).
However, current deep learning models are based on convolutional neural networks which are difficult to interpret and can fail to model large-scale atmospheric patterns called teleconnections. We propose the first application of spatiotemporal graph neural networks (GNNs), that can model teleconnections for seasonal forecasting. Our GNN outperforms other state-of-the-art machine learning-based (ML) models for forecasts up to 3 month ahead. The explicit modeling of information flow via edges makes our model more interpretable, and our model indeed is shown to learn sensible edge weights that correlate with the ENSO anomaly pattern.*
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
- To run experiment 2 from scratch/use transfer learning with CMIP5, run:
    ``python transfer_learning_exp.py --horizon <#lead months> --transfer_epochs <e_1> --epochs <e_2> --lat_min -40 --lat_max 40 --lon_min 0 --lon_max 360``
      as well as any other combination of hyperparameters (settable, as above, via ``--parameter <value>``).
      <br>
      E.g., to rerun 6lead_ONI_-40-40lats_0-360lons_3w2L2gcnDepth2dil_32bs0.1d0normed_prelu_100epPRETRAINED_150epTRAIN-CONCAT.pt
      you would do ``python transfer_learning_exp.py --horizon 6 --transfer_epochs 100 --epochs 150 --lat_min -40 --lat_max 40 --lon_min 0 --lon_max 360``
      
## Enhanced Interpretability
Besides the better inductive bias that a GNN encodes for large-scale phenomena like ENSO, it also is more interpretable than other deep learning methods.

We show that our GNN learns meaningful edges, see plots of [exp2](experiment2.ipynb).
Note how the one for 6 lead months closely ressembles the SSTAs ~6mon prior the extreme El Nino 2015/16:
![Alt Text](https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CMB/.GLOBAL/.Reyn_SmithOIv2/.monthly/.ssta/startcolormap/DATA/-5./5./RANGE/black/navy/blue/-5./VALUE/cyan/-0.5/VALUE/white/white/0.5/bandmax/yellow/0.5/VALUE/red/5./VALUE/firebrick/endcolormap/DATA/0.5/STEP/a-+++-a-++-a+X+Y+fig:+colors+nozero+contours+land+:fig+/T/last+24+sub/last/plotrange/X/20/380/plotrange/Y/-60/70/plotrange//plotaxislength+600+psdef//iftime+75+psdef//plotbordertop+40+psdef//plotborderbottom+40+psdef//XOVY+null+psdef//color_smoothing+null+psdef//antialias+true+psdef//mftime+75+psdef+.gif?T=May+2015+-+Oct+2015)

This is the typical ENSO pattern, another one from 6mon prior the extreme 1997/98 el Nino:
![Alt Text](https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CMB/.GLOBAL/.Reyn_SmithOIv2/.monthly/.ssta/startcolormap/DATA/-5./5./RANGE/black/navy/blue/-5./VALUE/cyan/-0.5/VALUE/white/white/0.5/bandmax/yellow/0.5/VALUE/red/5./VALUE/firebrick/endcolormap/DATA/0.5/STEP/a-+++-a-++-a+X+Y+fig:+colors+nozero+contours+land+:fig+/T/last+24+sub/last/plotrange/X/20/380/plotrange/Y/-60/70/plotrange//plotaxislength+600+psdef//iftime+75+psdef//plotbordertop+40+psdef//plotborderbottom+40+psdef//XOVY+null+psdef//color_smoothing+null+psdef//antialias+true+psdef//mftime+75+psdef+.gif?T=May+1997+-+Oct+1997&plotaxislength=740
)