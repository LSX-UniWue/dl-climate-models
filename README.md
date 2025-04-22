# Exploring Design Choices For Autoregressive Deep Learning Climate Models

This repository contains accompanying the paper "Exploring Design Choices For Autoregressive Deep Learning Climate Models"

Our work explores key design choices for autoregressive Deep Learning (DL) models to achieve stable 10-year rollouts that preserve the statistics of the reference climate. We quantitatively compare the long-term stability of three prominent model architectures — FourCastNet, SFNO, and ClimaX — trained on ERA5 reanalysis data at 5.625° resolution and systematically assess the impact of autoregressive training steps, model capacity and choice of prognostic variables.​


# Getting started

1. Set up environment (e.g. conda)
    - Python Version 3.11.11, e.g. via `conda create dlclim python=3.11.11 && conda activate dlclim`
    - Install the dependencies listed in the requirements.txt file, e.g. via `pip install -r requirements.txt`
    - Install this package in the environment via `pip install -e .`
2. Data
    - Set an environmental variable `DATA_DIR`, e.g. via bash `export DATA_DIR=YOUR_PATH`
    - For a quick start, download example data from this [zenodo repository](https://zenodo.org/records/15261558) and extract the files into the path of `DATA_DIR`
    - Alternatively, download the full raw dataset (see section [Data download](#data-download)) and run the script `bash example_scripts/preprocessing.sh`
3. Training of own models
    - Set an environmental variable `RESULTS_DIR`, e.g. via bash `export RESULTS_DIR=YOUR_PATH`
    - Run the training of a single model by running `bash example_scripts/train.sh`
    - Example configurations can be found in the folder example_configs
    - An example hyperparameter search can be run with the script `bash example_scripts/hyperparameter_search.sh`
4. Evaluation
    - Set an environmental variable `RESULTS_DIR`, e.g. via bash `export RESULTS_DIR=YOUR_PATH`
    - Run the evaluation script once you trained a model with the scripts `example_scripts/evaluate.sh`

# Data download

We use two data sources:
- [WeatherBench](https://github.com/pangeo-data/WeatherBench)
- [Official solar forcing from CMIP6](https://gmd.copernicus.org/articles/10/2247/2017/)

### WeatherBench

Download the 5.625 degree data from the [TUM server](https://dataserv.ub.tum.de/index.php/s/m1524895?path=%2F)---which can also be done via the command line, as detailed [here](https://mediatum.ub.tum.de/1524895) -- to the `DATA_DIR` directory.
For best compatibility with this repository, store the netcdf files of each variable in a separate folder, i.e.,

```
.
|-- DATA_DIR
|   |-- ERA5
|       |-- weatherbench1
|           |-- r64x32
|               |-- 10m_u_component_of_wind
|               |-- 10m_v_component_of_wind
|               |-- 2m_temperature
|               |-- constants
|               |-- geopotential
|               |-- potential_vorticity
|               |-- relative_humidity
|               |-- specific_humidity
|               |-- temperature
|               |-- toa_incident_solar_radiation
|               |-- total_cloud_cover
|               |-- total_precipitation
|               |-- u_component_of_wind
|               |-- v_component_of_wind
|               |-- vorticity
```

We provide a script `data_preprocessing/compute_normalization.py' to compute mean and standard deviation per variable and level which is used for normalization.

### Solar Forcing
Daily total solar irradiance (TSI) from an official forcing for CMIP6 and is available for download [here](https://www.wdc-climate.de/ui/cmip6?input=input4MIPs.CMIP6.CMIP.SOLARIS-HEPPA.SOLARIS-HEPPA-3-2)

We provide a script `data_preprocessing/compute_tisr_heppa.py` adapted from the [GraphCast](https://github.com/google-deepmind/graphcast) to compute Top-of-the-atmosphere incoming solar radiation from the TSI values.

# Project overview
The training script is based on [PyTorch Lightning CLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html).

- `train.py` starts model training
- `datamodule.py` defines a LightningDataModule which wraps a PyTorch Iterable-style dataset
- `modelmodule.py` defines a LightningModule defining multi-step autoregressive training logic
- `metrics.py` handles metric computation and its necessary data during training
- `autoregressive_rollout.py` is a python script to perform an inference rollout with a trained model
- the `models` folder contains code that defines the model architectures
- the `evaluation` folder contains scripts to evaluate inference rollouts
- the `data_preprocessing` folder contains scripts to derive necessary data products from the raw data (see section Data download)



