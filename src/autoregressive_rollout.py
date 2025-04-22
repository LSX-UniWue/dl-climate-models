"""
This script is used to generate a rollout using a trained model.
It provides functionality to incrementally write results into a NetCDF file to avoid memory overflow.
"""

from datamodule import MultidirNcDataset, load_xr_dataset, load_normalize_data
from models.base_model import NetArchitecture
from utils.lookup_mappings import VAR_TO_CMIP_NAME
from utils.data_utils import datetime_to_cftime, get_var_name_and_level, add_hpa_zeros_to_var_names
from utils.eval_utils import (
    get_checkpoint_path_from_config,
    load_model,
)

import argparse
import os
import math
import logging
import yaml
from typing import List

import torch
from torch.utils.data import DataLoader
import numpy as np
from netCDF4 import Dataset as nc_Dataset
import pandas as pd
import xarray as xr
import cftime


class PredictionWriter:
    """
    Class to handle writing predictions to a NetCDF file incrementally.
    """

    def __init__(self, dataset: MultidirNcDataset, save_file: str):
        self.file_path = save_file
        self.create_forecast_nc_file(dataset, save_file)
        self.write = self.write_forecast_nc_file

    def create_forecast_nc_file(self, dataset: MultidirNcDataset, save_file: str):
        '''
        Create a NetCDF file to store the forecast results with same longitude, latitude, pressure level dimensions
        and variables as provided dataset. Multiple forecasts are stored in the same file requiring to store the date
        of the initial condition and encode the time dimension as timedelta.

        Assumes same pressure levels for all variables on pressure levels
        TODO: Check if works if pressure levels are different.
        '''
        plev = []
        for v, levels in {**dataset.prognostic_vars, **dataset.diagnostic_vars}.items():
            plev.extend(levels)
        plev_data = sorted(list(set(plev)))

        ncfile = nc_Dataset(save_file,
                            'w', format='NETCDF4')
        ncfile.createDimension('lat', dataset.get_lat().shape[0])
        ncfile.createDimension('lon', dataset.get_lon().shape[0])
        ncfile.createDimension('plev', len(plev_data))
        ncfile.createDimension('timedelta', dataset.n_steps)
        ncfile.createDimension('init_condition', len(dataset.init_dates))

        lats = ncfile.createVariable('lat', np.float32, ('lat',))
        lons = ncfile.createVariable('lon', np.float32, ('lon',))
        plev = ncfile.createVariable('plev', np.float32, ('plev',))
        timedelta = ncfile.createVariable(
            'timedelta', np.int32, ('timedelta',))
        init_condition = ncfile.createVariable(
            'init_condition', np.float64, ('init_condition',))

        init_condition.units = f'hours since {dataset.init_dates[0]}'
        init_condition.calendar = 'gregorian'

        lats[:] = dataset.get_lat()
        lons[:] = dataset.get_lon()
        plev[:] = plev_data
        timedelta[:] = [
            (k+1)*dataset.lead_time_hours for k in range(dataset.n_steps)]

        init_condition[:] = cftime.date2num(datetime_to_cftime(
            dataset.init_dates), units=init_condition.units, calendar=init_condition.calendar)

        for v, levels in {**dataset.prognostic_vars, **dataset.diagnostic_vars}.items():
            if len(levels) > 0:
                ncfile.createVariable(
                    VAR_TO_CMIP_NAME.get(v, v), np.float32, ('init_condition', 'timedelta', 'plev', 'lat', 'lon'), zlib=True, complevel=0)
            else:
                ncfile.createVariable(
                    VAR_TO_CMIP_NAME.get(v, v), np.float32, ('init_condition', 'timedelta', 'lat', 'lon'), zlib=True, complevel=0)

        logging.info(init_condition)

        ncfile.close()

    def write_forecast_nc_file(self, all_forecasts_denormalized: np.ndarray, var_names_out: List[str], sample_indices, time_indices):
        '''
        args:
        all_forecasts_denormalized: (init_condition, time, var, lat, lon)
        '''

        nc_file = nc_Dataset(self.file_path, 'r+')
        for k, var in enumerate(var_names_out):
            name, level = get_var_name_and_level(var)
            if level is not None:
                pl_index = list(nc_file['plev'][:].astype(int)).index(level)
                nc_file[name][sample_indices, time_indices, pl_index,
                              :, :] = all_forecasts_denormalized[:, :, k]
            else:
                nc_file[name][sample_indices, time_indices, :,
                              :] = all_forecasts_denormalized[:, :, k]

        logging.info("Write results to disk")
        nc_file.sync()
        logging.info("Close file")
        nc_file.close()


def rollout(model: NetArchitecture,
            dataloader: DataLoader,
            batch_in_memory_steps: int,
            writer: PredictionWriter,
            keep_constant: List[str],
            inference_dropout_args: dict,
            device: str) -> torch.Tensor|None:
    """
    This function is used to generate a rollout using a trained model.
    Rollout is performed in several in-memory blocks to avoid memory overflow.
    For each block, forcings are loaded into memory and the forecast is saved to a NetCDF file.

    Args:
        model: model used for one-step-ahead prediction
        dataloader: dataloader providing intial condition and forcings
        batch_in_memory_steps: number of timesteps during rollout to keep in memory
        writer: provides functionality to write results to disk
        keep_constant: For further analysis, variable names to keep constant during rollout
        inference_dropout_args: args to create an ensemble during inference with dropout
        device: device to use for the model

    Returns: the predictions if not written to disk
    """
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size
    var_names_in = tuple(dataset.var_names_in)  # Must be hashable in model
    var_names_out = tuple(dataset.var_names_out)
    var_names_prognostics = tuple(dataset.var_names_prognostics)

    in_memory_results = {}
    in_memory_results['trajectories'] = []
    in_memory_results['dates'] = []

    for batch_id, batch in enumerate(dataloader):
        batch = {k: v.to(device) if torch.is_tensor(
            v) else v for k, v in batch.items()}

        initial_condition = batch.pop('initial_condition')
        dates = batch.pop('dates')

        # Forcing does not contain initial step and last step
        forcing = batch.pop('forcing')
        forecast_timedeltas = forcing['timedelta']
        logging.info(f"Batch dates {dates.shape[1]}, {dates}")
        logging.info(
            f"Time deltas {forecast_timedeltas.shape[0]}, {forecast_timedeltas}")

        # To Device
        model = model.to(device)

        # We could save memory by not storing the not normalized targets. We could then normalize everything in the dataloader
        initial_condition = dataset.norm_init(initial_condition)

        if writer is None:
            num_in_memory_blocks = 1
            in_memory_steps_per_sample = dataset.n_steps
        else:
            in_memory_steps_per_sample = math.ceil(batch_in_memory_steps /
                                                   initial_condition.shape[0])
            num_in_memory_blocks = math.ceil(
                forcing.timedelta.shape[0] / (in_memory_steps_per_sample))
        logging.info(f"Total forecast steps {dataset.n_steps}")
        logging.info(
            f"In memory steps per sample {in_memory_steps_per_sample}")
        logging.info(f"Number of in memory blocks {num_in_memory_blocks}")

        rollout_step_counter = 0
        for i in range(num_in_memory_blocks):
            logging.info(f"Block {i+1}/{num_in_memory_blocks}")
            block_time_steps = forecast_timedeltas.values[i *
                                                          in_memory_steps_per_sample:(i+1)*in_memory_steps_per_sample]
            # logging.info(f"Forcings in Block {block_time_steps.shape[0]}")
            # logging.info(f"Forcing time steps in block for batch {block_time_steps[:3]} ... {block_time_steps[-3:]}")
            in_memory_forcing = forcing.sel(timedelta=block_time_steps)

            # Stack variables of dimension (sample x timedelta x levels x lat x lon)
            in_memory_forcing = np.concatenate(
                [in_memory_forcing[c].sel(level=level).values if len(
                    level) > 0 else in_memory_forcing[c].values[:, :, None, :, :] for c, level in dataset.forcing_vars.items()], axis=2)

            in_memory_forcing = torch.as_tensor(in_memory_forcing)
            in_memory_forcing = dataset.norm_forcing(in_memory_forcing)
            in_memory_forcing = in_memory_forcing.to(device)

            all_forecasts_normalized = []
            with torch.no_grad():
                # initial conditions contains initial forcing
                if i == 0:
                    block_step_range = range(-1, block_time_steps.shape[0])
                else:
                    block_step_range = range(0, block_time_steps.shape[0])
                for j in block_step_range:

                    if i == 0 and j == -1:
                        input = initial_condition

                        keep_constant_idx = [v_idx for v_idx, v in enumerate(
                            var_names_in) if v in keep_constant]
                        keep_constant = initial_condition[:, keep_constant_idx]

                    else:
                        input = torch.cat([initial_condition[:, dataset.constant_vars_idx],
                                           in_memory_forcing[:, j], next_input[:, dataset.prognostic_vars_idx_output]], dim=1).float()

                        if len(keep_constant_idx) > 0:
                            print("Input shape", input.shape)
                            input[:, keep_constant_idx] = keep_constant

                    ensemble_preds = []
                    if inference_dropout_args['inference_dropout']:
                        model.train()
                        ensemble_size = inference_dropout_args['ensemble_size']
                    else:
                        model.eval()
                        ensemble_size = 1

                    for r in range(ensemble_size):

                        if len(inference_dropout_args['fix_seed']) > 0:
                            torch.manual_seed(
                                inference_dropout_args.fix_seed[r])

                        # B, prognostics+diagnostics, H, W
                        preds = model.forward(
                            input, var_names_in=var_names_in, var_names_out=var_names_out, var_names_prognostics=var_names_prognostics, **batch)

                        if not torch.isfinite(preds).all():
                            logging.info(torch.where(~torch.isfinite(preds)))
                            logging.info(f"Current step {block_time_steps[j]}")
                            raise ValueError("pred contains NaN or Inf values")

                        ensemble_preds.append(preds)

                    ensemble_preds = torch.stack(ensemble_preds, dim=0)
                    next_input = ensemble_preds.mean(dim=0)

                    all_forecasts_normalized.append(next_input.cpu())

            all_forecasts_normalized = torch.stack(
                all_forecasts_normalized, dim=1)
            all_forecasts_denormalized = dataset.denorm_preds(
                all_forecasts_normalized).numpy()

            if writer is not None:
                time_indices = slice(
                    rollout_step_counter, rollout_step_counter + len(block_step_range))
                sample_indices = slice(
                    batch_id*batch_size, batch_id*batch_size + initial_condition.shape[0])

                writer.write(all_forecasts_denormalized, var_names_out,
                             sample_indices=sample_indices, time_indices=time_indices)
            else:
                in_memory_results['trajectories'].append(
                    all_forecasts_normalized)
                in_memory_results['dates'].append(dates)

            rollout_step_counter += len(block_step_range)

    if writer is None:
        return torch.cat(in_memory_results['trajectories'], dim=0).numpy(), np.concatenate(in_memory_results['dates'], axis=0)
    else:
        return None, None


def main(config_paths, data_dirs, save_dir, forecast_time, batch_size, init_dates, in_memory_steps, overwrite_results,
         keep_constant, inference_dropout, ensemble_size, fix_seed):
    
    print('\nStart Autoregressive Rollout\n')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Iterate over all models
    for config_path in config_paths:
        with open(os.path.join(os.environ['RESULTS_DIR'], config_path), 'r') as f:
            config = yaml.safe_load(f)

        # Set path to data
        if len(data_dirs) > 0:
            data_dirs = [os.path.join(os.environ['DATA_DIR'], p)
                         for p in data_dirs]
        else:
            data_dirs = config['data']['data_dirs']

        # Set up logging
        save_dir = os.path.join(
            config['trainer']['default_root_dir'], save_dir, forecast_time)
        print("Save dir", save_dir)
        os.makedirs(save_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(os.path.join(
                    save_dir, f'log.log'), mode='w'),
                logging.StreamHandler()
            ]
        )
        logging.info('Start iterative roll-out')
        logging.info(f"Data dirs: {data_dirs}")

        save_file = os.path.join(save_dir, 'out.nc')

        logging.info("Load data")
        dataset = load_xr_dataset(
            data_dirs, in_memory_steps)

        logging.info("Load normalization data")
        normalize_data = load_normalize_data(
            config['data']['normalization_dirs'])

        checkpoint_path = get_checkpoint_path_from_config(config)
        print("Loading the following model checkpoint: ", checkpoint_path)
        model = load_model(config, checkpoint_path)

        # Overwrite n_steps in config with forecast_time
        if forecast_time is not None:
            n_steps = pd.Timedelta(forecast_time) / \
                np.timedelta64(model.lead_time_hours, 'h')
            assert int(
                n_steps)-n_steps < 1e-8, "Forecast time must be multiple of lead time"
            logging.info(
                f"Overwriting n_steps in config with {int(n_steps)}")
            config['data']['train_config']['n_steps'] = int(n_steps)

        # Overwrite some config parameters
        config['data']['train_config']['init_dates'] = init_dates
        config['data']['train_config']['init_condition_only'] = True
        config['data']['train_config']['noise'] = 0.0

        test_config = {**config['data']['train_config'],
                       **config['data']['test_config']}

        inference_dropout_args = {
            'inference_dropout': inference_dropout,
            'ensemble_size': ensemble_size,
            'fix_seed': fix_seed,
        }

        dataset = MultidirNcDataset(
            dataset=dataset, normalize_data=normalize_data, **test_config)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size if batch_size is not None else len(init_dates),
            shuffle=False,
            drop_last=False,
            collate_fn=dataset.collate
        )

        if in_memory_steps > 0:
            writer = PredictionWriter(dataset, save_file)
        else:
            writer = None

        keep_constant = set(keep_constant)

        try:
            rollout(model, dataloader, in_memory_steps,
                    writer, keep_constant, inference_dropout_args, device)
        except Exception as e:
            logging.error(f"Error during rollout: {str(e)}")
            # Write exception to file
            with open(os.path.join(save_dir, 'error.txt'), 'w') as f:
                f.write(str(e))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_paths", type=str,
                        nargs='+', help="Path to the config files")
    parser.add_argument("--forecast_time", type=str,
                        help="Defines length of rollout. Timedelta str in format '10D'", required=True)
    parser.add_argument("--data_dirs", type=str, nargs='+',
                        help="Path to data containing .nc file if a data source different from the one during training should be used",
                        default=[])
    parser.add_argument("--save_dir", type=str,
                        help="name of save directories", default="evaluation")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size for rollout. If None use number of init dates", default=None)
    parser.add_argument("--init_dates", type=str, nargs='+',
                        default=['2009-01-01T00', '2009-02-01T00'])
    parser.add_argument("--in_memory_steps", type=int,
                        help="Number of datapoints to keep in memory across batch. Set to -1 for all in memory, but result is not written to disk", default=500)
    parser.add_argument("--overwrite_results",
                        action='store_true', default=False, help="Overwrite existing results .nc file")
    parser.add_argument("--keep_constant", nargs='+', default=[],
                        help="List of variables to keep the same as initials condition during rollout")

    # Arguments to create an enesmble during inference with dropout
    parser.add_argument("--inference_dropout", action='store_true', default=False,
                        help="Use dropout during evaluation")
    parser.add_argument("--ensemble_size", type=int,
                        help="Number of ensemble members", default=1)
    parser.add_argument("--fix_seed", type=int, nargs="+",
                        help="Seeds for dropout", default=[])

    args = parser.parse_args()

    args = vars(args)

    main(**args)
