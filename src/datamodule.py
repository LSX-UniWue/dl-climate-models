
from typing import Optional, List, Union, Dict, TypedDict
import math

import numpy as np
import pandas as pd
import xarray as xr
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

from utils.data_utils import (
    load_xr_dataset,
    load_normalize_data,
    flatten_var_name_dict,
    var_names_to_CMIP,
    add_hpa_zeros_to_var_names,
    xr_variables_4dim_as_arrays,
    get_required_levels,
    xr_variables_3dim_as_arrays,
)


class MultidirNcDatasetConfig(TypedDict):
    """
    Configuration for the MultidirNcDataset.
    Lightning CLI with omegaconf requires all arguments to be typed and this avoids redundant information
    """
    start_date: str  # Datetime string in the format YYYY-MM-DD T HH:MM:SS to select a subset of the dataset
    end_date: str  # Datetime string in the format YYYY-MM-DD T HH:MM:SS to select a subset of the dataset
    # Either a string with a frequency (e.g. '1H', '1D') or a list of datetime strings in the format YYYY-MM-DD T HH:MM:SS to determine the timestamps of the initial conditions
    init_dates: Union[str, List[str]]
    # Dictionary with variable names as keys and levels as values. Levels in hPa
    forcing_vars: Dict[str, List[int]]
    # Dictionary with variable names as keys and levels as values
    prognostic_vars: Dict[str, List[int]]
    # Dictionary with variable names as keys and levels as values
    constant_vars: Dict[str, List[int]]
    # Dictionary with variable names as keys and levels as values
    diagnostic_vars: Dict[str, List[int]]
    lead_time: str  # Time delta string in the format '1H', '1D' to determine the lead time for the model
    n_steps: int  # Number of autoregressive steps that will be predicted for the model
    buffer_size: int  # Number of samples per shard that will be loaded into memory
    # Whether to load initial conditions in temporal order or shuffle them
    shuffle_dataset: bool
    # Whether to return only the initial conditions or also the forcing and target variables. This is used for long autoregressive runs
    init_condition_only: bool
    # Whether to compute the lead times for the model. This is used for ClimaX
    compute_lead_times: bool
    noise: float  # Noise to be added to the prognostic variables. This is used for training


class MultidirNcDataset(IterableDataset):
    """
    Iterable-style torch dataset to iterate over an xarray dataset.
    Data is loaded in shards into memory.
    Note that while level information is stored in xarray as a coordinate,
    the level dimension is flattened into the channel dimension for the model.
    The flattened variables are renamed to VName_lev e.g. ta_50000
    """

    def __init__(self,
                 dataset: xr.Dataset,
                 normalize_data: xr.Dataset,
                 **config: MultidirNcDatasetConfig
                 ):

        super().__init__()
        if len(config["diagnostic_vars"]) > 0:
            raise NotImplementedError(
                "Implementation of diagnostic variables not yet compatible with the model modules")

        # For convenience since levels are provided in hPa but the dataset uses Pa
        forcing_vars = add_hpa_zeros_to_var_names(config["forcing_vars"])
        prognostic_vars = add_hpa_zeros_to_var_names(config["prognostic_vars"])
        constant_vars = add_hpa_zeros_to_var_names(config["constant_vars"])
        diagnostic_vars = add_hpa_zeros_to_var_names(config["diagnostic_vars"])
        all_vars = {**forcing_vars, **prognostic_vars,
                    **constant_vars, **diagnostic_vars}

        self.lead_time = pd.Timedelta(config["lead_time"])

        # Compute initial dates
        if isinstance(config["init_dates"], str):
            self.init_dates = pd.date_range(config["start_date"], pd.to_datetime(
                config["end_date"])-config["n_steps"]*self.lead_time, freq=config["init_dates"]).to_numpy()
        else:
            self.init_dates = pd.to_datetime(config["init_dates"]).to_numpy()

        print("Number of Init dates", self.init_dates.shape[0])

        # Compute required timesteps
        required_timesteps = np.unique(
            [self.init_dates + i*self.lead_time for i in range(config["n_steps"]+1)])  # Plus 1 because we need to predict the next step

        print("Number of Required timesteps", required_timesteps.shape[0])

        # Select the requested subset of the dataset
        required_levels = get_required_levels(all_vars)
        required_variables = list(all_vars.keys())
        self.dataset = dataset[required_variables].sel(
            time=required_timesteps, plev=required_levels)

        # turn normalization xarray dataset into a dictionary where keys contain also the level
        self.normalize_data_dict = {}
        for v in normalize_data.data_vars:
            if 'plev' in normalize_data[v].dims:
                for l in normalize_data[v].plev.values:
                    self.normalize_data_dict[f'{v}_{int(l/100)}'] = normalize_data[v].sel(
                        plev=l).values.item()
            else:
                self.normalize_data_dict[v] = normalize_data[v].values.item()

        # Load constants into memory
        self.constants = xr_variables_3dim_as_arrays(
            self.dataset, constant_vars)

        if config["init_condition_only"]:
            # Note that xr_forcing could theoretically hold more levels than the forcing variables
            # Those must be removed during rollout
            self.xr_forcings = dataset[list(forcing_vars.keys())]

        # Store variable names and indices
        self.constant_vars = constant_vars
        self.forcing_vars = forcing_vars
        self.prognostic_vars = prognostic_vars
        self.diagnostic_vars = diagnostic_vars

        # Append level to variable names
        var_names_list = flatten_var_name_dict(constant_vars) + flatten_var_name_dict(
            forcing_vars) + flatten_var_name_dict(prognostic_vars) + flatten_var_name_dict(diagnostic_vars)
        var_names_array = np.array(var_names_list)

        # Compute indices for variables in numpy shards to be loaded
        self.init_vars_idx = np.array([var_names_list.index(v) for v in flatten_var_name_dict(
            constant_vars) + flatten_var_name_dict(forcing_vars) + flatten_var_name_dict(prognostic_vars)])

        self.target_vars_idx = np.array([var_names_list.index(v) for v in flatten_var_name_dict(
            prognostic_vars)+flatten_var_name_dict(diagnostic_vars)])

        # Get the transforms for variable normalization
        self.norm_init = self.get_normalize_variables(
            var_names_array[self.init_vars_idx])
        self.norm_target = self.get_normalize_variables(
            var_names_array[self.target_vars_idx])
        self.denorm_preds = self.get_denormalize(self.norm_target)

        # Rename variables to CMIP names to enable pre-training models on CMIP data
        constant_vars = var_names_to_CMIP(constant_vars)
        prognostic_vars = var_names_to_CMIP(prognostic_vars)
        diagnostic_vars = var_names_to_CMIP(diagnostic_vars)

        # Apply same processing to forcing variables
        if len(self.forcing_vars) > 0:
            self.forcing_vars_idx = np.array(
                [var_names_list.index(v) for v in flatten_var_name_dict(forcing_vars)])
            self.norm_forcing = self.get_normalize_variables(
                var_names_array[self.forcing_vars_idx])
            forcing_vars = var_names_to_CMIP(forcing_vars)
            self.var_names_forcings = flatten_var_name_dict(forcing_vars)

        # Store parameters
        self.n_steps = config["n_steps"]
        self.num_lat = dataset.lat.size
        self.num_lon = dataset.lon.size
        self.len_shard = config["buffer_size"]
        self.shuffle_dataset = config["shuffle_dataset"]
        self.init_condition_only = config["init_condition_only"]
        self.noise = config["noise"]
        self.lead_time_hours = int(self.lead_time / np.timedelta64(1, 'h'))
        self.compute_lead_times = config["compute_lead_times"]

        if config["compute_lead_times"]:  # For ClimaX
            self.lead_time_tensor = torch.tensor(
                self.lead_time_hours/100).float()

        # Precompute some helpers for data handling during multi-step training/rollouts
        self.var_names_constants = flatten_var_name_dict(constant_vars)
        self.var_names_prognostics = flatten_var_name_dict(prognostic_vars)
        self.var_names_in = flatten_var_name_dict(
            constant_vars) + flatten_var_name_dict(forcing_vars) + flatten_var_name_dict(prognostic_vars)
        self.var_names_out = flatten_var_name_dict(
            prognostic_vars)+flatten_var_name_dict(diagnostic_vars)

        self.constant_vars_idx = np.array([self.var_names_in.index(v) for v in flatten_var_name_dict(
            constant_vars)])
        self.prognostic_vars_idx_input = np.array([self.var_names_in.index(v) for v in flatten_var_name_dict(
            prognostic_vars)])
        self.prognostic_vars_idx_output = np.array([self.var_names_out.index(v) for v in flatten_var_name_dict(
            prognostic_vars)])

        # Compute the number of shards that the dataset will be split into
        self.num_samples = self.init_dates.shape[0]

        self.num_shards = math.ceil(
            self.init_dates.shape[0]/config["buffer_size"])

        if self.num_shards == 1 and not config["init_condition_only"]:
            print("Load full dataset into memory")
            self.loaded_dataset = self.load_shard(required_timesteps)
            print("Full dataset shape", self.loaded_dataset.shape)

        if self.shuffle_dataset:
            self.shard_numbers = np.random.permutation(self.num_shards)
        else:
            self.shard_numbers = np.arange(self.num_shards)

    def get_normalize_variables(self, variables: List[str]) -> transforms.Normalize:
        """
        Get the normalization transform for the given variables based on the mean and std in the normalization data.
        """

        mean = []
        std = []
        for var in variables:
            mean.append(self.normalize_data_dict[f'mean_{var}'])
            std.append(self.normalize_data_dict[f'std_{var}'])

        normalize_mean = np.array(mean)
        normalize_std = np.array(std)

        return transforms.Normalize(normalize_mean, normalize_std)

    def get_denormalize(self, normalize_transform: transforms.Normalize) -> transforms.Normalize:
        """
        Get the inverse transform for a given normalization transform.
        """
        mean_norm_out, std_norm_out = normalize_transform.mean, normalize_transform.std

        mean_denorm_out, std_denorm_out = -mean_norm_out / std_norm_out, 1 / std_norm_out
        return transforms.Normalize(mean_denorm_out, std_denorm_out)

    def get_lat(self) -> np.ndarray:
        return self.dataset.lat.values

    def get_lon(self) -> np.ndarray:
        return self.dataset.lon.values

    def load_shard(self, dates: pd.DatetimeIndex) -> torch.Tensor:
        """
        Load a shard into memory given a list of dates
        """

        shard_dataset = self.dataset.sel(time=dates)

        forcing = xr_variables_4dim_as_arrays(shard_dataset, self.forcing_vars)

        prognostic = xr_variables_4dim_as_arrays(
            shard_dataset, self.prognostic_vars)

        if self.noise > 0:
            prognostic = [
                d + np.float32(np.random.randn(*d.shape)*self.noise) for d in prognostic]
        diagnostic = xr_variables_4dim_as_arrays(
            shard_dataset, self.diagnostic_vars)

        # View constants
        constants = [np.broadcast_to(c, (dates.shape[0], *c.shape))
                     for c in self.constants]

        # This defines in which order the variables are concatenated
        # Important for variable handling in multi-step training/rollouts
        shard_dataset = torch.as_tensor(np.concatenate(
            constants+forcing+prognostic+diagnostic, axis=1)).float()

        return shard_dataset

    def __iter__(self):
        '''
        The main logic is to split the initial dates into shards.
        If shuffle is true, shuffle the order of the shards.
        Determine the initial dates and target steps for one shard and load them into memory.
        If shuffle is true, shuffle the order of the initial dates within the shard.
        Iterate over the (shuffled) initial dates and retrieve the target data from the in-memory shard.
        '''

        iter_start = 0
        iter_end = self.num_shards
        worker_info = torch.utils.data.get_worker_info()

        # Manage sharding for a dataloader with multiple workers
        # TODO: Check if codes works for distributed training
        if worker_info is not None:
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()

            if world_size > 1:
                raise NotImplementedError(
                    "Distributed training not supported yet. Shuffling not guaranteed to be the same across each gpu")

            num_workers_per_gpu = worker_info.num_workers
            num_workers = num_workers_per_gpu * world_size
            num_shards_per_worker = int(
                math.ceil(self.num_shards / float(num_workers)))
            worker_id = rank * num_workers_per_gpu + worker_info.id
            iter_start = worker_id * num_shards_per_worker
            iter_end = iter_start + num_shards_per_worker

        # Iterate over shards for the current worker. self.shard_numbers might be shuffled
        for i in self.shard_numbers[iter_start:iter_end]:

            # Retrieve the initial dates for the current shard
            init_date_idx = np.arange(
                i*self.len_shard, min((i+1)*self.len_shard, self.num_samples))
            init_dates = self.init_dates[init_date_idx]

            # Determine the required timesteps (init+target) for the current shard
            required_timesteps = np.unique(
                [init_dates + i*self.lead_time for i in range(self.n_steps+1)])

            if self.init_condition_only:
                timesteps_to_load = init_dates
            else:
                timesteps_to_load = required_timesteps

            # Load the shard into memory if not entire dataset is already in memory
            if self.num_shards > 1 or self.init_condition_only:
                shard_dataset = self.load_shard(timesteps_to_load)
            else:
                shard_dataset = self.loaded_dataset

            real_length = init_dates.shape[0]  # Last shard might be shorter

            if self.shuffle_dataset:
                shard_indices = np.random.permutation(real_length)
            else:
                shard_indices = np.arange(real_length)

            # Iterate over the initial dates in the current (shuffled) shard
            for init_dates_index in shard_indices:
                sample = {}

                # Get the target dates for the current initial date
                step_offset = np.array(
                    [init_dates[init_dates_index] + k*self.lead_time for k in range(self.n_steps+1)], dtype='datetime64[ns]')
                step_offset = np.isin(required_timesteps, step_offset)
                step_offset = np.where(step_offset)[0]
                sample['dates'] = required_timesteps[step_offset]

                if not self.init_condition_only:
                    sample['initial_condition'] = shard_dataset[step_offset[0],
                                                                self.init_vars_idx]
                    # Forcing not necesssary for last predicted step
                    sample['forcing'] = shard_dataset[np.ix_(
                        step_offset[1:-1], self.forcing_vars_idx)]
                    sample['target'] = shard_dataset[np.ix_(
                        step_offset[1:], self.target_vars_idx)]

                else:
                    # Only load initial conditions
                    # shard dataset contains only initial conditions
                    sample['initial_condition'] = shard_dataset[init_dates_index,
                                                                self.init_vars_idx]

                    # Forcing are kept out of memory for long rollouts and need to be loaded during rollouts
                    sample['forcing'] = self.xr_forcings.sel(
                        time=required_timesteps[step_offset[1:-1]])
                    # Change time dimension to time offset
                    sample['forcing']['time'] = np.array(
                        [k*self.lead_time_hours for k in range(1, self.n_steps)])
                    sample['forcing'] = sample['forcing'].rename(
                        name_dict={'time': 'timedelta'})

                if self.compute_lead_times:
                    sample['lead_times'] = self.lead_time_tensor

                yield sample

    def collate(self, batch):
        """
        Custom collate that can handle xarray datasets
        """
        out = {}
        for k in batch[0].keys():
            if isinstance(batch[0][k], torch.Tensor):
                out[k] = torch.stack([sample[k] for sample in batch], dim=0)
            elif isinstance(batch[0][k], np.ndarray):
                out[k] = np.stack([sample[k] for sample in batch], axis=0)
            elif isinstance(batch[0][k], xr.Dataset):
                out[k] = xr.concat([sample[k] for sample in batch], pd.Index(
                    [b['dates'][0] for b in batch], name="sample"))
            else:
                raise NotImplementedError(
                    f"Collate not implemented for {type(batch[0][k])}")

        return out


class MultistepDataModule(LightningDataModule):
    """
    Data module for loading and preprocessing data for training and evaluation with pytorch lightning.
    """

    def __init__(self,
                 data_dirs: List[str],
                 normalization_dirs: List[str],
                 train_config: MultidirNcDatasetConfig,
                 val_config,  # TODO: Link Arguments in lightning such that interpolation works with cli arguments
                 test_config,
                 batch_size: int,
                 num_workers: int,
                 pin_memory: bool
                 ):
        """

        Args:
            data_dirs: List of paths to .nc files or to directories containing .nc files. Can be a glob pattern.
            normalization_dirs: List of paths to directories containing .nc files. Can be a glob pattern.
            train_config: Arguments for training datast. See MultidirNcDatasetConfig
            val_config: Arguments for validation dataset that override the training dataset
            test_config: Arguments for test dataset that override the training dataset
            batch_size: batch size for training, validation and test dataloaders
            num_workers: number of workers for training, validation and test dataloaders
            pin_memory: whether to pin memory for training, validation and test dataloaders
        """

        super().__init__()

        self.train_config = train_config
        self.val_config = {**train_config, **val_config}
        self.test_config = {**train_config, **test_config}

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.xr_dataset = load_xr_dataset(data_dirs)
        self.normalize_data = load_normalize_data(normalization_dirs)

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage: Optional[str] = None):
        """
        Setup the data module. This is called by pytorch lightning when the module is created.
        """

        # if not self.data_train and not self.data_val and not self.data_test:

        if stage == 'fit':

            print("\nSetup train dataset\n")

            self.data_train = self.initialize_dataset(self.train_config)

            print("\nSetup val dataset\n")

            self.data_val = self.initialize_dataset(self.val_config)

        if stage == 'test':
            print("\nSetup test dataset\n")
            self.data_test = self.initialize_dataset(self.test_config)

        if stage == 'eval':
            print("\nSetup val dataset\n")
            self.data_val = self.initialize_dataset(self.val_config)

    def initialize_dataset(self, config):
        return MultidirNcDataset(
            dataset=self.xr_dataset,
            normalize_data=self.normalize_data,
            **config
        )

    def train_dataloader(self):
        """
        Pytorch lightning
        """
        return self.get_dataloader(self.data_train)

    def val_dataloader(self):
        """
        Pytorch lightning
        """
        return self.get_dataloader(self.data_val)

    def test_dataloader(self):
        """
        Pytorch lightning
        """
        return self.get_dataloader(self.data_test)

    def get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=dataset.collate
        )

    def get_dataset_for_statistics(self) -> MultidirNcDataset:
        """
        Choose the train dataset to set variables in the model module.
        If not available, use the val dataset, then the test dataset.
        """

        if self.data_train is not None:
            return self.data_train
        elif self.data_val is not None:
            return self.data_val
        elif self.data_test is not None:
            return self.data_test
        else:
            raise ValueError("No dataset available")
