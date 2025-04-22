import os
import glob

from typing import Dict
import numpy as np
import pandas as pd
import cftime
import xarray as xr

from typing import List

from utils.lookup_mappings import VAR_TO_CMIP_NAME

def dataset_fixes(dataset:xr.Dataset) -> xr.Dataset:
    '''
    Helper function to load multiple nc files with xr.open_mfdataset and apply some fixes to the dataset.
    Note that we follow CMIP6 conventions for the variable names and dimensions.
    '''
    # ERA fixes
    if 'level' in dataset.dims or 'pressure_level' in dataset.dims:
        if 'level' in dataset.dims:
            dataset = dataset.rename_dims({'level':'plev'})
            dataset = dataset.rename_vars({'level':'plev'})
        if 'pressure_level' in dataset.dims:
            dataset = dataset.rename_dims({'pressure_level':'plev'})
            dataset = dataset.rename_vars({'pressure_level':'plev'})
        dataset['plev'] = dataset['plev'].astype(np.float32)
        dataset['plev'] = dataset['plev'] * 100 # Convert to Pa following CMIP6 conventions

        # Pressure levels should be in increasing order
        dataset = dataset.reindex(plev=dataset['plev'][::-1])

    if "valid_time" in dataset.coords:
        dataset = dataset.rename_dims({'valid_time':'time'})
        dataset = dataset.rename_vars({'valid_time':'time'})
    if 'z' in dataset.data_vars:
        # Geopotential needs to be divided by earth's gravity to obtain geopotential height (zg)
        # (see https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height)
        dataset['z'] = dataset['z'] / 9.80665

    if 'expver' in dataset.coords:
        dataset = dataset.drop_vars('expver')

    if 'number' in dataset.coords:
        dataset = dataset.drop_vars('number')

    if 'latitude' in dataset.coords:
        dataset = dataset.rename_dims({'latitude':'lat'})
        dataset = dataset.rename_vars({'latitude':'lat'})
    
    if 'longitude' in dataset.coords:
        dataset = dataset.rename_dims({'longitude':'lon'})
        dataset = dataset.rename_vars({'longitude':'lon'})

    # CMIP fixes
    if 'height' in dataset.coords: # CMIP6 surface variables have height coordinate which is not needed since it is treated as single level variable
        dataset = dataset.drop_vars('height')

    dataset = dataset.rename_vars({v:VAR_TO_CMIP_NAME.get(v,v) for v in dataset.data_vars})

    return dataset

def normalize_data_fixes(dataset:xr.Dataset) -> xr.Dataset:
    '''
    Helper function to load multiple nc files of the computed normalization files with xr.open_mfdataset and apply some fixes to the dataset.
    Note that we follow CMIP6 conventions for the variable names and dimensions.
    '''

    if 'level' in dataset.dims: # ERA Style level
        dataset = dataset.rename_dims({'level':'plev'})
        dataset = dataset.rename_vars({'level':'plev'})
        dataset['plev'] = dataset['plev'].astype(np.float32)
        dataset['plev'] = dataset['plev'] * 100 # Convert to Pa following CMIP6 conventions

    if 'mean_z' in dataset.data_vars:
        # Geopotential needs to be divided by earth's gravity to obtain geopotential height (zg)
        # (see https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height)
        dataset['mean_z'] = dataset['mean_z'] / 9.80665
        dataset['std_z'] = dataset['std_z'] / 9.80665

    dataset = dataset.rename_vars({v:f"mean_{VAR_TO_CMIP_NAME.get(v.replace('mean_',''),v.replace('mean_',''))}" for v in dataset.data_vars if v.startswith('mean_')})
    dataset = dataset.rename_vars({v:f"std_{VAR_TO_CMIP_NAME.get(v.replace('std_',''),v.replace('std_',''))}" for v in dataset.data_vars if v.startswith('std_')})

    return dataset


def load_normalize_data(normalization_dirs: List[str]) -> xr.Dataset:
    """
    Load normalization data from the given directories.

    """


    print("Load normalization from", normalization_dirs)
    file_paths = []
    for p in normalization_dirs:
        file_paths += glob.glob(os.path.join(os.environ['DATA_DIR'], p,'*.nc'))


    return xr.open_mfdataset(file_paths, combine = 'nested', preprocess=normalize_data_fixes)

def load_xr_dataset(data_dirs: List[str], time_chunk_size = 1095) -> xr.Dataset:
    """
    Load data from the given directories.
    """


    file_paths = []
    print("Load data from", data_dirs)
    for p in data_dirs:
        if p.endswith('.nc'):
            path2add = os.path.join(os.environ['DATA_DIR'], p)
        else:
            path2add = os.path.join(os.environ['DATA_DIR'], p,'*.nc')

        file_paths += glob.glob(path2add)

    # Already chunking when loading results in a much faster loading time if chunks on disk are weird
    dataset =  xr.open_mfdataset(
            file_paths, combine='by_coords', engine='netcdf4', join = "outer", preprocess=dataset_fixes, chunks={"time":time_chunk_size, "lat":-1, "lon":-1, "plev":1})
    
    if 'plev' in dataset.coords:
        dataset = dataset.chunk({'plev':1})


    return dataset

def add_hpa_zeros_to_var_names(var_names):
    new_var_names = {}
    for v, levels in var_names.items():
        new_var_names[v] = []
        if len(levels) > 0:
            for l in levels:
                new_var_names[v].append(l*100)
    return new_var_names

def flatten_var_name_dict(var_names):
    names = []
    for v, levels in var_names.items():
        if len(levels) > 0:
            names.extend([f'{v}_{int(l/100)}' for l in levels])
        else:
            names.append(v)
    return names

def to_var_name_dict(var_names):
    var_dict = {}
    for v in var_names:
        var_name, level = get_var_name_and_level(v)
        if var_name not in var_dict:
            var_dict[var_name] = []
        if level is not None:
            var_dict[var_name].append(level)
    return var_dict

def get_var_name_and_level(var_name):
    parts = var_name.split('_')
    if len(parts) == 1:
        return var_name, None
    else:
        
        if parts[-1].isdigit():
            return '_'.join(parts[:-1]), int(parts[-1])*100
        else:
            return var_name, None
        
def var_names_to_CMIP(var_names:dict):
    return {VAR_TO_CMIP_NAME.get(k,k): v for k,v in var_names.items()}

def datetime_to_cftime(dates, kwargs={}):
    dates = pd.DatetimeIndex(dates)
    return [
        cftime.datetime(
            date.year,
            date.month,
            date.day,
            date.hour,
            date.minute,
            date.second,
            date.microsecond,
            **kwargs
        )
        for date in dates
    ]


def print_normalization_xr(normalize_data):
    for v in normalize_data.data_vars:
        if 'level' in normalize_data[v].dims:
            for l in normalize_data.level:
                print(str(v), l.values.item(), normalize_data[v].sel(level = l).values.item())
        else:
            print(v, normalize_data[v].values)


def xr_variables_4dim_as_arrays(dataset:xr.Dataset, var_names:Dict):
    '''
    var_names: as dict with levels
    dataset: time,plev,lat,lon
    '''

    return [dataset[c].sel(plev=level).compute(scheduler='single-threaded').values if len(
            level) > 0 else dataset[c].compute(scheduler='single-threaded').values[:, None, :, :] for c, level in var_names.items()]

def xr_variables_3dim_as_arrays(dataset:xr.Dataset, var_names:Dict):
    '''
    var_names: as dict with levels
    dataset: plev,lat,lon
    '''
    return [dataset[c].sel(plev=level).compute(scheduler='single-threaded').values if len(
            level) > 0 else dataset[c].compute(scheduler='single-threaded').values[None, :, :] for c, level in var_names.items()]

def get_required_levels(var_names:Dict):
    plev = []
    for v, levels in var_names.items():
        plev.extend([float(l) for l in levels])
    return sorted(list(set(plev)))
