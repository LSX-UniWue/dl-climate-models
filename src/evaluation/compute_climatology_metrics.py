'''
Script to pre-compute the climatology metrics which can be included into evaluation plots
of hyperparameter searches (see paper for details).

'''

from datamodule import load_xr_dataset, load_normalize_data
from utils.data_utils import add_hpa_zeros_to_var_names

from utils.eval_utils import (
    timed_print
)

import os
import argparse
import numpy as np
import pandas as pd

def time_mean(data):
    return data.mean(dim = ['time'])

def time_std(data):
    return data.std(dim = ['time'])

def compute_metrics(eval_period, ref_data_eval, metrics, run_name, var_str, stat_str, date_str, normalize_data_dict):

    w_lat = np.cos(np.deg2rad(eval_period.lat.values))
    w_lat = w_lat / w_lat.mean()
    
    eval_period = eval_period.values.squeeze()

    current_metrics = {'run_name':run_name, 'var':var_str, 'init_date': date_str, 'stat': stat_str}

    if ref_data_eval is not None:
        ref_data_eval = ref_data_eval.values.squeeze()

        ref_diff = eval_period - ref_data_eval

        mae = np.abs(w_lat[:,None]*ref_diff).mean()
        rmse = np.sqrt(np.square(w_lat[:,None]*ref_diff).mean())

        current_metrics['mae'] = mae
        current_metrics['rmse'] = rmse

        if normalize_data_dict is not None:
            norm_eval_period = (eval_period-normalize_data_dict[f'mean_{var_str}'])/normalize_data_dict[f'std_{var_str}']
            norm_ref_data_eval = (ref_data_eval-normalize_data_dict[f'mean_{var_str}'])/normalize_data_dict[f'std_{var_str}']

            norm_ref_diff = norm_eval_period - norm_ref_data_eval
            norm_mae = np.abs(w_lat[:,None]*norm_ref_diff).mean()
            norm_rmse = np.sqrt(np.square(w_lat[:,None]*norm_ref_diff).mean())

            current_metrics['norm-mae'] = norm_mae
            current_metrics['norm-rmse'] = norm_rmse


    metrics.append(current_metrics)

def main(save_path, climatology_start_year, climatology_end_year, init_conditions_eval, forecast_length, ref_data_path, normalize_data_dirs, statistics):


    variables = {
        'tas': [],
        'uas': [],
        'vas': [],
        'ta': [250, 300, 500 , 600, 700, 850, 925, 1000],
        'zg': [250, 300, 500 , 600, 700, 850, 925, 1000],
        'ua': [250, 300, 500 , 600, 700, 850, 925, 1000],
        'va': [250, 300, 500 , 600, 700, 850, 925, 1000],
        'hus': [250, 300, 500 , 600, 700, 850, 925, 1000]
    }

    ds = load_xr_dataset([ref_data_path],-1)
    normalize_data = load_normalize_data(normalize_data_dirs)

    normalize_data_dict = {}
    for v in normalize_data.data_vars:
        if 'plev' in normalize_data[v].dims:
            for l in normalize_data[v].plev.values:
                normalize_data_dict[f'{v}_{l}'] = normalize_data[v].sel(plev=l).values.item()
        else:
            normalize_data_dict[v] = normalize_data[v].values.item()

    # Get every 6 hours
    ds = ds.sel(time=ds.time.dt.hour.isin([0,6,12,18]))


    variables = add_hpa_zeros_to_var_names(variables)
    ds = ds[list(variables.keys())]


    levels = [tuple(l) for l in variables.values() if len(l) > 0]
    assert np.all(np.array(levels) == levels[0]), "All forcing variables must have the same levels"
    if len(levels) > 0:
        ds = ds.sel(plev=list(levels[0]))


    climatology = ds.sel(time=slice(str(climatology_start_year), str(climatology_end_year)))
    

    timed_print("Computing climatology metrics")
    metrics = []
    for init in init_conditions_eval:
        end_forecast =  pd.to_datetime(init)+pd.to_timedelta(forecast_length)
        timed_print("Forecast range", init, end_forecast)
        eval_period = ds.sel(time=slice(init, end_forecast))
        
        date_str = str(pd.to_datetime(pd.to_datetime(init).value).isoformat()).replace('T',' ')
        for s in statistics:
            timed_print("Computing statistics")
            stat_climatology = eval(s)(climatology).compute()
            stat_eval_period = eval(s)(eval_period).compute()
            for v in eval_period.data_vars:
                timed_print(f"Computing {v} {s} {date_str}")
                if 'plev' in eval_period[v].dims:
                    for level in eval_period['plev']:
                        var_str = f'{v}_{level.values.item()}'
                        compute_metrics(stat_climatology[v].sel(plev=level), stat_eval_period[v].sel(plev=level), metrics, 'climatology', var_str, s, date_str, normalize_data_dict)

                else:
                    var_str = f'{v}'
                    compute_metrics(stat_climatology[v], stat_eval_period[v], metrics, 'climatology', var_str, s, date_str, normalize_data_dict)

        path = os.path.join(os.environ['RESULTS_DIR'], save_path,forecast_length, init)
        os.makedirs(path, exist_ok=True)
        pd.DataFrame(metrics).to_csv(os.path.join(path,'metrics.csv'))
        timed_print("Metrics saved to", path)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute climatology metrics')
    parser.add_argument('--save_path', type=str, default='dlclim/runs/climatology_forecast/evaluation', help='Path to save the metrics')
    parser.add_argument('--climatology_start_year', type=int, default=1979, help='Start year of climatology period')
    parser.add_argument('--climatology_end_year', type=int, default=2007, help='End year of climatology period')
    parser.add_argument('--init_conditions_eval', type=str, nargs='+', default=['2009-01-01T00'], help='Start date of evaluation period')
    parser.add_argument('--forecast_length', type=str, default='3651d', help='End date of evaluation period')
    parser.add_argument('--ref_data_path', type=str, default='ERA5/weatherbench1/r64x32/*/', help='Path to reference data')
    parser.add_argument('--normalize_data_dirs', type=str, nargs='+', default=[
        'ERA5/weatherbench1/r64x32/2m_temperature/normalization/1979_2008/',
        'ERA5/weatherbench1/r64x32/10m_u_component_of_wind/normalization/1979_2008/',
        'ERA5/weatherbench1/r64x32/10m_v_component_of_wind/normalization/1979_2008/',
        'ERA5/weatherbench1/r64x32/constants/normalization/1979_2008/',
        'ERA5/weatherbench1/r64x32/geopotential/normalization/1979_2008/',
        'ERA5/weatherbench1/r64x32/specific_humidity/normalization/1979_2008/',
        'ERA5/weatherbench1/r64x32/temperature/normalization/1979_2008/',
        'ERA5/weatherbench1/r64x32/u_component_of_wind/normalization/1979_2008/',
        'ERA5/weatherbench1/r64x32/v_component_of_wind/normalization/1979_2008/',
        'CMIP/CMIP6/inputs4MIP/historical/solar/r64x32/1hr/tisr/normalization/1979_2008/' 
    ], help='Path to normalization data directories')
    parser.add_argument('--statistics', type=str, nargs='+', default=['time_mean', 'time_std'], help='Statistics to compute')
    args = parser.parse_args()

    args = vars(args)

    main(**args)

    