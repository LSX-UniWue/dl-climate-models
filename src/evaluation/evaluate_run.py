"""
Compute plots and metrics for a rollout computed with the autoregressive_rollout.py script.
"""

import xarray
import pandas as pd
import os
import numpy as np
import glob
import yaml

import argparse

from datamodule import load_xr_dataset, load_normalize_data
from utils.data_utils import add_hpa_zeros_to_var_names
from utils.eval_utils import (
    per_variable_plot,
    plot_global_means,
    plot_temporal_diff_map,
    timed_print
)

def time_mean(data):
    return data.mean(dim = ['time'])

def time_std(data):
    return data.std(dim = ['time'])

def _compute_metrics(prediction, target, normalize_data, var_dict, aggregation_statistics):

    w_lat = np.cos(np.deg2rad(prediction.lat.values))
    w_lat = w_lat / w_lat.mean()

    results = []

    for aggregate_func in aggregation_statistics:
        
        agg_prediction = eval(aggregate_func)(prediction)
        agg_target = eval(aggregate_func)(target)

    
        for v, levels in var_dict.items():
            if len(levels) > 0:

                variable_data = {f'{v}_{int(l/100)}': (agg_prediction[v].sel(plev=l),
                                                       agg_target[v].sel(plev = l) if target is not None else None,
                                                       normalize_data[f'mean_{v}'].sel(plev=l).values.item() if normalize_data is not None else None,
                                                       normalize_data[f'std_{v}'].sel(plev=l).values.item() if normalize_data is not None else None)
                                for l in levels}
            else:
                variable_data = {v: (agg_prediction[v],
                                     agg_target[v] if target is not None else None,
                                     normalize_data[f'mean_{v}'].values.item() if normalize_data is not None else None,
                                     normalize_data[f'std_{v}'].values.item() if normalize_data is not None else None)}

            for v_name,(pred_var,target_var, mean, std) in variable_data.items():

                pred_var = pred_var.values.squeeze()
                target_var = target_var.values.squeeze()

                diff = pred_var - target_var

                rmse = np.sqrt(np.mean((diff*w_lat[:,None])**2))
                mae = np.abs(diff*w_lat[:,None]).mean()


                current_metrics = {'var': v_name, 'stat':aggregate_func,'rmse': rmse, 'mae': mae}

                if normalize_data is not None:
                    norm_diff = (diff - mean) / std

                    norm_rmse = np.sqrt(np.mean((norm_diff*w_lat[:,None])**2))
                    norm_mae = np.abs(norm_diff*w_lat[:,None]).mean()

                    current_metrics['norm-rmse'] = norm_rmse
                    current_metrics['norm-mae'] = norm_mae

                results.append(current_metrics)

    results = pd.DataFrame(results)

    return results


def main(run_path, eval_folder,ref_dataset, normalization_path, start_date,end_date, time_chunks):

    print('\nStart Evaluation\n')

    eval_paths = glob.glob(os.path.join(os.environ['RESULTS_DIR'],run_path, eval_folder,'out.nc'))



    with open(os.path.join(os.environ['RESULTS_DIR'], run_path, 'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)

    out_var_dict = add_hpa_zeros_to_var_names({**config['data']['train_config']['prognostic_vars'],**config['data']['train_config']['diagnostic_vars']})

    timed_print("EVAL PATHS", eval_paths)

    if len(ref_dataset) >0:
        timed_print("Open ref data")
        ref_data = load_xr_dataset(ref_dataset,time_chunk_size=time_chunks)
        ref_data = ref_data.reindex(lat=list(reversed(ref_data.lat)))

        if len(normalization_path) > 0:
            normalize_data = load_normalize_data(normalization_path)


    
    # Load the data
    data = xarray.open_mfdataset(eval_paths)

    data = data.chunk({'init_condition':1, 'timedelta':time_chunks, 'lat':-1, 'lon':-1,'plev':1})


    # Check latitudes from positives to negatives
    if data.lat[-1] > data.lat[0]:
        data = data.reindex(lat=list(reversed(data.lat)))


    for i,init_date in enumerate(data['init_condition']):
        trajectory = data.sel(init_condition = init_date)

        init_date = pd.to_datetime(init_date.values.item())

        timed_print("INIT DATE",init_date)
        
        timestamps = pd.to_datetime(init_date + pd.to_timedelta(data.timedelta.values, unit='h'))
        trajectory['timedelta'] = timestamps
        trajectory = trajectory.rename_dims({'timedelta':'time'})
        trajectory = trajectory.rename_vars({'timedelta':'time'})


        timed_print("Load eval data")
        if start_date is not None and end_date is not None:
            trajectory = trajectory.sel(time = slice(start_date,end_date))

        timed_print("Load ref data")
        ref_data_trajectory = ref_data[list(trajectory.data_vars)].sel(time = trajectory.time.values, plev = trajectory.plev) if len(ref_dataset) >0 else None

        # get dir of file
        folder = os.path.dirname(eval_folder)
        trajectory_save_dir = os.path.join(os.environ['RESULTS_DIR'], run_path, folder, str(init_date.isoformat()).replace('T',' '))
        os.makedirs(trajectory_save_dir, exist_ok=True)

        timed_print("Save path", trajectory_save_dir)

        timed_print("Plot Maps")

        save_path = os.path.join(trajectory_save_dir, 'maps.pdf')
        per_variable_plot(trajectory, ref_data_trajectory, out_var_dict, plot_temporal_diff_map, save_path,  n_col=4, hspace = 0.4, height = 3, width = 4, share_labels = {'y'}, plot_coastlines = True, title = '')

        timed_print("Plot Time Series")
        save_path = os.path.join(trajectory_save_dir, 'global_means.pdf')
        per_variable_plot(trajectory, ref_data_trajectory, out_var_dict, plot_global_means, save_path, n_col=4, height = 2, width = 3, share_labels = {'x'} , time_stamps = trajectory.time.values)

        if len(ref_dataset) >0:
            timed_print("Compute Metrics")
            save_path = os.path.join(trajectory_save_dir, 'global_means.pdf')
            metrics_df = _compute_metrics(trajectory, ref_data_trajectory, normalize_data, out_var_dict,['time_mean', 'time_std'])
            metrics_df['init_date'] = init_date.isoformat()
            metrics_df['run_name'] = os.path.basename(os.path.normpath(run_path))
            metrics_df.to_csv(os.path.join(trajectory_save_dir, 'metrics.csv'))
   
        timed_print("Resample monthly")
        trajectory_monthly = trajectory.resample(time="1ME").mean()

        ref_data_trajectory_monthly = ref_data_trajectory.resample(time="1ME").mean() if len(ref_dataset) >0 else None

        
        timed_print("Plot Monthly Time Series")
        save_path = os.path.join(trajectory_save_dir, 'global_means_monthly.pdf')
        per_variable_plot(trajectory_monthly, ref_data_trajectory_monthly, out_var_dict, plot_global_means, save_path, n_col=4, height = 2, width = 3, share_labels = {'x'}, time_stamps = trajectory_monthly.time.values)

        timed_print("Resample yearly")
        trajectory_monthly = trajectory.resample(time="1YE").mean()
        ref_data_trajectory_monthly = ref_data_trajectory.resample(time="1YE").mean() if len(ref_dataset) >0 else None

        timed_print("Plot Yearly Time Series")
        save_path = os.path.join(trajectory_save_dir, 'global_means_yearly.pdf')
        per_variable_plot(trajectory_monthly, ref_data_trajectory_monthly, out_var_dict, plot_global_means, save_path, n_col=4, height = 2, width = 3, share_labels = {'x'}, time_stamps = trajectory_monthly.time.values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_path', type=str, required=True, help="Relative Path to the run folder to RESULTS_DIR")
    parser.add_argument('--eval_folder', type=str, required=True,help='Relative Path from the run to the eval folder that contain the out.nc file')
    parser.add_argument('--ref_dataset', type=str,  nargs='+', default = [])
    parser.add_argument('--normalization_path', type=str, nargs='+', default = [])
    parser.add_argument('--start_date', type=str, default = None)
    parser.add_argument('--end_date', type=str, default = None)
    parser.add_argument('--time_chunks', type=int, default = 720)

    args = parser.parse_args()

    args = vars(args)

    

    main(**args)

    