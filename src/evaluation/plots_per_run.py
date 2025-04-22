'''
This script creates a figure with multiple subplots, one for each run.
Makes comparison of plots between runs easier.
'''

import xarray
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import math
import cartopy.crs as ccrs

import argparse

from utils.data_utils import dataset_fixes

from utils.eval_utils import (
    plot_global_means,
    plot_temporal_diff_map,

)

MODEL_NAME_DICT = {
    'climax': 'ClimaX',
    'sfno': 'SFNO',
    'fourcastnet': 'FourCastNet'
}

def run_name_to_title(run_name):
    run_name_list = run_name.split('_')
    model_name = run_name_list[0]
    num_vars = run_name_list[1].replace('vars','')
    num_steps = run_name_list[2].replace('steps','')
    num_layers = run_name_list[3].replace('layers','')
    dim = run_name_list[4].replace('dim','')
    seed = run_name_list[5].replace('seed','')

    title = f"{MODEL_NAME_DICT[model_name]}\n#Vars: {num_vars} , #Steps: {num_steps}, #Layers: {num_layers}, Dim: {dim}, Seeed {seed}"

    return title


def main(run_paths, exp_name, plot_fn, ref_dataset, variable, level, start_date,end_date, n_cols, save_dir):

    print('\nStart to create plots per run\n')

    file_paths = []
    exp_name_wout_date = os.sep.join(exp_name.split(os.sep)[:-1])
    for f in run_paths:
        print(os.path.join(os.environ['RESULTS_DIR'],f,exp_name_wout_date,'out.nc'))
        file_paths += glob.glob(os.path.join(os.environ['RESULTS_DIR'],f,exp_name_wout_date,'out.nc'))

    print("File paths", file_paths)

    
    n_cols = min(n_cols,len(file_paths))
    n_rows = math.ceil(len(file_paths)/n_cols)

    if plot_fn == 'plot_global_means':
        fig, ax = plt.subplots(n_rows,n_cols,figsize=(6*n_cols,3*n_rows))
        fig.subplots_adjust(hspace=0.8)
        fig.subplots_adjust(wspace=0.2)
    elif plot_fn == 'plot_temporal_diff_map':
        fig, ax = plt.subplots(n_rows,n_cols,figsize=(6*n_cols,4*n_rows),subplot_kw={'projection': ccrs.PlateCarree()})
        fig.subplots_adjust(hspace=0.4)
    
    if ref_dataset is not None:
        print("Open ref data")
        print(os.path.join(os.environ['DATA_DIR'],ref_dataset))
        ref_data = xarray.open_mfdataset(os.path.join(os.environ['DATA_DIR'],ref_dataset), preprocess=dataset_fixes, chunks={'time': 1460, 'plev': 1})

        if level is not None:
            ref_data = ref_data.sel(plev = level)[variable]
        else:
            ref_data = ref_data[variable]

        ref_data = ref_data.reindex(lat=list(reversed(ref_data.lat)))


    for i, f in enumerate(file_paths):

        f = os.path.join(os.environ['RESULTS_DIR'],f)

        # Load the data
        print("Open dataset", i)
        data = xarray.open_dataset(f)
        data = data.reindex(lat=list(reversed(data.lat)))

        if level is not None:
            data = ref_data.data(plev = level)[variable]
        else:
            data = data[variable]


        init_date = pd.to_datetime(exp_name.split(os.sep)[-1])
        trajectory = data.sel(init_condition = init_date)

       

        print("INIT DATE",init_date)
        
        timestamps = pd.to_datetime(init_date + pd.to_timedelta(data.timedelta.values, unit='h'))
        trajectory['timedelta'] = timestamps
        trajectory = trajectory.rename({'timedelta':'time'})


        print("Load eval data")
        if start_date is not None and end_date is not None:
            trajectory = trajectory.sel(time = slice(start_date,end_date))



        if i == 0:
            print("Load ref data")
            ref_data_trajectory = ref_data.sel(time = trajectory.time.values) if ref_dataset is not None else None


        current_row = i//n_cols
        current_col = i%n_cols
        if n_rows == 1 and n_cols == 1:
            current_ax = ax
        elif n_rows == 1:                                                                                                                                                                        
            current_ax = ax[i]
        elif n_cols == 1:
            current_ax = ax[i]
        else:
            current_ax = ax[(current_row,current_col)]

        run_name = os.path.normpath(f.replace(os.path.join(exp_name_wout_date,'out.nc'),"")).split(os.sep)[-1]
        print(run_name)

        var_name = variable if level is None else f"{variable}_{level}"

        if plot_fn == 'plot_global_means':
            plot_global_means(trajectory, ref_data_trajectory, var_name, current_ax, trajectory.time.values, title=run_name_to_title(run_name))
        elif plot_fn == 'plot_temporal_diff_map':
            plot_temporal_diff_map(trajectory, ref_data_trajectory, var_name, current_ax, plot_coastlines=True, title=run_name_to_title(run_name))


    if plot_fn == 'plot_global_means':
        handles, labels = current_ax.get_legend_handles_labels()
        fig.legend(handles, labels,loc='upper center', bbox_to_anchor=(0.5, -0.03))

    if len(file_paths)>1:
        # delete empty subplots
        for j in range(i+1, n_rows*n_cols):
            fig.delaxes(ax.flatten()[j])

    # add_headers(fig,x_label=current_ax.get_xlabel())

    # get dir of file
    if save_dir is None:
        # This is a bit cumbersome
        s = '/'.join(os.path.normpath(run_paths[0]).split(os.sep)[:-1])
        save_dir = os.path.join(s,'plots', os.path.normpath(run_paths[0]).split(os.sep)[-1], exp_name)

    trajectory_save_dir = os.path.join(os.environ['RESULTS_DIR'],save_dir)
    os.makedirs(trajectory_save_dir, exist_ok=True)

    print("Save figure in", trajectory_save_dir)

    fig.savefig(os.path.join(trajectory_save_dir,f'{plot_fn}.pdf'),dpi = 300, bbox_inches="tight")
    plt.close()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_paths', type=str, nargs='+', help='Paths to the runs to evaluate. Can contain glob patterns. Must be relative to $RESULTS_DIR', required=True)
    parser.add_argument('--exp_name', type=str, help='relative paths to the run path, which contains the out.nc file created with the autoregressive_rollout.py script')
    parser.add_argument('--plot_fn', type=str, default = 'plot_temporal_diff_map',help='Plot function to create a plot per run')
    parser.add_argument('--ref_dataset', type=str, default = 'ERA5/weatherbench1/r64x32/*/*.nc',help='Path to dataset which is used as target in plot function')
    parser.add_argument('--variable', type=str, default = 'tas',help='Variable to plot. For non surface variables make sure to set the level')
    parser.add_argument('--level', type=int, default = None, help='Level to plot. Set to None fur surface variables')
    parser.add_argument('--start_date', type=str, default = None, help= 'Subset the data to a specific time range. Format: YYYY-MM-DD. If None, the whole time range of the selected run is used')
    parser.add_argument('--end_date', type=str, default = None)
    parser.add_argument('--n_cols', type=int, default = 1, help='Number of columns in the plot. If more than 2, the plot is a grid with n_cols columns')
    parser.add_argument('--save_dir', type=str, default = None, help='Path to save the plots. If None, will be saved in the same directory as the run paths. If multiple run paths are provided, plots are in the last one')

    args = parser.parse_args()

    args = vars(args)

    main(**args)

    