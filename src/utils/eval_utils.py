
from utils.lookup_mappings import VAR_NAMES_TO_UNIT, VAR_TO_CMIP_NAME
from utils.data_utils import get_var_name_and_level

# imports are needed to create a model instance before loading the weights
import models.climax
import models.fourcastnet
import models.sfno

import numpy as np
import pandas as pd
import xarray as xr
import math
import torch


from time import time
from functools import wraps
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib
matplotlib.use('Agg') # having tkinter error, this fixes i
import matplotlib.pyplot as plt

import matplotlib.colors as colors
import os
import datetime
from typing import Callable, Dict, List, Tuple


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'Function {f.__name__} took {te-ts:2.4f} seconds')
        return result
    return wrap

def timed_print(*args, **kwargs):
    print(f'[{datetime.datetime.now().strftime("%H:%M:%S")}] ', end='')
    print(*args, **kwargs)

def load_model(config, checkpoint_path):
    model = eval(config['model']['net']['class_path'])(
        **config['model']['net']['init_args'])

    state_dict = torch.load(checkpoint_path, map_location=torch.device(
        'cpu'), weights_only=True)['state_dict']

    for k in list(state_dict.keys()):

        if k.startswith("net."):
            state_dict[k.replace("net.", "")] = state_dict[k]
            del state_dict[k]
        else:
            del state_dict[k]

    msg = model.load_state_dict(state_dict)
    print(msg)

    return model


def get_checkpoint_path_from_config(config):
    for callback_dict in config['trainer']['callbacks']:
        if callback_dict['class_path'] == 'pytorch_lightning.callbacks.ModelCheckpoint':
            checkpoint_dir = callback_dict['init_args']['dirpath']
            checkpoint_list = [f for f in os.listdir(
                checkpoint_dir) if f.startswith('best_epoch_')]
            assert len(checkpoint_list) == 1, "Multiple best epochs found"
            checkpoint_file = checkpoint_list.pop(0)  # Get the best epoch

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

    return checkpoint_path

def per_variable_plot(prediction:xr.Dataset, target:xr.Dataset, var_dict: Dict[str, List[int]], ax_plot_func:Callable, save_path:str, n_col = 4, hspace = 0.2, wspace= 0.2, height = 6, width = 8, share_labels = {}, **kwargs):
    '''
    Make a plot with a subplot for each variable in var_dict for the provided ax_plot_func.
    '''

    # count variables + levels
    num_vars = sum([max(1,len(level)) for level in var_dict.values()])

    num_rows = math.ceil(num_vars/n_col)
    if 'plot_coastlines' in kwargs:
        fig,ax = plt.subplots(num_rows,n_col,figsize=(width*n_col,height*num_rows),subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig,ax = plt.subplots(num_rows,n_col,figsize=(width*n_col,height*num_rows))

    plot_number = 0
    for v, levels in var_dict.items():
        
        if len(levels) > 0:

            plot_data = {f'{v}_{int(l/100)}': (prediction[v].sel(plev=l), target[v].sel(plev = l) if target is not None else None)
                            for l in levels}
        else:
            plot_data = {v: (prediction[v], target[v] if target is not None else None)}

        for v_name,(plot_pred,plot_target) in plot_data.items():
            timed_print("Plotting variable",v_name)
            if num_rows == 1:
                current_ax = ax[plot_number%n_col]
            else:
                current_ax = ax[plot_number//n_col,plot_number%n_col]
            ax_plot_func(prediction = plot_pred,
                        target = plot_target, 
                        var_name = v_name,
                        ax = current_ax, **kwargs)  
            plot_number += 1
    
    handles, labels = current_ax.get_legend_handles_labels()

    fig.legend(handles, labels,loc='upper center', bbox_to_anchor=(0.5, -0.01))

    # delete empty subplots
    for i in range(plot_number, num_rows*4):
        fig.delaxes(ax.flatten()[i])

    # Set font size


    if 'x' in share_labels:
        add_headers(fig,x_label=current_ax.get_xlabel())
    
    if 'y' in share_labels:
        add_headers(fig,y_label=current_ax.get_ylabel())
    plt.rc('font', size=int(num_vars/2)) 

    # fig.subplots_adjust(hspace=hspace)
    # fig.subplots_adjust(wspace=wspace)

    if save_path is not None:
        
        fig.savefig(save_path,dpi = 300, bbox_inches="tight")
        plt.close()


def plot_global_means(prediction: xr.Dataset, target: xr.Dataset, var_name:str, ax, time_stamps, title = None):
    '''
    Time series plot of the global mean of a variable.
    '''
    if target is not None:
        assert prediction.shape == target.shape, f"Prediction and target shapes do not match: {prediction.shape} and {target.shape}"

    x_axis = ((time_stamps - time_stamps[0]) / np.timedelta64(24,'h'))

    w_lat = np.cos(np.deg2rad(prediction.lat.values))
    w_lat = w_lat / w_lat.mean()

    w_lat_arr = xr.DataArray(w_lat, dims = ['lat'])

    weighted_prediction = prediction*w_lat_arr
    
    mean_pred = weighted_prediction.mean(dim=['lat','lon']).values

    ax.plot(x_axis, mean_pred, label=f'Prediction',alpha=0.7 if target is not None else 1)


    if target is not None:
        weighted_target = target*w_lat_arr
        mean_target = weighted_target.mean(dim=['lat','lon']).values
        std_target = weighted_target.std(dim=['lat','lon']).values
        target_neg_std = mean_target-std_target
        target_pos_std = mean_target+std_target

        

        ax.plot(x_axis, mean_target, label=f'Target', alpha=0.7)


        # Compute temporal rmse
        temp_diff = prediction.mean(dim=['time']).values - target.mean(dim=['time']).values

        rmse = np.sqrt(np.mean((temp_diff*w_lat[:,None])**2))


    if title is not None:
        ax.set_title(f'{title}\nRMSE: {rmse:.2e}' if target is not None else f'{title}')


    ax.set_xlabel('Days from initialization')

    name, level = get_var_name_and_level(var_name)
    ax.set_ylabel(f' {var_name} [{VAR_NAMES_TO_UNIT[name]}]')


def plot_temporal_diff_map(prediction:xr.Dataset, target: xr. Dataset, var_name:str, ax, plot_coastlines = False, title = None):
    '''
    Plot the difference between prediction and target on a map and set title as RMSE
    '''

    if plot_coastlines:
        ax.add_feature(cfeature.COASTLINE)
        a, b = np.meshgrid(prediction.lon.values, prediction.lat.values)

    w_lat = np.cos(np.deg2rad(prediction.lat.values))
    w_lat = w_lat / w_lat.mean()

    prediction = prediction.mean(dim='time')

    if target is None:
        plt.set_cmap('plasma')

        if plot_coastlines:
            im = ax.imshow(prediction, transform= ccrs.PlateCarree(), origin='lower', extent=[0, 360, 90, -90])
        else:
            im = ax.imshow(prediction)
       
    else:
        plt.set_cmap('bwr')
        target = target.mean(dim='time').values.squeeze()

        diff = prediction.values.squeeze()-target


        # Colorbar always centered at 0
        vmin = min(diff.min(),0)
        vmax = max(diff.max(),0)
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        if plot_coastlines:
            im = ax.scatter(a, b, c=diff, norm=norm)
        else:
            im = ax.imshow(diff, norm=norm)


        rmse = np.sqrt(np.mean((diff*w_lat[:,None])**2))

    if title is not None:
        ax.set_title(f'{title}\nRMSE: {rmse:.2e}' if target is not None else f'{title}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    name, level = get_var_name_and_level(var_name)

    plt.colorbar(im, ax=ax, orientation='horizontal', label= f'{var_name} [{VAR_NAMES_TO_UNIT[name]}]')
        

def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    x_label=None,
    y_label=None,
    x_label_pad=5,
    **text_kwargs
):
    '''
    Make pretty headers and axes description for plots containing subplots.
    '''
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # get number of rows and cols
        n_rows, n_cols, start, stop = sbs.get_geometry()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                fontweight='bold',
                **text_kwargs,
            )
        if sbs.is_first_col() and y_label is not None:
            ax.set_ylabel(y_label)
        elif y_label is not None:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                fontweight='bold',
                **text_kwargs,
            )

            if y_label is not None:
                ax.set_ylabel(y_label)

        # If there are delete plots in the last row, add x_label in the second last row
        if len(axes) != n_cols*n_rows:
            if start>= (len(axes) - n_cols) and x_label is not None:
                ax.set_xlabel(x_label, labelpad=x_label_pad)
            elif x_label is not None:
                ax.set_xlabel('')
                ax.set_xticklabels([])
        else:
            if sbs.is_last_row() and x_label is not None:
                ax.set_xlabel(x_label, labelpad=x_label_pad)
            elif x_label is not None:
                ax.set_xlabel('')
                ax.set_xticklabels([])

