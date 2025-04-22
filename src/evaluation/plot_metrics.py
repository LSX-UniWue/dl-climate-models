'''
This script plots the mean and standard deviation of the metrics of multiple runs
(e.g. results of hyperparameter search) that were repeated with different seeds.
'''

import argparse
import pandas as pd
from collections import defaultdict
import os
import glob
import re

import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np


Y_LABEL_DICT = {
    'eval-norm-mae': 'Normalized MAE',
    'eval-norm-rmse': 'Normalized RMSE',
    'eval-mae': 'MAE',
    'eval-rmse': 'RMSE'
}

MODEL_NAME_DICT = {
    'climax': 'ClimaX',
    'sfno': 'SFNO',
    'fourcastnet': 'FourCastNet'
}


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
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

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

        if not sbs.is_first_col():
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

        if sbs.is_last_row() and x_label is not None:
            ax.set_xlabel(x_label, labelpad=x_label_pad)


def add_xaxis_description(
    fig
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()
        # Putting headers on cols
        if sbs.is_last_col():
            ax.annotate(
                'Config',
                xy=(1, 0),
                xytext=(10, -14),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="left",
                va="baseline",
            )

            ax.annotate(
                '# Runs',
                xy=(1, 0),
                xytext=(10, -31),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="left",
                va="baseline",
            )


def plot_metrics_across_variables(metrics_df, plots, ax_position, metrics2plot, log_scale, ylim, climatolgy_metrics):

    config_params = ['model', 'train_vars', 'n_steps', 'n_layers', 'dim']

    # group by model and seed
    grouped = metrics_df.groupby(config_params+['init_date', 'stat', 'seed'])

    if climatolgy_metrics is not None:
        clim = climatolgy_metrics.copy()

        clim = clim.groupby(['stat'])

    print(metrics_df.head())

    for metric in metrics2plot:

        # Mean over all variables
        metric_grouped = grouped[metric].mean().reset_index()

        if climatolgy_metrics is not None:
            clim_grouped = clim[metric].mean().reset_index()

        # Seeds as list
        metric_grouped = metric_grouped.groupby(
            config_params+['stat'])[metric].apply(list).reset_index()

        # Join the columns of params to a single string and add it as column
        metric_grouped['config_name'] = metric_grouped[config_params].apply(
            lambda x: '_'.join(x), axis=1)

        for s in metric_grouped['stat'].unique():

            fig, ax = plots[metric][s]
            plt.figure(fig.number)

            if ax_position is None:
                current_ax = ax
            else:
                current_ax = ax[ax_position]

            data_rows = metric_grouped[(metric_grouped['stat'] == s)]
            data_rows = data_rows.sort_values('dim')

            unique_run_names = data_rows['config_name'].unique()
            assert (len(unique_run_names) == len(data_rows))

            color_every = 3

            # Choose colormaps
            green_cmap = matplotlib.colormaps.get_cmap('Greens')
            blue_cmap = matplotlib.colormaps.get_cmap('Blues')
            red_cmap = matplotlib.colormaps.get_cmap('Purples')

            # Get 3 shades from each colormap (evenly spaced)
            greens = [green_cmap(i) for i in np.linspace(
                0.2, 1, color_every+2)[1:-1]][::-1]
            blues = [blue_cmap(i) for i in np.linspace(
                0.2, 1, color_every+2)[1:-1]][::-1]
            reds = [red_cmap(i) for i in np.linspace(
                0.2, 1, color_every+2)[1:-1:]][::-1]
            colors = greens + blues + reds

            values = []
            non_nan = []
            for i, name in enumerate(unique_run_names):
                run_data = data_rows[data_rows['config_name'] == name]

                metric_data = np.array(run_data[metric].values[0])

                # non count nan in list
                is_finite = np.isfinite(metric_data)
                non_nan_count = np.count_nonzero(is_finite)
                visible = np.count_nonzero(metric_data <= ylim)

                values.extend(list(metric_data[is_finite]))
                non_nan.append(f'{visible}/{metric_data.shape[0]}')

                current_ax.scatter(
                    [i]*non_nan_count, metric_data[is_finite], alpha=0.3, marker='x', color=colors[i])

                current_ax.errorbar(i, np.mean(metric_data[is_finite]), yerr=np.std(metric_data[is_finite]), fmt='o', capsize=5,
                                    label=f"Config {i}: [#Steps: {run_data['n_steps'].values[0]} #Layers: {run_data['n_layers'].values[0]} Dim: {run_data['dim'].values[0]}]", color=colors[i])

            if climatolgy_metrics is not None:
                clim_value = clim_grouped[(
                    clim_grouped['stat'] == s)][metric].values[0]
                # plot climatoloy as horizontal line, thin
                current_ax.axhline(
                    y=clim_value, color='r', linestyle='--', label='Climatology', linewidth=0.5)

            current_ax.set_xticks(range(len(unique_run_names)))
            current_ax.set_xticklabels(
                [f'{i}\n\n{non_nan[i]}' for i in range(len(unique_run_names))], linespacing=0.7)

            if log_scale:
                current_ax.set_yscale('log')

            current_ax.set_ylim(0, ylim)


def main(run_paths, exp_name, split_run_name, metrics2plot, eval_vars, log_scale, ylim, clim_metrics_path, save_dir):

    print('\nStart to plot metrics\n')

    plots = {}
    n_cols = min(1, len(run_paths))
    n_rows = math.ceil(len(run_paths)/n_cols)

    row_labels = {}
    column_labels = {}

    for i, r in enumerate(run_paths):

        if save_dir is None:

            # This is a bit cumbersome
            s = os.sep.join(os.path.normpath(r).split(os.sep)[:-1])
            save_dir = os.path.join(s, 'plots', os.path.normpath(
                r).split(os.sep)[-1], exp_name)

        save_dir = os.path.join(os.environ['RESULTS_DIR'], save_dir)
        os.makedirs(save_dir, exist_ok=True)

        files = glob.glob(os.path.join(
            os.environ['RESULTS_DIR'], r, exp_name, 'metrics.csv'))

        metrics = []
        for f in files:
            metrics.append(pd.read_csv(f, index_col=0))

        metrics_df = pd.concat(metrics)

        if clim_metrics_path is not None:
            climatolgy_metrics = pd.read_csv(os.path.join(
                os.environ['RESULTS_DIR'], clim_metrics_path), index_col=0)
        else:
            climatolgy_metrics = None

        if len(eval_vars) <= 2:
            plot_pref = ' '.join(eval_vars)
        else:
            if len(eval_vars) == 0:
                plot_pref = len(metrics_df['var'].unique())
            else:
                plot_pref = f'{len(eval_vars)}'

        if len(eval_vars) > 0:
            # Only keep rows where var matches one of the eval_vars
            metrics_df = metrics_df[metrics_df['var'].isin(eval_vars)]

            if climatolgy_metrics is not None:
                climatolgy_metrics = climatolgy_metrics[climatolgy_metrics['var'].isin(
                    eval_vars)]

        split_run_dict = defaultdict(list)
        # Iterate over all rows
        for index, row in metrics_df.iterrows():
            # Split the run_name into its components
            run_name = row['run_name']
            split_run = run_name.split('_')
            for j, split in enumerate(split_run_name):
                # Match number from split_run[i]
                num = re.search(r'\d+', split_run[j])
                split_run_dict[split].append(
                    num.group() if num else split_run[j])

        # ADD run meta data to the metrics dataframe
        for key, value in split_run_dict.items():
            metrics_df[key] = value

        if i == 0:

            for metric in metrics2plot:
                plots[metric] = {}
                for stat in metrics_df['stat'].unique():
                    plots[metric][stat] = plt.subplots(
                        n_rows, n_cols, figsize=(6*n_cols, 2.3*n_rows))

        current_row = i//n_cols
        current_col = i % n_cols
        if n_rows == 1 and n_cols == 1:
            ax_position = None
        elif n_rows == 1:
            current_row = 0
            ax_position = i
        elif n_cols == 1:
            current_col = 0
            ax_position = i
        else:
            ax_position = (current_row, current_col)

        if ax_position is not None:
            row_labels[current_row] = split_run_dict['model'][0]
            column_labels[current_col] = f"#Prognostic Traininig Variables: {split_run_dict['train_vars'][0]}"

        print("Plot across variables")
        plot_metrics_across_variables(
            metrics_df, plots, ax_position, metrics2plot, log_scale, ylim, climatolgy_metrics)

    print(f"Saving to {save_dir}")
    for metric in plots.keys():
        for s in plots[metric].keys():
            fig, ax = plots[metric][s]
            plt.figure(fig.number)

            if ax_position is None:
                handles, labels = ax.get_legend_handles_labels()
            else:
                handles, labels = ax[ax_position].get_legend_handles_labels()

            # legend to the bottom
            fig.legend(handles, labels, loc='upper center',
                       bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=11)

            # legend to the side
            # fig.legend(handles, labels,loc='center left', bbox_to_anchor=(1.05, 0.5), ncol = 1, fontsize = 11)

            fig.subplots_adjust(hspace=0.4)
            fig.subplots_adjust(wspace=0.1)

            if ax_position is not None:
                add_headers(fig, row_headers=[MODEL_NAME_DICT[row_labels[i]] for i in sorted(row_labels.keys())],
                            col_headers=[column_labels[i] for i in sorted(column_labels.keys())], row_pad=20, col_pad=20, rotate_row_headers=True, fontsize=14,
                            y_label=Y_LABEL_DICT[metric], x_label_pad=10)

            add_xaxis_description(fig)

            for ax in fig.get_axes():
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(11)

            if log_scale:
                plt.savefig(os.path.join(
                    save_dir, f'{plot_pref}_var_{s}_{metric}_log.pdf'), bbox_inches="tight")
            else:
                plt.savefig(os.path.join(
                    save_dir, f'{plot_pref}_var_{s}_{metric}.pdf'), bbox_inches="tight")

            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_paths', type=str, nargs='+',
                        help='Paths to the runs to evaluate. Can contain glob patterns. Must be relative to $RESULTS_DIR', required=True)
    parser.add_argument('--exp_name', type=str,
                        default='', help='relative paths to the run path, which contains the metrics.csv file created with the evaluate_run.py script')
    parser.add_argument("--eval_vars", type=str, nargs='+', default=['tas','uas','vas','t_85000.0','zg_50000.0'],
                        help='Variables whose error should be accumulated. If empty, all variables are used.')
    parser.add_argument("--split_run_name", type=str, nargs='+', default=[
                        'model', 'train_vars', 'n_steps', 'n_layers', 'dim', 'seed'], help='Run name components to split on.')
    parser.add_argument("--metrics2plot", type=str, nargs='+', default=[
                        'norm-rmse'], help='Metrics to plot. Can be any of the following: norm-mae, norm-rmse, mae, rmse')
    parser.add_argument("--clim_metrics_path", type=str, default=None,
                        help='Path to the climatology metrics file. Must be relative to $RESULTS_DIR. Can be None')
    parser.add_argument("--log_scale", type=bool, default=False)
    parser.add_argument("--ylim", type=float, default=0.5)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Path to save the plots. If None, will be saved in the same directory as the run paths. If multiple run paths are provided, plots are in the last one')
    args = parser.parse_args()

    args = vars(args)
    main(**args)
