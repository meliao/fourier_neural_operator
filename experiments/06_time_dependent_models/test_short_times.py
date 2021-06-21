import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import os
import re
import logging

from train_models import FNO1dComplexTime, SpectralConv1d


def load_data(fp):
    logging.info("Loading data from {}".format(fp))
    data = sio.loadmat(os.path.expanduser(fp))
    return data['output'], data['t']

def load_model(fp, device):
    # Model datatypes are loaded from train_models.py
    model = torch.load(fp, map_location=device)
    assert type(model) == FNO1dComplexTime
    return model

def l2_normalized_error(pred, actual):
    """Short summary.

    Parameters
    ----------
    pred : type
        Description of parameter `pred`.
    actual : type
        Description of parameter `actual`.

    Returns
    -------
    types
        Description of returned object.

    """
    errors = pred - actual
    error_norms = torch.linalg.norm(torch.tensor(errors), dim=-1, ord=2)
    actual_norms = torch.linalg.norm(torch.tensor(actual), dim=-1, ord=2)
    normalized_errors = torch.divide(error_norms, actual_norms)
    return normalized_errors.detach().numpy()

def find_errors(preds, y, ord=2):
    return np.mean(preds - y, axis=1)


def prepare_t_input(t, n_batches):
    return torch.tensor(t, dtype=torch.float).repeat((n_batches, 1, 1))

def prepare_input(X):
    # X has shape (nbatch, 1, grid_size)
    s = X.shape[-1]
    n_batches = X.shape[0]

    # Convert to tensor
    X_input = torch.view_as_real(torch.tensor(X, dtype=torch.cfloat))

    # FNO code appends the spatial grid to the input as below:
    x_grid = torch.linspace(-np.pi, np.pi, 1024).view(-1,1)
    X_input = torch.cat((X_input, x_grid.repeat(n_batches, 1, 1)), axis=2)

    return X_input

def test_models_in_dir(X, t_grid, models_dir, plots_dir, prefix=None):
    errors_dd = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Running computation on device: {}".format(device))
    model_lst = os.listdir(models_dir)
    for m in model_lst:
        if prefix is not None:
            if not m.startswith(prefix):
                continue
        model_fp = os.path.join(models_dir, m)
        logging.info("Working on model {}".format(m))
        preds = load_one_model_and_test(model_fp, X, t_grid, device)
        # print("PREDS: {}".format(preds.shape))
        # print("X: {}".format(X.shape))
        for i in range(5):
            fp_heatmap_i = os.path.join(plots_dir, "heatmap_{}_sample_{}.png".format(m, i))
            plot_one_testcase_heatmap_T(preds[i], X[i], fp_heatmap_i)

        norm_errors = l2_normalized_error(preds, X)
        # print(norm_errors)
        # print("NORM ERRORS: {}".format(norm_errors.shape))
        means = np.mean(norm_errors, axis=0)
        # print("NORM ERRORS MEANS: {}".format(means.shape))
        stds = np.std(norm_errors, axis=0)
        # print(stds)
        # print("NORM ERRORS STDS: {}".format(stds.shape))
        errors_dd[m] = {'mean': means, 'std': stds}
    return errors_dd

def load_one_model_and_test(model_fp, X, t_grid, device):

    # X has shape (nbatch, n_tsteps, grid_size)
    # t_grid has shape (1, n_tsteps)

    model = load_model(model_fp, device)
    model_results_lst = []

    predictions_arr = np.zeros(X.shape, dtype=np.cdouble)
    predictions_arr[:,0] = X[:,0,:]
    # First input is given by the initial condition
    input = prepare_input(X[:,0,:]).to(device)
    with torch.no_grad():
        # Iterate along timesteps
        for i in range(1, X.shape[1]):
            time_i = prepare_t_input(t_grid[0,i], X.shape[0]).to(device)
            # Errors from composing the model
            preds_i = model(input, time_i).cpu().numpy()
            predictions_arr[:,i,:] = preds_i
    return predictions_arr


def plot_one_testcase_heatmap_T(preds, solns, fp=None):
    INTERPOLATION = 'bicubic'
    CMAP='hot'
    CMAP_2='hot'
    XT, XT_STR = gen_ticks(0, 1023, -np.pi, np.pi, 5)
    ASPECT= (preds.shape[1] / preds.shape[0]) * 100/256

    # preds has shape (n_tsteps, grid_size)
    errors = np.abs(preds - solns)
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
    fig.set_size_inches(12.8, 9.6)
    im_0 = ax[0].imshow(np.abs(solns), interpolation=INTERPOLATION, aspect=ASPECT, cmap=CMAP)
    ax[0].set_title("NLS Equation Solutions: $|u|$")
    ax[2].set_xticks(XT)
    ax[2].set_xticklabels(XT_STR)
    ax[0].set_ylabel("$t$")
    im_1 = ax[1].imshow(np.abs(preds), interpolation=INTERPOLATION, aspect=ASPECT, cmap=CMAP)
    ax[1].set_title("FNO Predictions: $|\\hat u |$")
    ax[1].set_xlabel("$x$")
    im_2 = ax[2].imshow(errors, interpolation=INTERPOLATION, aspect=ASPECT, cmap=CMAP_2)
    ax[2].set_title("Absolute Errors: $| \\hat u - u |$")
    ax[2].set_ylabel("$t$")
    ax[2].set_xlabel("$x$")
    fig.colorbar(im_0, ax=ax[0])
    fig.colorbar(im_1, ax=ax[1])
    fig.colorbar(im_2, ax=ax[2])
    if fp is not None:
        plt.savefig(fp)
    else:
        plt.show()
    plt.close(fig)

def plot_time_errors(errors_dd, t_grid, fp):
    x_vals = t_grid.flatten()[1:]
    for k,v in errors_dd.items():
        plt.plot(x_vals, v['mean'][1:], label=k, alpha=0.7)
        plt.fill_between(x_vals,
                            v['mean'][1:] + v['std'][1:],
                            v['mean'][1:] - v['std'][1:],
                            alpha=0.3)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("$L_2$-Normalized Errors")
    plt.savefig(fp)
    plt.clf()


def gen_ticks(lb_im, ub_im, lb_world, ub_world, n_ticks):
    tick_locs = np.linspace(lb_im, ub_im, n_ticks)
    dist_world = ub_world - lb_world
    tick_names_flt = [lb_world + dist_world * i /ub_im for i in tick_locs]
    tick_names_str = ["{:.2f}".format(i) for i in tick_names_flt]
    return tick_locs, tick_names_str


def make_training_results_plot_1(result_df, plot_fp):
    plt.plot(result_df.experiment_str.values,
                result_df.test_l2_normalized_error.values, '.', markersize=10)
    plt.xlabel('Experiment')
    plt.ylabel('Test $L_2$-Normalized Error')
    plt.title('FNO Performance on NLS data with random ICs')
    plt.tight_layout()
    plt.savefig(plot_fp)
    plt.clf()

def plot_heatmap(values, axis_y, axis_x, title, cbarlabel, path):
    # l1_axis = np.unique(lambda_1) #Lambda 1 on vertical
    # l2_axis = np.unique(lambda_2) #Lambda 2 on horizontal
    # values_arr = values.reshape((l1_axis.shape[0], l2_axis.shape[0]))
    values = np.log10(values)
    fig, ax = plt.subplots()
    im = ax.imshow(values, origin='lower')
    ax.set_xticks([i for i in range(axis_x.shape[0])])
    ax.set_xticklabels(axis_x)
    # ax.set_xticklabels(["{:.2f}".format(i) for i in l2_axis])
    ax.set_xlabel('Time step subsample factor')
    ax.set_yticks([i for i in range(axis_y.shape[0])])
    ax.set_yticklabels(axis_y)
    # ax.set_yticklabels(["{:.2f}".format(i) for i in l1_axis])
    ax.set_ylabel('Model no. frequency modes')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(cbarlabel)
    # plt.tight_layout()
    plt.title(title)
    plt.savefig(path)
    plt.clf()



def make_training_results_plot_2(result_df, plot_fp):
    plt.plot(result_df.ntrain.values, result_df.train_time.values, '.', markersize=10)
    plt.xlabel('Number of training samples')
    plt.ylabel('Training time (seconds)')
    plt.title('Training time across different dataset sizes')
    plt.tight_layout()
    plt.savefig(plot_fp)
    plt.clf()


def main(args):
    """
    1. Load training results dataframe and make simple overview plots
    2. For each model, load model and make predictions. Make a heatmap of
    errors, and save L2-normalized error statistics (mean, std) for each timestep.
    3. Plot L2-normalized error statistics for each model against each other.
    """

    if not os.path.isdir(args.plots_dir):
        os.mkdir(args.plots_dir)

    # df = pd.read_table(args.training_results)
    # df = df.sort_values('modes')
    # df = df[df['modes'] < 256]
    #
    # dd_sub = {100: 200,
    #             450: 100,
    #             1900: 50,
    #             7800: 25,
    #             49500: 10}
    # df['sub'] = df['ntrain'].map(dd_sub)
    # df = df.sort_values('sub')
    #
    # # df = df[df['ntrain'] < 256]
    # df['experiment_str'] = ('modes_'
    #                         + df['modes'].astype(str)
    #                         + '_sub_'
    #                         + df['ntrain'].astype(str)
    #                         + 'NLS_1d')
    #
    # fp_training_results = os.path.join(args.plots_dir, 'FNO_NLS_test_performance.png')
    # plt_df = df[['modes', 'sub', 'test_l2_normalized_error']].pivot(index='modes', columns='sub', values='test_l2_normalized_error')
    # plot_heatmap(plt_df.values,
    #                 plt_df.index.values,
    #                 plt_df.columns.values,
    #                 'FNO short-time test performance',
    #                 '$log_{10}$ $L_2$-Normalized Error',
    #                 fp_training_results)
    # fp_training_results_t = os.path.join(args.plots_dir, 'FNO_NLS_training_times.png')
    # make_training_results_plot_2(df, fp_training_results_t)

    X, t_grid = load_data(args.data_fp)
    logging.info("Loaded data from {}".format(args.data_fp))

    errors_dd = test_models_in_dir(X,
                                    t_grid,
                                    args.models_dir,
                                    args.plots_dir,
                                    '0')

    errors_fp = os.path.join(args.plots_dir, "FNO_time_errors.png")
    plot_time_errors(errors_dd, t_grid, errors_fp)

    # # We aren't interested in the t=0 -> t=1 part, because at t=0, the initial
    # # condition has vanishing imaginary part.
    # X = X[:,1:]
    # predictions_arr, errors_arr, no_comp_errors_arr = load_one_model_and_test(args.test_model_fp, X)
    # logging.info("Finished composition for model {}".format(args.test_model_fp))
    #
    # for i in range(10):
    #     fp_i = os.path.join(args.plots_dir, 'heatmap_errors_testcase_{}.png'.format(i))
    #     plot_one_testcase_heatmap_T(predictions_arr[i], X[i], fp=fp_i)
    #
    # fp_errors_stats = os.path.join(args.plots_dir, 'composition_test_results.png')
    # plot_test_error_bar_stats(errors_arr, no_comp_errors_arr, fp=fp_errors_stats)
    # out_dd = {'predictions': predictions_arr,
    #             'errors': errors_arr,
    #             'no_composition_errors': no_comp_errors_arr}
    # fp_out = os.path.join(args.plots_dir, 'predictions_and_errors.mat')
    # sio.savemat(fp_out, out_dd)
    # logging.info("Saving predictions and errors to {}".format(fp_out))
    # logging.info("Finished")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fp', default="/home-nfs/meliao/projects/fourier_neural_operator/data/2021-06-11_NLS_data_02/NLS_data_seed_1.mat")
    parser.add_argument('--plots_dir', default="/home-nfs/meliao/projects/fourier_neural_operator/experiments/06_time_dependent_models/plots/")
    parser.add_argument('--training_results', default="/home-nfs/meliao/projects/fourier_neural_operator/experiments/06_time_dependent_models/experiment_results.txt")
    parser.add_argument('--models_dir', default="/home-nfs/meliao/projects/fourier_neural_operator/experiments/06_time_dependent_models/models")

    args = parser.parse_args()
    fmt = "%(asctime)s:FNO: %(levelname)s - %(message)s"
    time_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=time_fmt)
    main(args)
