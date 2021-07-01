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
from train_models_baseline import FNO1dComplex

def load_data(fp):
    logging.info("Loading data from {}".format(fp))
    data = sio.loadmat(os.path.expanduser(fp))
    return data['output'], data['t']

def load_model(fp, device):
    # Model datatypes are loaded from train_models.py
    model = torch.load(fp, map_location=device)
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

def load_one_model_and_test(model_fp, X, t_grid, device):

    # X has shape (nbatch, n_tsteps, grid_size)
    model = load_model(model_fp, device)
    time_dep_model = type(model) == FNO1dComplexTime

    TEST_KEYS = ['comp']

    if time_dep_model:
        TEST_KEYS.append('from_ic')

    preds_dd = {i: np.zeros((X.shape[0], X.shape[1]-1, X.shape[2]), dtype=np.cdouble) for i in TEST_KEYS}
    errors_dd = {i: np.zeros((X.shape[0], X.shape[1]-1), dtype=np.double) for i in TEST_KEYS}

    one_tensor = torch.tensor(1, dtype=torch.float).repeat([X.shape[0],1,1])
    half_tensor = torch.tensor(0.5, dtype=torch.float).repeat([X.shape[0],1,1])
    IC_input = prepare_input(X[:,0,:])

    # First input is given by the initial condition
    comp_input_i = prepare_input(X[:,0,:])
    # Iterate along timesteps
    for i in range(t_grid.shape[1]-1):
        SOLN_I = torch.tensor(X[:,i+1,:])
        # First test: composing the model
        if time_dep_model:
            comp_preds_i = model(comp_input_i, one_tensor).detach().numpy()
        else:
            comp_preds_i = model(comp_input_i).detach().numpy()
        preds_dd['comp'][:,i,:] = comp_preds_i
        comp_input_i = prepare_input(comp_preds_i)
        errors_i = l2_normalized_error(torch.tensor(comp_preds_i), SOLN_I)
        errors_dd['comp'][:,i] = errors_i

        # Second test: prediction from initial condition
        if time_dep_model:
            i_tensor = torch.tensor(t_grid[0,i+1], dtype=torch.float).repeat([X.shape[0],1,1])
            preds_k = model(IC_input, i_tensor).detach().numpy()
            preds_dd['from_ic'][:,i,:] = preds_k
            errors_k = l2_normalized_error(torch.tensor(preds_k), SOLN_I)
            errors_dd['from_ic'][:,i] = errors_k
    # for k,v in errors_dd.items():
    #     print("{}: {}: {}".format(k, v.shape, v))
    dd_out = {'preds': preds_dd, 'errors': errors_dd }
    return dd_out


def plot_time_errors(errors_dd, t_grid, title, fp):

    n_t_steps = t_grid.shape[1]
    x_vals = t_grid.flatten()

    for k, v in errors_dd.items():
        # print("{}: {}: {}".format(k, v.shape, v))
        v_means = np.mean(v, axis=0)
        v_stds = np.std(v, axis=0)
        plt.plot(x_vals, v_means, label=k, alpha=0.7)
        plt.fill_between(x_vals,
                            v_means + v_stds,
                            v_means - v_stds,
                            alpha=0.3)
    plt.legend()
    plt.xlabel("Time step")
    plt.xticks(ticks=np.arange(0, n_t_steps),
               labels=make_special_ticks(n_t_steps),
              rotation=45,
              ha='right',
              )
    plt.ylabel("$L_2$-Normalized Errors")
    # plt.yscale('log')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fp)
    plt.clf()


def make_special_ticks(n):
    s = "$t={} \\ \\to  \\ t={}$"
    return [s.format(i, i+1) for i in range(n)]


def main(args):
    """
    1. For each model, load model and make predictions. Make a heatmap of
    errors, and save L2-normalized error statistics (mean, std) for each timestep.
    """

    if not os.path.isdir(args.plots_dir):
        os.mkdir(args.plots_dir)
    X, t_grid = load_data(args.data_fp)
    logging.info("Loaded data from {}".format(args.data_fp))
    logging.info("X shape: {}, t_grid shape: {}".format(X.shape, t_grid.shape))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Working on baseline 1")
    baseline_1_dd = load_one_model_and_test(args.baseline_1, X, t_grid, device)

    logging.info("Working on baseline 2")
    baseline_2_dd = load_one_model_and_test(args.baseline_2, X, t_grid, device)


    t_pts = 20
    errors_dd = {"FNO: Experiment 1": baseline_1_dd['errors']['comp'][:,:t_pts],
                    "FNO: Experiment 2": baseline_2_dd['errors']['comp'][:,:t_pts]}
    plt_errors_dd = {}
    for k, v in errors_dd.items():
        plt_errors_dd[k] = np.delete(v, [59], axis=0)


    fp_out = os.path.join(args.plots_dir, "time_errors_baseline_comparison.png")
    plot_time_errors(plt_errors_dd, t_grid[:, :t_pts], "Comparing FNO Baselines", fp_out)

    logging.info("Finished")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fp', default="~/projects/fourier_neural_operator/data/2021-06-24_NLS_data_04_test.mat")
    parser.add_argument('--baseline_2', default="/home/owen/projects/fourier_neural_operator/experiments/07_long_time_dependent_runs/models/01_baseline_ep_1000")
    parser.add_argument('--baseline_1', default="/home/owen/projects/fourier_neural_operator/experiments/05_NLS_composition_baseline/models/freq_8_NLS_1d")
    parser.add_argument('--plots_dir', default="/home/owen/projects/fourier_neural_operator/experiments/08_FNO_pretraining/plots/")

    args = parser.parse_args()
    fmt = "%(asctime)s:FNO: %(levelname)s - %(message)s"
    time_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=time_fmt)
    main(args)
