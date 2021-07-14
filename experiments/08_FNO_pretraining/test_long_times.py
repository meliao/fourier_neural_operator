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

def test_models(X, t_grid, baseline_fp, time_dep_fp, pretrained_fp):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Running computation on device: {}".format(device))
    logging.info("Working on baseline model")
    b_preds_dd, b_errors_dd = load_one_model_and_test(baseline_fp, X, t_grid, device)
    baseline_dd = {'preds': b_preds_dd, 'errors':b_errors_dd}

    logging.info("Working on time-dependent model")
    a_preds_dd, a_errors_dd = load_one_model_and_test(time_dep_fp, X, t_grid, device)
    time_dep_dd = {'preds': a_preds_dd, 'errors': a_errors_dd}

    logging.info("Working on pretrained model")
    c_preds_dd, c_errors_dd = load_one_model_and_test(pretrained_fp, X, t_grid, device)
    pretrained_dd = {'preds': c_preds_dd, 'errors': c_errors_dd}

    return baseline_dd, time_dep_dd, pretrained_dd

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
    return preds_dd, errors_dd


def plot_one_testcase_heatmap_T(preds, solns, model_name, fp=None):
    INTERPOLATION = None
    CMAP='hot'
    CMAP_2='hot'
    XT, XT_STR = gen_ticks(0, 1023, -np.pi, np.pi, 5)
    ASPECT= (preds.shape[1] / preds.shape[0]) * 100/256

    # preds has shape (n_tsteps, grid_size)
    errors = np.abs(preds - solns)
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
    fig.set_size_inches(12.8, 9.6)
    im_0 = ax[0].imshow(np.abs(solns), interpolation=INTERPOLATION, aspect=ASPECT, cmap=CMAP)
    ax[0].set_title("NLS Equation Solutions: $|u|$", fontsize=20)
    ax[2].set_xticks(XT)
    ax[2].set_xticklabels(XT_STR)
    ax[0].set_ylabel("$t$", fontsize=20)
    ax[1].set_ylabel("$t$", fontsize=20)
    im_1 = ax[1].imshow(np.abs(preds), interpolation=INTERPOLATION, aspect=ASPECT, cmap=CMAP)
    ax[1].set_title("{} Predictions: $|\\hat u |$".format(model_name), fontsize=20)
    im_2 = ax[2].imshow(errors, interpolation=INTERPOLATION, aspect=ASPECT, cmap=CMAP_2)
    ax[2].set_title("Absolute Errors: $| \\hat u - u |$", fontsize=20)
    ax[2].set_ylabel("$t$", fontsize=20)
    ax[2].set_xlabel("$x$", fontsize=20)
    fig.colorbar(im_0, ax=ax[0])
    fig.colorbar(im_1, ax=ax[1])
    fig.colorbar(im_2, ax=ax[2])
    if fp is not None:
        plt.savefig(fp)
    else:
        plt.show()
    plt.close(fig)


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


def gen_ticks(lb_im, ub_im, lb_world, ub_world, n_ticks):
    tick_locs = np.linspace(lb_im, ub_im, n_ticks)
    dist_world = ub_world - lb_world
    tick_names_flt = [lb_world + dist_world * i /ub_im for i in tick_locs]
    tick_names_str = ["{:.2f}".format(i) for i in tick_names_flt]
    return tick_locs, tick_names_str


def make_train_test_plot(a_train, a_test, b_train, b_test, fp=None):
    fig, ax = plt.subplots(1, 2, sharey=False)


    # a_train and a_test are the time-dependent FNO data. They're in the first column
    ax[0].set_title("Pretraining: Baseline FNO")
    ax[0].plot(a_train.epoch, a_train.MSE, '-', color='red', label='train')
    ax[0].plot(a_test.epoch, a_test.test_mse, '--', color='red', label='test')
    ax[0].set_xlabel("Epoch", fontsize=13)
    ax[0].legend()
    ax[0].set_yscale('log')

    ax[0].set_ylabel("MSE", fontsize=13)

    # b_train and b_test are the time-dependent. They're in the seecond column
    ax[1].set_title("Time-Dependent Training")
    ax[1].plot(b_train.epoch, b_train.MSE, '-', color='blue', label='train')
    ax[1].plot(b_test.epoch, b_test.test_mse, '--', color='blue', label='test')
    ax[1].set_xlabel("Epoch", fontsize=13)
    ax[1].set_yscale('log')
    ax[1].legend(fontsize=13)


    plt.tight_layout()

    if fp is not None:
        plt.savefig(fp)
    else:
        plt.show()
    plt.close(fig)


def plot_one_testcase_panels(preds, baseline_preds, solns, fp=None):
    SHOW_T_TIMESTEPS = 20
    ALPHA = 0.7 # Controls saturation of overlapping lines. 1 is full saturation, 0 is none.
    # preds has shape (n_tsteps, grid_size)
    _, grid_size = preds.shape
    n_tsteps = SHOW_T_TIMESTEPS

    fig, ax = plt.subplots(n_tsteps, 3, sharex='col', sharey=False)
    fig.set_size_inches(15,20) #15, 20 works well
    ax[0,0].set_title("$Re(u)$", size=20)
    ax[0,1].set_title("$Im(u)$", size=20)
    ax[0,2].set_title("$| u - \\hat u|$", size=20)

    for i in range(n_tsteps):
        # First column has Re(prediction), Re(solution)
        ax[i,0].plot(np.real(preds[i]), alpha=ALPHA, label='Pre-Trained FNO preds')
        ax[i,0].plot(np.real(solns[i+1]), alpha=ALPHA, label='solutions')
        ax[i,0].plot(np.real(baseline_preds[i]), '--', alpha=ALPHA, label='FNO Baseline preds')
        ax[i,0].set_ylabel("t = {}".format(i+1), size=15)

        # Second column has Im(prediction), Im(solution)
        ax[i,1].plot(np.imag(preds[i]), alpha=ALPHA, label='Pre-Trained FNO preds')
        ax[i,1].plot(np.imag(solns[i+1]), alpha=ALPHA, label='solutions')
        ax[i,1].plot(np.imag(baseline_preds[i]), '--', alpha=ALPHA, label='FNO Baseline preds')

        # Third column is Abs(prediction - solution)
        errors_i = np.abs(preds[i] - solns[i+1])
        b_errors_i = np.abs(baseline_preds[i] - solns[i+1])
        ax[i,2].plot(errors_i, label='Pre-Trained FNO errors', color='red')
        ax[i,2].plot(b_errors_i, label='FNO baseline errors', color='green')
        ax[i,2].hlines(0, xmin=0, xmax=len(errors_i)-1, linestyles='dashed')

    ax[0,0].legend(fontsize=13, markerscale=2)
    ax[0,1].legend(fontsize=13, markerscale=2)
    ax[0,2].legend(fontsize=13, markerscale=2)
    if fp is not None:
        plt.savefig(fp)
    else:
        plt.show()
    plt.close(fig)


def main(args):
    """
    1. For each model, load model and make predictions. Make a heatmap of
    errors, and save L2-normalized error statistics (mean, std) for each timestep.
    """

    if not os.path.isdir(args.plots_dir):
        os.mkdir(args.plots_dir)

    df_results = pd.read_table(args.df_results)
    plt.plot(df_results.learning_rate, df_results.test_mse, '.')
    plt.xlabel("Learning Rate", size=14)
    plt.xscale('log')
    plt.ylabel("Test MSE", size=14)
    plt.tight_layout()
    fp_0 = os.path.join(args.plots_dir, 'results_test_MSE.png')
    plt.savefig(fp_0)
    plt.clf()

    df_pretrain_train = pd.read_table(args.df_pretrain_train)
    df_pretrain_test = pd.read_table(args.df_pretrain_test)

    df_time_dep_test = pd.read_table(args.df_time_dep_test)
    df_time_dep_train = pd.read_table(args.df_time_dep_train)

    fp_train_test = os.path.join(args.plots_dir, "train_test_MSE.png")
    make_train_test_plot(df_pretrain_train, df_pretrain_test,
                            df_time_dep_train, df_time_dep_test,
                            fp_train_test)
    X, t_grid = load_data(args.data_fp)
    logging.info("Loaded data from {}".format(args.data_fp))
    logging.info("X shape: {}, t_grid shape: {}".format(X.shape, t_grid.shape))

    baseline_dd, time_dep_dd, pretrained_dd = test_models(X, t_grid,
                                            args.baseline_model,
                                            args.time_dep_model,
                                            args.pretrained_model)
    baseline_errors = baseline_dd['errors']['comp']

    bad_cases = [59, 37, 41]
    l = list(range(5))
    l.extend(bad_cases)

    for i in l:
        fp_baseline_preds = os.path.join(args.plots_dir, 'baseline_heatmap_{}.png'.format(i))
        plot_one_testcase_heatmap_T(baseline_dd['preds']['comp'][i], X[i, 1:], "FNO Baseline", fp_baseline_preds)

        fp_time_dep_preds = os.path.join(args.plots_dir, 'time_dep_heatmap_{}.png'.format(i))
        plot_one_testcase_heatmap_T(time_dep_dd['preds']['from_ic'][i], X[i, 1:], "Time-Dependent FNO", fp_time_dep_preds)

        fp_pretrained_preds = os.path.join(args.plots_dir, 'pretrained_heatmap_{}.png'.format(i))
        plot_one_testcase_heatmap_T(pretrained_dd['preds']['from_ic'][i], X[i, 1:], "Pre-Trained FNO", fp_pretrained_preds)

        fp_panels = os.path.join(args.plots_dir, 'panels_{}.png'.format(i))
        plot_one_testcase_panels(pretrained_dd['preds']['from_ic'][i],
                                    baseline_dd['preds']['comp'][i],
                                    X[i],
                                    fp=fp_panels)

    baseline_errors = baseline_dd['errors']['comp']
    time_dep_errors = time_dep_dd['errors']['from_ic']
    pretrained_errors = pretrained_dd['errors']['from_ic']
    pretrained_comp_errors = pretrained_dd['errors']['comp']

    errors_dd_0 = {'FNO': baseline_errors[:, :15],
            'TD-FNO': time_dep_errors[:, :15],
            'TD-FNO (Pretrained)': pretrained_errors[:, :15],
            # 'Pretrained FNO composition': pretrained_comp_errors[:, :15]
            # 'Time-dependent composed': time_dep_comp_errors
            }

    fp_time_errors = os.path.join(args.plots_dir, 'time_errors.png')
    plot_time_errors(errors_dd_0,
                        t_grid[:, :15],
                        'FNO Baseline vs. Time-Dependent Model',
                        fp_time_errors)

    baseline_outlier_removal = np.delete(baseline_errors, [59], axis=0)
    time_dep_outlier_removal = np.delete(time_dep_errors, [59], axis=0)
    pretrained_outlier_removal = np.delete(pretrained_errors, [59], axis=0)
    pretrained_comp_outlier_removal = np.delete(pretrained_comp_errors, [59], axis=0)
    errors_dd_1 = {'FNO': baseline_outlier_removal,
                    'TD-FNO': time_dep_outlier_removal,
                    'TD-FNO (Pretrained)': pretrained_outlier_removal,
                    # 'Pretrained FNO composition': pretrained_comp_outlier_removal
                    }
    print(time_dep_outlier_removal.shape)
    fp_outlier_removal = os.path.join(args.plots_dir, 'time_errors_outlier_removal.png')
    plot_time_errors(errors_dd_1,
                        t_grid[:, :-1],
                        'FNO Baseline vs. Time-Dependent Model',
                        fp_outlier_removal)

    logging.info("Finished")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fp', default="/local/meliao/projects/fourier_neural_operator/data/2021-06-24_NLS_data_04_test.mat")
    parser.add_argument('--df_pretrain_train', default="/local/meliao/projects/fourier_neural_operator/experiments/08_FNO_pretraining/results/00_pretrain_train.txt")
    parser.add_argument('--df_pretrain_test', default="/local/meliao/projects/fourier_neural_operator/experiments/08_FNO_pretraining/results/00_pretrain_test.txt")
    parser.add_argument('--df_time_dep_train', default="/local/meliao/projects/fourier_neural_operator/experiments/08_FNO_pretraining/results/lr_exp_-2_train.txt")
    parser.add_argument('--df_time_dep_test', default="/local/meliao/projects/fourier_neural_operator/experiments/08_FNO_pretraining/results/lr_exp_-2_test.txt")
    parser.add_argument('--df_results', default="/local/meliao/projects/fourier_neural_operator/experiments/08_FNO_pretraining/experiment_results.txt")
    parser.add_argument('--baseline_model', default="/local/meliao/projects/fourier_neural_operator/experiments/07_long_time_dependent_runs/models/01_baseline_ep_1000")
    parser.add_argument('--time_dep_model', default="/local/meliao/projects/fourier_neural_operator/experiments/07_long_time_dependent_runs/models/00_time_dep_ep_1000")
    parser.add_argument('--pretrained_model', default="/local/meliao/projects/fourier_neural_operator/experiments/08_FNO_pretraining/models/lr_exp_-2_ep_1000")
    parser.add_argument('--plots_dir', default="/local/meliao/projects/fourier_neural_operator/experiments/08_FNO_pretraining/plots/")

    args = parser.parse_args()
    fmt = "%(asctime)s:FNO: %(levelname)s - %(message)s"
    time_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=time_fmt)
    main(args)
