import scipy.io as sio
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import argparse
import pandas as pd


def find_normalized_errors(preds, y, ord):
    diffs = preds - y

    raw_errors = np.linalg.norm(diffs, ord=ord, axis=1)
    raw_mean = raw_errors.mean()
    norms = np.linalg.norm(y, ord=ord, axis=1)
    normalized_errors = np.divide(raw_errors, norms)
    return normalized_errors

def load_data(fp):
    N_TEST = 100
    SUB = 2**3
    data = sio.loadmat(fp)
    X = data['a'][:,::SUB]
    y = data['u'][:,::SUB]
    del data

    X = X[-N_TEST:,:]
    y = y[-N_TEST:,:]
    return (X, y)

def load_preds(fp):
    try:
        x = sio.loadmat(fp)['pred']
        return x
    except KeyError:
        print( sio.loadmat(fp).keys())
        raise ValueError

def quick_hist(data, t, xlabel, fp):
    plt.hist(data)
    plt.ylabel("counts", size=13)
    plt.xlabel(xlabel, size=13)
    plt.title(t)
    plt.savefig(fp)
    plt.clf()

def plot_one_testcase(X, y, preds, i, t, fp=None):
    err = y[i] - preds[i]
    plt.plot(X[i], label='input')
    plt.plot(y[i],'-', label='target')
    plt.plot(preds[i], '--', label='predictions')
    plt.plot(err, label='errors')
    plt.legend()
    plt.title(t)
    if fp is not None:
        plt.savefig(fp)
        plt.clf()
    else:
        plt.show()

def keep_k_fourier_modes(preds, k):
    fourier_coeffs = np.fft.fft(preds)
    # We want the first k and last k entries of axis 1 for each row in axis 0
    fourier_coeffs[:,k+1:-k] = 0+0j
    out_arr = np.fft.ifft(fourier_coeffs)
    # out_arr is real (up to numerical precision) but explicitly casting it avoids
    # numpy warnings
    return np.real(out_arr)

def plot_truncated_testcase(X, y, preds, trunc_preds, i, t, fp=None):
    plt.plot(preds[i], '-', label='predictions')
    plt.plot(trunc_preds[i], '--', label='truncated predictions')
    plt.legend()
    plt.title(t)
    if fp is not None:
        plt.savefig(fp)
        plt.clf()
    else:
        plt.show()

def plot_experiment(key, X, y, preds_fp, plots_out):
    logging.info("Working on experiment: {}".format(key))
    preds = load_preds(preds_fp)

    # Compute and report average errors
    l2_normalized_errors = find_normalized_errors(preds, y, 2)
    #
    # decreasing_l2_error_indices = np.flip(np.argsort(l2_normalized_errors))
    # for i in range(10):
    #     idx = decreasing_l2_error_indices[i]
    #     l2_err = l2_normalized_errors[idx]
    #     t = "{} test sample {}: norm. l2 error {:.4f}".format(key, idx, l2_err)
    #     fp = os.path.join(plots_out, '{}_worst_l2_errors_{}.png'.format(key, i))
    #     plot_one_testcase(X, y, preds, idx, t, fp)

    # Specially-chosen test cases for all of the frequencies
    for i in [1,2,3,4]:
        l2_err = l2_normalized_errors[i]
        t = "{} test sample {}: norm. l2 error {:.4f}".format(key, i, l2_err)
        fp = os.path.join(plots_out, '{}_test_sample_{}.png'.format(key, i))
        plot_one_testcase(X, y, preds, i, t, fp)

    # Truncate the Fourier modes and recompute errors. In the next section,
    # we will be using 2**k Fourier modes.
    # fourier_results = pd.DataFrame(columns=['k', 'num_modes', 'l2_norm_error'])
    # for k in range(1, 11):
    #     num_modes = int(2**k)
    #     truncated_preds = keep_k_fourier_modes(preds, num_modes)
    #     l2_norm_error = find_normalized_errors(truncated_preds, y, 2).mean()
    #     fourier_results = fourier_results.append({'k':k, 'num_modes': num_modes, 'l2_norm_error': l2_norm_error}, ignore_index=True)
    #     for i in [1, 64, 75, 86]:
    #         t = "{} test sample {}: truncated to {} frequency modes".format(key, i, num_modes)
    #         fp = os.path.join(plots_out, '{}_truncated_pred_{}_test_sample_{}.png'.format(key, num_modes, i))
    #         plot_truncated_testcase(X, y, preds, truncated_preds, i, t, fp)
    #
    # plt.plot(fourier_results.k.values, fourier_results.l2_norm_error.values, '.')
    # plt.xticks(fourier_results.k.values, labels=["%i" % 2**j for j in fourier_results.k.values])
    # plt.xlabel("Fourier modes kept", size=13)
    # plt.ylabel("$L_2$ Normalized Error", size=13)
    # t = "{}: Errors after truncation to {} modes".format(key, num_modes)
    # plt.title(t)
    # fp_2 = os.path.join(plots_out, '{}_errors_after_truncation.png'.format(key))
    # plt.savefig(fp_2)
    # plt.clf()

def load_results_df(fp):
    d = pd.read_table(fp)
    d = d[d['modes'] > 4]
    d['freq_str'] = d['modes'].astype(str)
    d = d.sort_values('modes', ascending=True)
    d = d.reset_index()
    return d


def main(args):

    if not os.path.isdir(args.plots_dir):
        logging.info("Making output directory at {}".format(args.plots_dir))
        os.mkdir(args.plots_dir)

    results_df = load_results_df(args.results_df)
    # print(results_df.index.values)
    results_df_no_reg = load_results_df(args.results_df_without_reg)
    # print(results_df_with_spacetime.index.values)

    width_0=0.3

    best_no_weight_test_error = results_df_no_reg.test_l2_normalized_errors.min()
    for n_modes, df_g in results_df.groupby('modes'):
        fp_0 = os.path.join(args.plots_dir, 'l2_error_reg_scatter_modes_{}.png'.format(n_modes))

        t_0 = "{} frequency modes in model".format(n_modes)
        df_g = df_g.sort_values('l1_lambda', ascending=True)
        df_g['lambda_str'] = [ "{:.3e}".format(i) for i in df_g.l1_lambda]

        df_g = df_g[df_g['l1_lambda'] < 2.75e-06]

        plt.plot(df_g.lambda_str.values,
                    df_g.test_l2_normalized_errors.values,
                    'v',
                    color='blue',
                    markersize=10,
                    label='test error')
        plt.plot(df_g.lambda_str.values,
                    df_g.train_l2_normalized_errors.values,
                    '^',
                    color='blue',
                    markersize=10,
                    label='train error')
        plt.hlines(best_no_weight_test_error, 0., df_g.shape[0] + 0.5, linestyles='dashed',
                    label='best test error, 8\nfrequency modes')
        plt.xticks([i for i in range(len(df_g))],
                    labels=df_g.lambda_str.values,
                    rotation=45,
                    ha='right')
        # plt.tight_layout()
        plt.ylim(top=0.12)


        # plt.yscale('log')

        plt.title(t_0)
        plt.xlabel('$L_1$ regularization weight')
        plt.ylabel("Normalized error")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fp_0)
        plt.clf()

        X, y = load_data(args.data_fp)
        for l1_lambda in df_g.l1_lambda:
            k = "{:03d}_freq_modes_l1_{:03e}".format(n_modes, l1_lambda)
            preds_fp = os.path.join(args.preds_dir, 'freq_{}_l1-reg_{}_burgers_1d.mat'.format(n_modes, l1_lambda))
            try:
                plot_experiment(k, X, y, preds_fp, args.plots_dir)
            except FileNotFoundError:
                logging.warning("Could not find {}".format(preds_fp))
    #
    #
    # # First plot l2 errors across all trial results.
    # fp_0 = os.path.join(args.plots_dir, 'l2_error_frequency_scatter.png')
    #
    # plt.plot(results_df.index.values+width_0/2,
    #             results_df.train_l2_normalized_errors.values,
    #             '^',
    #             color='blue',
    #             markersize=10)
    # plt.plot(results_df.index.values+width_0/2,
    #             results_df.test_l2_normalized_errors.values,
    #             'v',
    #             color='blue',
    #             markersize=10)
    # plt.vlines(results_df.index.values+width_0/2,
    #             results_df.test_l2_normalized_errors.values,
    #             results_df.train_l2_normalized_errors.values,
    #             color='blue',
    #             label='without spacetime')
    # plt.plot(results_df_with_spacetime.index.values-width_0/2,
    #             results_df_with_spacetime.test_l2_normalized_errors.values,
    #             'v',
    #             color='red',
    #             markersize=10)
    # plt.plot(results_df_with_spacetime.index.values-width_0/2,
    #             results_df_with_spacetime.train_l2_normalized_errors.values,
    #             '^',
    #             color='red',
    #             markersize=10)
    # plt.vlines(results_df_with_spacetime.index.values-width_0/2,
    #             results_df_with_spacetime.test_l2_normalized_errors.values,
    #             results_df_with_spacetime.train_l2_normalized_errors.values,
    #             color='red',
    #             label='with spacetime')
    # # plt.bar(x=results_df_with_spacetime.index.values-width_0/2 ,
    # #         width=width_0,
    # #         height=results_df_with_spacetime.train_l2_normalized_errors.values,
    # #         bottom=results_df_with_spacetime.test_l2_normalized_errors.values,
    # #         label='with spacetime', alpha=0.5)
    # plt.legend()
    # plt.title("Train/test normalized $L_2$ errors across frequencies")
    # plt.xticks(results_df.index.values,
    #             labels=results_df.freq_str.values)
    # plt.xlabel('Number of Frequency Modes', size=13)
    # plt.ylabel('Normalized $L_2$ Error', size=13)
    # plt.savefig(fp_0)
    # plt.clf()
    # logging.info("Saved plot to {}".format(fp_0))
    #
    # # # L_inf errors across frequencies
    # # fp_1 = os.path.join(args.plots_dir, 'linf_error_frequency_scatter.png')
    # # plt.plot(results_df.freq_str.values, results_df.test_linf_normalized_errors.values, '.',
    # #             label='test')
    # # plt.plot(results_df.freq_str.values, results_df.train_linf_normalized_errors.values, '.',
    # #             label='train')
    # # plt.legend()
    # # plt.title("Normalized $L_\\infty$ errors across frequencies")
    # # plt.xlabel('Number of Frequency Modes', size=13)
    # # plt.ylabel('Normalized $L_\\infty$ Error', size=13)
    # # plt.savefig(fp_1)
    # # plt.clf()
    # # logging.info("Saved plot to {}".format(fp_1))
    # #


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-data_fp", default="/home/owen/projects/fourier_neural_operator/data/2021-03-17_training_Burgers_data_GRF1.mat")
    parser.add_argument("-plots_dir", default="/home/owen/projects/fourier_neural_operator/experiments/02_l1_regularization/plots")
    parser.add_argument("-preds_dir", default="/home/owen/projects/fourier_neural_operator/experiments/02_l1_regularization/preds")
    parser.add_argument("-results_df", default="/home/owen/projects/fourier_neural_operator/experiments/02_l1_regularization/experiment_results.txt")
    parser.add_argument("-results_df_without_reg", default="/home/owen/projects/fourier_neural_operator/experiments/00_increase_frequency_modes/experiment_results.txt")

    args = parser.parse_args()

    fmt = "%(asctime)s: %(levelname)s - %(message)s"
    time_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=time_fmt)
    main(args)
