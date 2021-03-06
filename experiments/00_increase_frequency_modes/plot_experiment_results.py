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

    decreasing_l2_error_indices = np.flip(np.argsort(l2_normalized_errors))
    for i in range(10):
        idx = decreasing_l2_error_indices[i]
        l2_err = l2_normalized_errors[idx]
        t = "{} test sample {}: norm. l2 error {:.4f}".format(key, idx, l2_err)
        fp = os.path.join(plots_out, '{}_worst_l2_errors_{}.png'.format(key, i))
        plot_one_testcase(X, y, preds, idx, t, fp)

    # Specially-chosen test cases for all of the frequencies
    for i in [64]:
        l2_err = l2_normalized_errors[i]
        t = "{} test sample {}: norm. l2 error {:.4f}".format(key, i, l2_err)
        fp = os.path.join(plots_out, '{}_test_sample_{}.png'.format(key, i))
        plot_one_testcase(X, y, preds, i, t, fp)

    # Truncate the Fourier modes and recompute errors. In the next section,
    # we will be using 2**k Fourier modes.
    fourier_results = pd.DataFrame(columns=['k', 'num_modes', 'l2_norm_error'])
    for k in range(1, 11):
        num_modes = int(2**k)
        truncated_preds = keep_k_fourier_modes(preds, num_modes)
        l2_norm_error = find_normalized_errors(truncated_preds, y, 2).mean()
        fourier_results = fourier_results.append({'k':k, 'num_modes': num_modes, 'l2_norm_error': l2_norm_error}, ignore_index=True)
        for i in [1, 64, 75, 86]:
            t = "{} test sample {}: truncated to {} frequency modes".format(key, i, num_modes)
            fp = os.path.join(plots_out, '{}_truncated_pred_{}_test_sample_{}.png'.format(key, num_modes, i))
            plot_truncated_testcase(X, y, preds, truncated_preds, i, t, fp)

    plt.plot(fourier_results.k.values, fourier_results.l2_norm_error.values, '.')
    plt.xticks(fourier_results.k.values, labels=["%i" % 2**j for j in fourier_results.k.values])
    plt.xlabel("Fourier modes kept", size=13)
    plt.ylabel("$L_2$ Normalized Error", size=13)
    t = "{}: Errors after truncation to {} modes".format(key, num_modes)
    plt.title(t)
    fp_2 = os.path.join(plots_out, '{}_errors_after_truncation.png'.format(key))
    plt.savefig(fp_2)
    plt.clf()




def main(args):

    if not os.path.isdir(args.plots_dir):
        logging.info("Making output directory at {}".format(args.plots_dir))
        os.mkdir(args.plots_dir)
    results_df = pd.read_table(args.results_df)
    results_df['freq_str'] = results_df['modes'].astype(str)

    results_df = results_df.sort_values('modes', ascending=True)

    # First plot l2 errors across all trial results.
    fp_0 = os.path.join(args.plots_dir, 'l2_error_frequency_scatter.png')
    plt.plot(results_df.freq_str.values, results_df.test_l2_normalized_errors.values, '.',
                label='test')
    plt.plot(results_df.freq_str.values, results_df.train_l2_normalized_errors.values, '.',
                label='train')
    plt.legend()
    plt.title("Normalized $L_2$ errors across frequencies")
    plt.xlabel('Number of Frequency Modes', size=13)
    plt.ylabel('Normalized $L_2$ Error', size=13)
    plt.savefig(fp_0)
    plt.clf()
    logging.info("Saved plot to {}".format(fp_0))

    # L_inf errors across frequencies
    fp_1 = os.path.join(args.plots_dir, 'linf_error_frequency_scatter.png')
    plt.plot(results_df.freq_str.values, results_df.test_linf_normalized_errors.values, '.',
                label='test')
    plt.plot(results_df.freq_str.values, results_df.train_linf_normalized_errors.values, '.',
                label='train')
    plt.legend()
    plt.title("Normalized $L_\\infty$ errors across frequencies")
    plt.xlabel('Number of Frequency Modes', size=13)
    plt.ylabel('Normalized $L_\\infty$ Error', size=13)
    plt.savefig(fp_1)
    plt.clf()
    logging.info("Saved plot to {}".format(fp_1))

    X, y = load_data(args.data_fp)
    for freq in results_df.modes:
        k = "{:03d}_freq_modes".format(freq)
        preds_fp = os.path.join(args.preds_dir, 'freq_{}_burgers_1d.mat'.format(freq))
        plot_experiment(k, X, y, preds_fp, args.plots_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-data_fp", default="/home/owen/projects/fourier_neural_operator/data/2021-03-17_training_Burgers_data_GRF1.mat")
    parser.add_argument("-plots_dir", default="/home/owen/projects/fourier_neural_operator/experiments/00_increase_frequency_modes/plots")
    parser.add_argument("-preds_dir", default="/home/owen/projects/fourier_neural_operator/experiments/00_increase_frequency_modes/preds")
    parser.add_argument("-results_df", default="/home/owen/projects/fourier_neural_operator/experiments/00_increase_frequency_modes/experiment_results.txt")

    args = parser.parse_args()

    fmt = "%(asctime)s: %(levelname)s - %(message)s"
    time_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=time_fmt)
    main(args)
