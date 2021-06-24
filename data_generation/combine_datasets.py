import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import argparse
import sys

def plot_examples(a_out, u_out, out_dir):
    for i in range(10):
        fp_i = os.path.join(out_dir, 'sample_{}.png'.format(i))
        logging.info("Sample plotted at {}".format(fp_i))
        plt.plot(a_out[i], label='a')
        plt.plot(u_out[i], label='u')
        plt.legend()
        plt.savefig(fp_i)
        plt.clf()


def main(args):
    file_lst = [os.path.join(args.in_dir, i) for i in os.listdir(args.in_dir)]
    file_lst.sort()

    dd_of_lsts = {k:[] for k in args.key}
    logging.info("Gathering keys: {}".format(list(dd_of_lsts.keys())))


    for i in file_lst:
        data = sio.loadmat(i)
        for k, l in dd_of_lsts.items():
            l.append(data[k])
        # a_arr.append(data['output'])
        # logging.info(data['output'].shape)
        # u_arr.append(data['u'])
        logging.info("Loaded {}".format(i))

    out_dd = {}
    for k, l in dd_of_lsts.items():
        if k in ['t', 'x']:
            out_dd[k] = l[0]
        else:
            out_dd[k] = np.concatenate(l, axis=0)
        logging.info("Output has key {} with shape {}".format(k, out_dd[k].shape))
    sio.savemat(args.out_fp, out_dd)
    logging.info("Output saved to {}".format(args.out_fp))

    try:
        if args.train_split is not None:
            ntrain = int(args.train_split[0])
            train_fp = args.train_split[1]
            train_dd = {'output': out_dd['output'][:ntrain],
                            'x': out_dd['x'],
                            't': out_dd['t']}
            sio.savemat(train_fp, train_dd)
            logging.info("Train dataset saved to {}".format(train_fp))

        if args.test_split is not None:
            ntest = int(args.test_split[0])
            test_fp = args.test_split[1]
            test_dd = {'output': out_dd['output'][-ntest:],
                            'x': out_dd['x'],
                            't': out_dd['t']}
            sio.savemat(test_fp, test_dd)
            logging.info("Test dataset saved to {}".format(test_fp))
    except KeyError:
        logging.error("Train/Test splitting is only implemented for keys 'output', 'x', 't'")
    logging.info("Finished")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-in_dir', required=True)
    parser.add_argument('-out_fp', required=True)
    parser.add_argument('-key', nargs='+', required=True)
    parser.add_argument('-train_split', nargs=2, required=False)
    parser.add_argument('-test_split', nargs=2, required=False)

    args = parser.parse_args()
    fmt = "%(asctime)s: %(levelname)s - %(message)s"
    time_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=time_fmt)
    main(args)
