
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.fft as fft
from torch.nn.parameter import Parameter
# import matplotlib.pyplot as plt
import scipy.io
import h5py

import operator
from functools import reduce
from functools import partial
from timeit import default_timer

from utilities3 import *
import Wavelets

torch.manual_seed(0)
np.random.seed(0)

class WaveletBlock1d(nn.Module):
    def __init__(self, width, input_len, keep, device=None):
        super(WaveletBlock1d, self).__init__()
        self.input_len = input_len
        self.width = width
        self.keep = keep

        self.DWT = Wavelets.HaarDWT(width=width, input_len=input_len, device=device)
        self.IDWT = Wavelets.IHaarDWT(width=width, input_len=input_len, device=device)

        self.scale = (1 / (self.width**2))
        self.weights = nn.Parameter(self.scale * torch.rand(self.width,
                                                            self.width,
                                                            self.keep))

    def forward(self, x):
        # x has shape (batch_size, width, input_len)
        # Do DWT row-by-row
        z = torch.zeros(x.size(), device=x.device)

        x = self.DWT(x)


        # We want to keep only the specified number of coefficients.
        # The high-DWT-level coefficients are towards the left of the array.
        z[:,:,:self.keep] = x[:,:,:self.keep]


        # ok so now z has the DWT coefficients. z is of shape (batch_size, width, input_len)
        z = torch.einsum('bix,iox->box', z, self.weights)

        out = self.IDWT(z)
        return out


class SimpleBlock1d(nn.Module):
    def __init__(self, width, input_len, keep, level=None, device=None):
        super(SimpleBlock1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.width = width
        self.input_len = input_len
        self.device = device
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.wave0 = WaveletBlock1d(self.width, self.input_len, keep, device=self.device)
        self.wave1 = WaveletBlock1d(self.width, self.input_len, keep, device=self.device)
        self.wave2 = WaveletBlock1d(self.width, self.input_len, keep, device=self.device)
        self.wave3 = WaveletBlock1d(self.width, self.input_len, keep, device=self.device)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):


        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.wave0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x1)

        x1 = self.wave1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x1)

        x1 = self.wave2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x1)

        x1 = self.wave3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Net1d(nn.Module):
    def __init__(self, width, input_len, keep, level=None, device=None):
        super(Net1d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock1d(width, input_len, keep, level=level, device=device)


    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

def write_result_to_file(fp, missing_str='', **trial):
    """Write a line to a tab-separated file saving the results of a single
        trial.

    Parameters
    ----------
    fp : str
        Output filepath
    missing_str : str
        (Optional) What to print in the case of a missing trial value
    **trial : dict
        One trial result. Keys will become the file header
    Returns
    -------
    None

    """
    header_lst = list(trial.keys())
    header_lst.sort()
    if not os.path.isfile(fp):
        header_line = "\t".join(header_lst) + "\n"
        with open(fp, 'w') as f:
            f.write(header_line)
    trial_lst = [str(trial.get(i, missing_str)) for i in header_lst]
    trial_line = "\t".join(trial_lst) + "\n"
    with open(fp, 'a') as f:
        f.write(trial_line)

def find_normalized_errors(preds, y, ord):
    diffs = preds - y

    raw_errors = np.linalg.norm(diffs, ord=ord, axis=1)
    raw_mean = raw_errors.mean()
    norms = np.linalg.norm(y, ord=ord, axis=1)
    normalized_errors = np.divide(raw_errors, norms)
    normalized_mean = normalized_errors.mean()
    return raw_mean, normalized_mean

def main(args):

    # Figure out CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Running computation on device: {}".format(device))

    ################################################################
    #  configurations
    ################################################################
    ntrain = 1000
    ntest = 100

    # sub = 2**3 #subsampling rate
    # h = 2**13 // sub #total grid size divided by the subsampling rate
    sub = args.subsample_rate
    h = args.grid_size // sub
    input_len = h

    batch_size = 20
    learning_rate = 0.001

    epochs = args.epochs
    step_size = 100
    gamma = 0.5

    # width = 64

    # results_dd stores trial results and metadata. It will be printed as
    # a single line to a text file at args.results_fp
    results_dd = {'ntrain': ntrain,
                    'ntest': ntest,
                    'sub': sub,
                    'input_len': input_len,
                    'epochs': epochs,
                    'dwt_coeff_kept': args.keep,
                    'width': args.width}

    ################################################################
    # read data
    ################################################################

    # Data is of the shape (number of samples, grid size)
    dataloader = MatReader(args.data_fp)
    x_data = dataloader.read_field('a')[:,::sub]
    y_data = dataloader.read_field('u')[:,::sub]

    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]

    # cat the locations information
    grid = np.linspace(0, 2*np.pi, input_len).reshape(1, input_len, 1)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain, input_len, 1),
                        grid.repeat(ntrain,1,1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest, input_len, 1),
                        grid.repeat(ntest,1,1)], dim=2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train,
                                                                                y_train),
                                                batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test,
                                                                                y_test),
                                                batch_size=batch_size, shuffle=False)

    # model
    model = Net1d(width=args.width,
                    input_len=input_len,
                    keep=args.keep,
                    device=device).to(device)
    logging.info("Number of model parameters: %i" % model.count_params())


    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)

            mse = F.mse_loss(out, y, reduction='mean')
            # mse.backward()
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))

            # If we specify a l1 regularization on the weights, we add up the
            # l1 norms of all of the different weights and add this to the loss.
            # l1_reg = torch.tensor(0.).to(device)
            # for param in model.parameters():
            #     l1_reg += torch.norm(param, p=1)

            # Our loss function is Lp loss + regulariztion term
            # loss = l2 + args.l1_lambda * l1_reg

            # backprop the gradients, then perform one step of the algorithm.
            l2.backward()
            optimizer.step()


            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        logging.info("Epoch: {}, time: {:.2f}, train_mse: {:.4f}, train_l2: {:.4f}, test_l2: {:.4f}".format(ep, t2-t1, train_mse, train_l2, test_l2))

    torch.save(model, args.model_fp)
    logging.info("Saved model at {}".format(args.model_fp))

    # Compute training errors:
    train_pred = torch.zeros(y_train.shape)
    train_y = torch.zeros(y_train.shape)
    idx = 0
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            # print(out.size())
            # print(train_pred[idx].size())

            train_pred[batch_size*idx: batch_size*(idx + 1)] = out
            train_y[batch_size*idx: batch_size*(idx + 1)] = y

            idx += 1

    train_pred = train_pred.cpu().numpy()
    train_y = train_y.cpu().numpy()

    results_dd['train_l2_errors'], results_dd['train_l2_normalized_errors'] = find_normalized_errors(train_pred, train_y, 2)
    results_dd['train_linf_errors'], results_dd['train_linf_normalized_errors'] = find_normalized_errors(train_pred, train_y, np.inf)


    pred = torch.zeros(y_test.shape)
    index = 0
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0
            x, y = x.to(device), y.to(device)

            out = model(x)
            pred[index] = out

            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            logging.info("Test index {}, test_l2: {:.4f}".format(index, test_l2))
            index = index + 1

    scipy.io.savemat(args.preds_fp, mdict={'pred': pred.cpu().numpy()})



    # I'm doing aggregate error reporting in numpy
    pred_test = pred.cpu().numpy()
    y_test = y_test.cpu().numpy()

    results_dd['test_l2_errors'], results_dd['test_l2_normalized_errors'] = find_normalized_errors(pred_test, y_test, 2)
    results_dd['test_linf_errors'], results_dd['test_linf_normalized_errors'] = find_normalized_errors(pred_test, y_test, np.inf)
    if args.results_fp is not None:
        write_result_to_file(args.results_fp, **results_dd)
        logging.info("Wrote results to {}".format(args.results_fp))
    else:
        logging.info("No results_fp specified, so here are the results")
        logging.info(results_dd)

    logging.info("Finished")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fp')
    parser.add_argument('--model_fp')
    parser.add_argument('--preds_fp')
    parser.add_argument('--results_fp', default=None,
                        help='If specified, trial results will be dumped into \
                                one line of this text file')
    parser.add_argument('--subsample_rate', type=int, default=2**3)
    parser.add_argument('--grid_size', type=int, default=2**13)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--keep', type=int, default=16)
    parser.add_argument('--width', type=int, default=64)

    args = parser.parse_args()
    fmt = "%(asctime)s:FNO: %(levelname)s - %(message)s"
    time_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=time_fmt)
    main(args)
