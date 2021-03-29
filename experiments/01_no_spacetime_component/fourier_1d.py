"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import scipy.io
import h5py

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
# from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

#Complex multiplication
def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 1, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.irfft(out_ft, 1, normalized=True, onesided=True, signal_sizes=(x.size(-1), ))
        return x

class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width):
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

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.w0 = nn.Conv1d(self.width, self.width, 1)
        # self.w1 = nn.Conv1d(self.width, self.width, 1)
        # self.w2 = nn.Conv1d(self.width, self.width, 1)
        # self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        # x2 = self.w0(x)
        # x = x1 + x2
        x = F.relu(x1)

        x1 = self.conv1(x)
        # x2 = self.w1(x)
        # x = x1 + x2
        x = F.relu(x1)

        x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        x = F.relu(x1)

        x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2

        x = x1.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net1d(nn.Module):
    def __init__(self, modes, width):
        super(Net1d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock1d(modes, width)


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

    ################################################################
    #  configurations
    ################################################################
    ntrain = 1000
    ntest = 100

    # sub = 2**3 #subsampling rate
    # h = 2**13 // sub #total grid size divided by the subsampling rate
    sub = args.subsample_rate
    h = args.grid_size // sub
    s = h

    batch_size = 20
    learning_rate = 0.001

    epochs = args.epochs
    step_size = 100
    gamma = 0.5

    modes = args.freq_modes
    width = 64

    # results_dd stores trial results and metadata. It will be printed as
    # a single line to a text file at args.results_fp
    results_dd = {'ntrain': ntrain,
                    'ntest': ntest,
                    'sub': sub,
                    'effective_grid_size': s,
                    'epochs': epochs,
                    'modes': modes,
                    'width': width}

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
    grid = np.linspace(0, 2*np.pi, s).reshape(1, s, 1)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain,s,1), grid.repeat(ntrain,1,1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # model
    model = Net1d(modes, width).cuda()
    print("Number of model parameters:", model.count_params())


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
            x, y = x.cuda(), y.cuda()
            # print("TRAINING X SHAPE: {}".format(x.size()))
            # print("TRAINING Y SHAPE: {}".format(y.size()))

            optimizer.zero_grad()
            out = model(x)

            mse = F.mse_loss(out, y, reduction='mean')
            # mse.backward()
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

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
            x = x.cuda()
            y = y.cuda()
            out = model(x)
            print(out.size())
            print(train_pred[idx].size())

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
            x, y = x.cuda(), y.cuda()

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
    parser.add_argument('--freq_modes', type=int, default=16)

    args = parser.parse_args()
    fmt = "%(asctime)s:FNO: %(levelname)s - %(message)s"
    time_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=time_fmt)
    main(args)
