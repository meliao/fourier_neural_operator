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
# import torch.fft as fft
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py

import operator
from functools import reduce
from functools import partial
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)


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
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
class FNO1dComplex(nn.Module):
    def __init__(self, modes, width):
        super(FNO1dComplex, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (Re(a(x)), Im(a(x)), x)
        input shape: (batchsize, x=s, c=3)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=2)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (Re(a(x)), Im(a(x)), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.view_as_complex(x)


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

def MSE(x, y):
    errors = x - y
    return torch.mean(torch.square(errors.abs()))

def l2_normalized_error(pred, actual):
    errors = pred - actual
    error_norms = torch.linalg.norm(errors, dim=1, ord=2)
    actual_norms = torch.linalg.norm(actual, dim=1, ord=2)
    return torch.mean(torch.divide(error_norms, actual_norms))

def main(args):
    # Figure out CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Running computation on device: {}".format(device))

    ################################################################
    #  configurations
    ################################################################
    ntrain = 1000
    ntest = 100

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
                    'epochs': epochs,
                    'modes': modes,
                    'width': width}

    ################################################################
    # read data
    ################################################################

    x = sio.loadmat(args.data_fp)['output']
    x_data = x[:,1]
    y_data = x[:,2]

    x_train = torch.view_as_real(torch.tensor(x_data[:ntrain,:], dtype = torch.cfloat))
    y_train = torch.tensor(y_data[:ntrain,:], dtype = torch.cfloat)
    x_test = torch.view_as_real(torch.tensor(x_data[-ntest:,:], dtype = torch.cfloat))
    y_test = torch.tensor(y_data[-ntest:,:], dtype = torch.cfloat)

    x_grid = torch.linspace(-np.pi, np.pi, 1024).view(-1,1)
    x_train = torch.cat((x_train, x_grid.repeat(ntrain, 1, 1)), axis=2)
    x_test = torch.cat((x_test, x_grid.repeat(ntest, 1, 1)), axis=2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    ##################################################################
    # initialize model and optimizer
    ##################################################################

    model = FNO1dComplex(width=2, modes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    ################################################################
    # training and evaluation
    ################################################################
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)

            mse = MSE(out, y)
            mse.backward()
            optimizer.step()

            train_mse += mse.item()

        scheduler.step()
        model.eval()

        train_mse /= len(train_loader)

        t2 = default_timer()
        logging.info("Epoch: {}, time: {:.2f}, train_mse: {:.4f}".format(ep, t2-t1, train_mse))
    
    results_dd['train_mse'] = train_mse

    ################################################################
    # create and evaluate test predictions
    ################################################################
    # pred = torch.zeros(y_test.shape, dtype=torch.cfloat)
    pred_arr = []
    # print(pred.size())
    # index = 0
    test_mse = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            print(out.size())
            pred_arr.append(out)
            # pred[index] = out

            test_mse += MSE(out, y)
            # index = index + batch_size

    pred = torch.cat(pred_arr, axis=0)
    print(pred.size())
    sio.savemat(args.preds_fp, {'pred': pred.cpu().numpy()})
    logging.info("Saving predictions to {}".format(args.preds_fp))

    test_mse = test_mse / len(test_loader)
    results_dd['test_mse'] = test_mse.cpu().numpy()

    results_dd['test_l2_normalized_errors'] = l2_normalized_error(pred.to('cpu'), y_test.to('cpu')).cpu().numpy()
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
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--freq_modes', type=int, default=16)

    args = parser.parse_args()
    fmt = "%(asctime)s:FNO: %(levelname)s - %(message)s"
    time_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=time_fmt)
    main(args)
