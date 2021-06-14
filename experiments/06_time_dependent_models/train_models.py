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


class FNO1dComplexTime(nn.Module):
    def __init__(self, modes, width):
        super(FNO1dComplexTime, self).__init__()

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
        self.fc0 = nn.Linear(4, self.width) # input channel is 3: (Re(a(x)), Im(a(x)), x, t)

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

    def forward(self, x, t):
        # print("INPUT X SHAPE: {} DTYPE: {}".format(x.shape, x.dtype))
        # print("INPUT T SHAPE: {} DTYPE: {}".format(t.shape, t.dtype))
        # print("T: {}".format(t))
        t = t.view(-1, 1, 1).repeat([1, x.shape[1], 1])
        # print("T0: {}".format(t[0]))
        # print("T1: {}".format(t[1]))
        # print("INPUT T SHAPE: {} DTYPE: {}".format(t.shape, t.dtype))
        # o = torch.ones((1,  x.size()[1]), dtype = torch.float)
        # print("INPUT O SHAPE: {} DTYPE: {}".format(o.shape, o.dtype))
        # t_arr = torch.matmul(t,  o)
        # print("T_ARR SHAPE: {}".format(t_arr.shape))
        x = torch.cat([x, t], dim=2)
        # print("X SHAPE: {}".format(x.shape))
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

class TimeDataSet(torch.utils.data.Dataset):
    def __init__(self, X, t_grid, x_grid):
        super(TimeDataSet, self).__init__()
        assert X.shape[1] == t_grid.shape[-1]
        self.X = X
        self.t = torch.tensor(t_grid.flatten(), dtype=torch.float)
        self.x_grid = torch.tensor(x_grid, dtype=torch.float).view(-1, 1)
        self.n_tsteps = self.t.shape[0] - 1
        self.n_batches = self.X.shape[0]
        self.time_indices = [ (i,j)  for j in range(self.n_tsteps) for i in range(j)]
        self.n_t_pairs = len(self.time_indices)
        self.dataset_len = self.n_t_pairs * self.n_batches

    def make_x_train(self, x_in):
        x_in = torch.view_as_real(torch.tensor(x_in, dtype=torch.cfloat))
        y = torch.cat([x_in, self.x_grid], axis=-1)
        return y

    def __getitem__(self, idx):
        idx_original = idx
        t_idx = int(idx % self.n_t_pairs)
        idx = int(idx // self.n_t_pairs)
        batch_idx = int(idx % self.n_batches)
        start_time_idx, end_time_idx = self.time_indices[t_idx]
        # print("IDX: {}, T_IDX: {}, B_IDX: {}, START_T_IDX: {}, END_T_IDX: {}".format(idx_original, t_idx, batch_idx, start_time_idx, end_time_idx))
        x = self.make_x_train(self.X[batch_idx, start_time_idx]) #.reshape(self.output_shape)
        y = self.X[batch_idx, end_time_idx] #.reshape(self.output_shape)
        t = self.t[end_time_idx - start_time_idx]
        return x,y,t

    def __len__(self):
        return self.dataset_len

    def __repr__(self):
        return "TimeDataSet with length {}, n_tsteps {}, n_t_pairs {}, n_batches {}".format(self.dataset_len,
                                                                                            self.n_tsteps,
                                                                                            self.n_t_pairs,
                                                                                            self.n_batches)

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

    batch_size = 1024
    learning_rate = 0.001

    step_size = 100
    gamma = 0.5

    modes = args.freq_modes
    width = args.width

    # results_dd stores trial results and metadata. It will be printed as
    # a single line to a text file at args.results_fp
    results_dd = {'ntrain': ntrain,
                    'ntest': ntest,
                    'modes': modes,
                    'width': width}

    ################################################################
    # read training data
    ################################################################

    d = sio.loadmat(args.data_fp)
    usol = d['output'][:,::args.time_subsample]
    t_grid = d['t'][:,::args.time_subsample]
    x_grid = d['x']
    logging.info("USOL SHAPE {}, T_GRID SHAPE: {}, X_GRID SHAPE: {}".format(usol.shape, t_grid.shape, x_grid.shape))

    train_dataset = TimeDataSet(usol, t_grid, x_grid)
    logging.info("Dataset: {}".format(train_dataset))
    results_dd['ntrain'] = len(train_dataset)
    # logging.info("N_TSTEPS: {}, N_BATCHES: {}, MAX_TSTEPS: {}, DATA_LEN: {}".format(train_dataset.n_tsteps,
    #                                                                             train_dataset.n_batches,
    #                                                                             train_dataset.max_tsteps,
    #                                                                             train_dataset.dataset_len))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # I want to approximately normalize the time spent on training by enforcing
    # n_epochs * len(train_dataset) = constant \approx 2 hours on GPU
    n_epochs_varied = int(50 * 500000 // len(train_dataset))

    if args.epochs is not None:
        epochs = args.epochs
        logging.info("Beginning training for {} epochs, set externally".format(epochs))
    else:
        epochs = n_epochs_varied
        logging.info("Beginning training for {} epochs, set by data len".format(epochs))

    results_dd['epochs'] = epochs

    ##################################################################
    # initialize model and optimizer
    ##################################################################

    model = FNO1dComplexTime(width=width, modes=modes).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    training_dd = {'model_fp': args.model_fp}

    ################################################################
    # training and evaluation
    ################################################################
    t0_train = default_timer()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y, t in train_data_loader:
            x, y, t = x.to(device), y.to(device), t.to(device)
            # print("X SHAPE: {}, Y SHAPE: {}".format(x.shape, y.shape))

            optimizer.zero_grad()
            out = model(x, t)

            mse = MSE(out, y)
            mse.backward()
            optimizer.step()

            train_mse += mse.item()

        scheduler.step()
        model.eval()

        train_mse /= len(train_data_loader)

        t2 = default_timer()
        logging.info("Epoch: {}, time: {:.2f}, train_mse: {:.4f}".format(ep, t2-t1, train_mse))
        training_dd['epoch'] = ep
        training_dd['MSE'] = train_mse
        training_dd['time'] = t2-t1
        write_result_to_file(args.train_df, **training_dd)

    results_dd['train_mse'] = train_mse

    t1_train = default_timer()
    results_dd['train_time'] = t1_train - t0_train
    logging.info("Completed training in seconds: {:.2f}".format(t1_train - t0_train))

    ################################################################
    # save model
    ################################################################
    torch.save(model, args.model_fp)
    logging.info("Saving model to {}".format(args.model_fp))

    ################################################################
    # read testing data
    ################################################################

    d_test = sio.loadmat(args.test_data_fp)
    usol_test = d_test['output'][:,::args.time_subsample]
    t_grid_test = d_test['t'][:,::args.time_subsample]
    x_grid_test = d_test['x']

    test_dataset = TimeDataSet(usol_test, t_grid_test, x_grid_test)
    logging.info("Test Dataset: {}".format(test_dataset))
    results_dd['ntest'] = len(test_dataset)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    ################################################################
    # test the model
    ################################################################
    test_mse = 0.
    test_l2_norm_error = 0.

    with torch.no_grad():
        for x, y, t in test_data_loader:
            x, y, t = x.to(device), y.to(device), t.to(device)

            out = model(x, t)

            mse = MSE(out, y)
            test_mse += mse.item()

            l2_err = l2_normalized_error(out, y)
            test_l2_norm_error += l2_err.item()

    test_mse /= len(test_data_loader)
    test_l2_norm_error /= len(test_data_loader)

    results_dd['test_mse'] = test_mse
    results_dd['test_l2_normalized_error'] = test_l2_norm_error

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
    parser.add_argument('--test_data_fp')
    parser.add_argument('--model_fp')
    parser.add_argument('--train_df')
    parser.add_argument('--results_fp', default=None,
                        help='If specified, trial results will be dumped into \
                                one line of this text file')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--freq_modes', type=int, default=16)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--time_subsample', type=int, default=1)

    args = parser.parse_args()
    fmt = "%(asctime)s:FNO: %(levelname)s - %(message)s"
    time_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=time_fmt)
    main(args)
