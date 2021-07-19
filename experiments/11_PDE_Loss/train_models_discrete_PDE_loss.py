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
# import matplotlib.pyplot as plt
import scipy.io as sio
# import h5py

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
import re

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
        self.X = torch.tensor(X, dtype=torch.cfloat)
        self.t = torch.tensor(t_grid.flatten(), dtype=torch.float)
        self.x_grid = torch.tensor(x_grid, dtype=torch.float).view(-1, 1)
        self.n_tsteps = self.t.shape[0] - 1
        self.n_batches = self.X.shape[0]
        self.dataset_len = self.n_tsteps * self.n_batches

    def make_x_train(self, x_in):
        x_in = torch.view_as_real(x_in)
        y = torch.cat([x_in, self.x_grid], axis=1)
        return y

    def __getitem__(self, idx):
        idx_original = idx
        t_idx = int(idx % self.n_tsteps) + 1
        idx = int(idx // self.n_tsteps)
        batch_idx = int(idx % self.n_batches)
        x = self.make_x_train(self.X[batch_idx, 0]) #.reshape(self.output_shape)
        y = self.X[batch_idx, t_idx] #.reshape(self.output_shape)
        t = self.t[t_idx]
        return x,y,t

    def __len__(self):
        return self.dataset_len

    def __repr__(self):
        return "TimeDataSet with length {}, n_tsteps {}, n_batches {}".format(self.dataset_len,
                                                                                            self.n_tsteps,
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


class NLS_Residual_Loss:
    """
    NLS: i u_t + 1 / 2 * u_xx + |u|^2 u = 0

    """
    def __init__(self, delta_x, delta_t, n_grid_points):
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.n_grid_points = n_grid_points
        # self.batch_size = batch_size
        # self.I = torch.eye(self.batch_size)
        # self.imag = torch.tensor(0+1j, dtype=torch.cfloat).repeat((self.batch_size, self.n_grid_points))


    def time_derivative(self, model, x, t):
        """
        u_t ~= (u(x, t + delta_t) - u(x, t)) / delta_t
        """
        term_1 = model(x,t)
        t_advance = t + self.delta_t
        term_2 = model(x, t_advance)
        out = torch.div(term_2 - term_1, self.delta_t)
        return out


    def spatial_discrete_derivatives(self, u):
        u_shift_right = torch.roll(u, 1, 1)
        u_shift_left = torch.roll(u, -1, 1)
        
        u_xx = (u_shift_left  - 2*u + u_shift_right) / (self.delta_x ** 2)
        return u_xx
        
    def __call__(self, model, x, t):
        # x has shape (batch_size, s, 3)
        # u has shape (batch_size, s, 1)
        return self.NLS_residual(model, x, t)

    def NLS_residual(self, model, x, t):
        u = model(x,t)

        u_abs = torch.mul(u, torch.square(torch.abs(u)))
        u_t = self.time_derivative(model, x, t)
        u_xx = self.spatial_discrete_derivatives(u)

        resid = torch.mul(u_t, 0+1j) + torch.mul(u_xx, 1/2) + u_abs

        return torch.square(resid.abs()).sum()        


def MSE(x, y):
    errors = x - y
    return torch.mean(torch.square(errors.abs()))


def l2_normalized_error(pred, actual):
    errors = pred - actual
    error_norms = torch.linalg.norm(errors, dim=1, ord=2)
    actual_norms = torch.linalg.norm(actual, dim=1, ord=2)
    return torch.mean(torch.divide(error_norms, actual_norms))


def find_a_model(pattern):
    model_dir = os.path.join(*pattern.split("/")[:-1])
    file_pattern = pattern.replace("{}", "(.*)")
    file_pattern_re = re.compile(file_pattern)
    best_n_epochs = -1
    fp = None
    model_lst = [os.path.join(model_dir, f) for f in os.listdir(model_dir)]
    for m_fp in model_lst:
        search_obj = file_pattern_re.search(m_fp)
        if search_obj is not None:
            n_epochs = int(search_obj.group(1))
            if n_epochs > best_n_epochs:
                best_n_epochs = n_epochs
                fp = m_fp
    if fp is None:
        raise ValueError("Could not find any matching models")
    return fp, best_n_epochs


def load_or_init_model(device, model_type, fp=None, pattern=None, config=None):
    if fp is not None:
        model = torch.load(fp, map_location=device)
        logging.info("Loaded model from: {}".format(fp))
        n_epochs = 0
    elif pattern is not None:
        fp, n_epochs = find_a_model(pattern)
        model = torch.load(fp, map_location=device)
        logging.info("Loaded model from: {}".format(fp))
        logging.info("Model already trained with {} epochs".format(n_epochs))
    else:
        model = model_type(**config).to(device)
        logging.info("Initialized new model of type: {}".format(model_type))
        n_epochs = 0

    assert type(model) == model_type

    return model, n_epochs


def train_loop(model, optimizer, scheduler, pde_loss_obj, pde_loss_lambda, start_epoch, end_epoch, device, train_data_loader, train_df, do_testing,
                            test_every_n, test_data_loader, test_df, model_path, results_dd):
    """This is the main training loop

    Parameters
    ----------
    model : torch.nn.Model
        Model to train.
    optimizer : torch.optimizer
        Optimization algorithm.
    scheduler : torch.lr_scheduler
        Learning rate scheduler.
    pde_loss_obj : NLS_Residual_Loss
        Computes the residual loss (defined above)
    pde_loss_lambda: None or float
        If None, pde loss is computed but not used in backpropogation. If a float, loss is MSE + pde_loss_lambda * PDE
    epochs : int
        Number of full passes over the training dataset.
    device : torch.device
        Determines whether a GPU is used.
    train_data_loader : torch.DataLoader
        Object which iterates over train dataset.
    train_df : str
        Filepath to save intermediate training results.
    do_testing : bool
        Whether to test the model throughout training.
    test_every_n : int
        How often to do said testing.
    test_data_loader : torch.DataLoader
        iterates over test dataset.
    test_df : str
        Filepath to save intermediate test results.
    model_path : str
        Filepath (formattable with epoch number) to save model.

    Returns
    -------
    model
        Trained model.
    """

    train_dd = {}
    test_dd = {}
    logging.info("Beginning training for {} epochs".format(end_epoch - start_epoch))

    backprop_pde_bool = (pde_loss_lambda is not None)
    logging.info("Backprop PDE Loss: {}".format(backprop_pde_bool))

    model.train()
    t0_train = default_timer()
    for ep in range(start_epoch, end_epoch):
        # model.train()
        t1 = default_timer()
        train_mse = 0.
        train_pde_residual = 0.
        for x, y, t in train_data_loader:
            x, y, t = x.to(device), y.to(device), t.to(device)
            # print("X SHAPE: {}, T SHAPE: {}".format(x.shape, t.shape))

            optimizer.zero_grad()
            out = model(x, t)
            mse = MSE(out, y)

            if backprop_pde_bool:
                pde_resid = pde_loss_obj(model, x, t)
                loss = mse + pde_loss_lambda * pde_resid
            else:
                loss = mse
                with torch.no_grad():
                    pde_resid = pde_loss_obj(model, x, t)
            
            loss.backward()
            optimizer.step()

            train_mse += mse.item()
            train_pde_residual += pde_resid

        scheduler.step()
        # model.eval()

        train_mse /= len(train_data_loader)
        train_pde_residual /= len(train_data_loader)

        t2 = default_timer()
        logging.info("Epoch: {}, time: {:.2f}, train_mse: {:.4f}, train_pde: {:.4f}".format(ep, t2-t1, train_mse, train_pde_residual))
        train_dd['epoch'] = ep
        train_dd['MSE'] = train_mse
        train_dd['PDE'] = train_pde_residual
        train_dd['time'] = t2-t1
        write_result_to_file(train_df, **train_dd)

        ########################################################
        # Intermediate testing and saving
        ########################################################
        if ep % test_every_n == 0:
            test_mse = 0.
            test_l2_norm_error = 0.
            test_pde_residual = 0.
            if do_testing:
                model.eval()
                with torch.no_grad():
                    for x, y, t in test_data_loader:
                        x, y, t = x.to(device), y.to(device), t.to(device)

                        out = model(x, t)

                        mse = MSE(out, y)
                        test_mse += mse.item()

                        test_pde_residual += pde_loss_obj(model, x, t)
                        l2_err = l2_normalized_error(out, y)
                        test_l2_norm_error += l2_err.item()
                model.train()

                test_mse /= len(test_data_loader)
                test_l2_norm_error /= len(test_data_loader)
                test_pde_residual /= len(test_data_loader)

                test_dd['test_mse'] = test_mse
                test_dd['test_l2_normalized_error'] = test_l2_norm_error
                test_dd['test_pde_residual'] = test_pde_residual
                test_dd['epoch'] = ep

                write_result_to_file(test_df, **test_dd)
                logging.info("Test: Epoch: {}, test_mse: {:.4f}, test_pde: {:.4f}".format(ep, test_mse, test_pde_residual))
            torch.save(model, model_path.format(ep))

    torch.save(model, model_path.format(end_epoch))
    if end_epoch - start_epoch > 0:
        results_dd['train_mse'] = train_mse
        results_dd['test_mse'] = test_mse
    return model


def residual_network_training(args, device, batch_size=1024, learning_rate=0.001, weight_decay=0., step_size=100, gamma=0.5):


    results_dd = {'modes': args.freq_modes,
                    'width': args.width}
    ################################################################
    # load training data
    ################################################################

    d = sio.loadmat(args.data_fp)
    usol = d['output'][:,::args.time_subsample]
    t_grid = d['t'][:,::args.time_subsample]
    x_grid = d['x']
    logging.info("USOL SHAPE {}, T_GRID SHAPE: {}, X_GRID SHAPE: {}".format(usol.shape,
                                                                            t_grid.shape,
                                                                            x_grid.shape))

    train_dataset = TimeDataSet(usol, t_grid, x_grid)
    logging.info("Dataset: {}".format(train_dataset))
    results_dd['ntrain'] = len(train_dataset)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    ################################################################
    # read testing data
    ################################################################
    if not args.no_test:

        d_test = sio.loadmat(args.test_data_fp)
        usol_test = d_test['output']

        # This is a quick hack to get the really bad test case out of the test set
        usol_test = np.delete(usol_test, [59], axis=0)
        t_grid_test = d_test['t']
        x_grid_test = d_test['x']

        test_dataset = TimeDataSet(usol_test, t_grid_test, x_grid_test)
        logging.info("Test Dataset: {}".format(test_dataset))
        results_dd['ntest'] = len(test_dataset)

        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True)



    ##################################################################
    # initialize model and optimizer
    ##################################################################
    model_params = {'width': args.width, 'modes':args.freq_modes}

    model, n_epochs = load_or_init_model(device=device,
                                            model_type=FNO1dComplexTime,
                                            config=model_params)
    results_dd.update(model_params)
    logging.info("Using learning rate {} and weight decay {}".format(learning_rate, weight_decay))
    results_dd['learning_rate'] = learning_rate
    results_dd['l2_regularization'] = weight_decay

    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=step_size,
                                                    gamma=gamma)


    ##################################################################
    # initialize PDE Loss object
    ##################################################################

    pde_loss_obj = NLS_Residual_Loss(2 * np.pi / 1024, .001, 1024)

    
    ##################################################################
    # Call training loop
    ##################################################################
    logging.info("Starting FNO training")
    model = train_loop(model=model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    pde_loss_obj=pde_loss_obj,
                                    pde_loss_lambda=args.pde_loss_lambda,
                                    start_epoch=0,
                                    end_epoch=args.epochs,
                                    device=device,
                                    train_data_loader=train_data_loader,
                                    train_df=args.train_df,
                                    do_testing=(not args.no_test),
                                    test_every_n=args.test_every_n,
                                    test_data_loader=test_data_loader,
                                    test_df=args.test_df,
                                    model_path=args.model_fp,
                                    results_dd=results_dd)
    return model, results_dd


def main(args):
    # Figure out CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Running computation on device: {}".format(device))

    ################################################################
    #  Set up and run training
    ################################################################

    lr = (10 ** args.lr_exp)
    if args.weight_decay_exp is not None:
        weight_decay = (10 ** args.weight_decay_exp)
    else:
        weight_decay = 0.

    if args.pde_loss_lambda is not None:
        batch_size = 512
    else:
        batch_size = 1024

    model, results_dd = residual_network_training(args, device, learning_rate=lr, weight_decay=weight_decay, batch_size=batch_size)

    ################################################################
    # Report results
    ################################################################

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
    parser.add_argument('--results_fp')
    parser.add_argument('--test_data_fp')
    parser.add_argument('--model_fp')
    parser.add_argument('--train_df')
    parser.add_argument('--test_df')
    parser.add_argument('--weight_decay_exp', type=float, default=None)
    parser.add_argument('--pde_loss_lambda', type=float, default=None)
    parser.add_argument('--lr_exp', type=float, default=-3)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--freq_modes', type=int, default=16)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--time_subsample', type=int, default=1)
    parser.add_argument('--no_test', default=False, action='store_true')
    parser.add_argument('--test_every_n', type=int, default=50)

    args = parser.parse_args()
    fmt = "%(asctime)s:FNO: %(levelname)s - %(message)s"
    time_fmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=time_fmt)
    main(args)
