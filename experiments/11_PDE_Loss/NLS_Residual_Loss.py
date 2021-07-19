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
# import h5py

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


class NLS_Residual_Loss:
    """
    NLS: i u_t + 1 / 2 * u_xx + |u|^2 u = 0

    """
    def __init__(self, delta_x, n_grid_points, batch_size, device):
        self.delta_x = delta_x
        self.n_grid_points = n_grid_points
        self.batch_size = batch_size
        self.I = torch.eye(self.batch_size).to(device)
        self.imag = torch.tensor(0+1j, dtype=torch.cfloat).repeat((self.batch_size, self.n_grid_points)).to(device)


    def time_derivative(self, model, x, t):
        jac_t = torch.autograd.functional.jacobian(lambda t: model(x,t), t, create_graph=True, vectorize=False)
        # (batch_size x grid_size x batch_size) * (batch_size x batch_size) -> (batch_size x grid_size)
        return torch.einsum('bgb,bb->bg', jac_t, self.I)


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
        t0 = default_timer()
        u = model(x,t)
        t1 = default_timer()
        print("Forward pass in {:.4f}".format(t1-t0))

        t0 = default_timer()
        u_abs = torch.mul(u, torch.square(torch.abs(u)))
        t1 = default_timer()
        print("PDE Loss Nonlin Term in {:.4f}".format(t1-t0))
        t0 = default_timer()
        u_t = self.time_derivative(model, x, t)
        t1 = default_timer()
        print("PDE Loss autodiff u_t term in {:.4f}".format(t1-t0))

        t0 = default_timer()
        u_xx = self.spatial_discrete_derivatives(u)
        t1 = default_timer()
        print("PDE Loss lin term in {:.4f}".format(t1-t0))
        resid = torch.mul(self.imag, u_t) + torch.mul(u_xx, 1/2) + u_abs

        return torch.abs(resid).sum()        
        
