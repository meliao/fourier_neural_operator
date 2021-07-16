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

from NLS_Residual_Loss import FNO1dComplexTime, SpectralConv1d, NLS_Residual_Loss

model_fp = '/home-nfs/meliao/projects/fourier_neural_operator/experiments/10_linear_approx/models/one_step_lr_exp_-1.5_l2_exp_-2.5_ep_400'
model = torch.load(model_fp, map_location=torch.device('cuda'))


new_model = FNO1dComplexTime(modes=8, width=64).to(torch.device('cuda'))

def compute_and_backprop_loss(model, batch_num, n_grid, device):
    # n_grid = 1024
    dx = 2 * np.pi / n_grid
    loss_obj = NLS_Residual_Loss(dx, n_grid, batch_num, device)

    x = torch.rand((batch_num, n_grid, 3)).to(device)
    t = torch.ones((batch_num)).to(device)

    loss_val = loss_obj(model, x, t)
    t0 = default_timer()
    loss_val.backward()
    t1 = default_timer()
    print("Backward pass in {:.4f}".format(t1-t0))


def time_forward_backward(n_batches, n_grid, model, device):
    time_lst = []
    n_rounds = 10

    for i in range(10):
        t0 = default_timer()
        compute_and_backprop_loss(model, n_batches, n_grid, device)
        t1 = default_timer()
        time_lst.append(t1-t0)
        # print("Done with", i)
    print("###################################")
    print("Batch size: {}, n_grid: {}, mean: {}, std: {}".format(n_batches, n_grid, np.mean(time_lst), np.std(time_lst)))
    print("###################################")


time_forward_backward(1, 512, new_model, torch.device('cuda'))

