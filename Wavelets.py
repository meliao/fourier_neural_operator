import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarDWT(nn.Module):
    """Short summary.

    Attributes
    ----------
    c_filter : pytorch Tensor
        Filter to produce c_i coefficients in DWT. Sometimes referred to as h_0.
    d_filter : type
        Filter to produce d_i coefficients in DWT. Sometimes referred to as h_1.
    padder : pytorch nn Module
        Appropriately pads odd-length arrays
    level : int
        Level of the DWT
    """
    def __init__(self, level=1):
        super().__init__()
        self.c_filter = torch.tensor(np.divide(np.array([1., 1.]),
                                                np.sqrt(2))).reshape((1,1,2))
        self.d_filter = torch.tensor(np.divide(np.array([1., -1.]),
                                                np.sqrt(2))).reshape((1,1,2))
        self.filter_len = 2
        self.level = level

        # If the input array is odd-length, pad the array by repeating the
        # last element
        self.padder = nn.ReplicationPad1d((0,1))

    def max_dwt_level(self, data_len):
        """
        This is a function to compute the maximum level DWT that is possible
        on a 1D input of length data_len. This formula is copied from
        PyWavelets: https://tinyurl.com/y9u7yvbw
        """
        return np.floor(np.log2(data_len / (self.filter_len - 1)))

    def forward(self, x):
        xlen = x.size()[-1]
        if self.level > self.max_dwt_level(xlen):
            raise ValueError("Input array length {} gives max DWT level {}".format(xlen, self.max_dwt_level(xlen)))

        

        # To get odd-length arrays matching pywt output, we need to repeat the
        # last element of the array
        if x.shape[-1] % 2:
            x = self.padder(x)

        c_out = F.conv1d(x, self.c_filter, stride=2)
        d_out = F.conv1d(x, self.d_filter, stride=2)
        out = torch.cat((c_out, d_out), axis=-1)
        return out
