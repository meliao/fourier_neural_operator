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
    def __init__(self, width, input_len, level=None, device=None):
        super().__init__()

        self.filter_len = 2

        if level is not None:
            self.level = level
        else:
            self.level = self.max_dwt_level(input_len)
        self.input_len = input_len
        self.width = width

        # This is bit manipulation to assert that xlen is a power of 2
        if not ((self.input_len & (self.input_len - 1) == 0) and self.input_len != 0):
            raise ValueError("Input array length {} is not power of 2".format(self.input_len))
        if self.level > self.max_dwt_level(self.input_len):
            s = "Input array length {} gives max DWT level {}".format(self.input_len,
                                                                      self.max_dwt_level(self.input_len))
            raise ValueError(s)


        self.c_filter = torch.tensor(np.divide(np.array([1., 1.]),
                                                np.sqrt(2)), dtype=torch.float).reshape((1,1,2))
        self.d_filter = torch.tensor(np.divide(np.array([1., -1.]),
                                                np.sqrt(2)), dtype=torch.float).reshape((1,1,2))
        if device is not None:
            self.c_filter = self.c_filter.to(device)
            self.d_filter = self.d_filter.to(device)

        # If the input array is odd-length, pad the array by repeating the
        # last element
        self.padder = nn.ReplicationPad1d((0,1))

    def max_dwt_level(self, data_len):
        """
        This is a function to compute the maximum level DWT that is possible
        on a 1D input of length data_len. This formula is copied from
        PyWavelets: https://tinyurl.com/y9u7yvbw
        """
        return int(np.floor(np.log2(data_len / (self.filter_len - 1))))

    def filter(self, x):
        xlen = x.size()[-1]


        # We want the filter to work independently on each element of the batch
        # and each width channel.
        x = x.reshape((-1, 1, xlen))

        c_out = F.conv1d(x, self.c_filter, stride=2)
        d_out = F.conv1d(x, self.d_filter, stride=2)
        out = torch.cat((c_out, d_out), axis=-1)
        out = out.reshape((-1, self.width, xlen))
        return out


    def forward(self, x):
        """
        Expects input of size (batch_n, width, input_len) where
            batch_num unconstrained and xlen is a power of 2
        """

        for l in range(self.level):
            level_i_arr_idx = int(self.input_len / (2 ** l))
            x_out = x.clone()
            x_in = x[:,:,:level_i_arr_idx]
            x_out[:,:,:level_i_arr_idx] = self.filter(x_in)
            x = x_out
        return x


class IHaarDWT(nn.Module):
    """Short summary.

    Parameters
    ----------
    width : type
        Description of parameter `width`.
    input_len : type
        Description of parameter `input_len`.
    level : type
        Description of parameter `level`.
    device : type
        Description of parameter `device`.

    Attributes
    ----------
    filter_len : type
        Description of attribute `filter_len`.
    max_dwt_level : type
        Description of attribute `max_dwt_level`.
    c_filter : type
        Description of attribute `c_filter`.
    d_filter : type
        Description of attribute `d_filter`.
    level
    input_len
    width

    """
    def __init__(self, width, input_len, level=None, device=None):
        super().__init__()

        self.filter_len = 2

        if level is not None:
            self.level = level
        else:
            self.level = self.max_dwt_level(input_len)
        self.input_len = input_len
        self.width = width

        # This is bit manipulation to assert that xlen is a power of 2
        if not ((self.input_len & (self.input_len - 1) == 0) and self.input_len != 0):
            raise ValueError("Input array length {} is not power of 2".format(self.input_len))
        if self.level > self.max_dwt_level(self.input_len):
            s = "Input array length {} gives max IDWT level {}".format(self.input_len,
                                                                      self.max_dwt_level(self.input_len))
            raise ValueError(s)
        self.c_filter = torch.tensor(np.divide(np.array([1., 1.]),
                                                np.sqrt(2)), dtype=torch.float).reshape((1,2,1))
        self.d_filter = torch.tensor(np.divide(np.array([1., -1.]),
                                                np.sqrt(2)), dtype=torch.float).reshape((1,2,1))
        if device is not None:
            self.c_filter = self.c_filter.to(device)
            self.d_filter = self.d_filter.to(device)


    def max_dwt_level(self, data_len):
        """
        This is a function to compute the maximum level DWT that is possible
        on a 1D input of length data_len. This formula is copied from
        PyWavelets: https://tinyurl.com/y9u7yvbw
        """
        return int(np.floor(np.log2(data_len / (self.filter_len - 1))))

    def unfilter(self, x):
        """Perform a Level-1 Inverse Haar DWT via filter banks. Performs this
        transformation along the last axis. The elements of the first 2 axes are
        treated completely independently

        Parameters
        ----------
        x : pytorch Tensor
            Assumed to be of size (batch_size, self.width, xlen) where xlen is
            a power of 2

        Returns
        -------
        pytorch Tensor
            output size (batch_size, self.width, xlen).

        """

        # X has shape (batch_num, self.width, xlen)
        xlen = x.size()[-1]
        batch_num = x.size()[0]

        # We want X to have shape (batch_num, self.width, 2, xlen/2)
        x = x.reshape((batch_num, self.width, 2, int(xlen/2))) # .permute(0,1,3,2)

        # The IDWT works independently on each element along the batch and
        # width axes, so we're combining those two.
        # x now has shape (batch_num * self.width, 2, xlen / 2)
        x = x.flatten(start_dim=0, end_dim=1)
        c_out = F.conv1d(x, self.c_filter)
        d_out = F.conv1d(x, self.d_filter)

        # This code is messy, but in essence, it takes c_out and d_out, each
        # with shapes (batch_num * self.width, 2, xlen / 2), and interleaves
        # along the last axis. So the output array has shape (batch_num * self.width, 1, xlen)
        out = torch.cat((c_out, d_out), dim=1).permute(0,2,1).flatten(start_dim=1)

        # This undoes the combining of the batch and width axes. Output shape
        # now is (batch_num, self.width, xlen) == input shape.
        out = out.reshape((batch_num, self.width, xlen))
        return out

    def forward(self, x):
        """Perform a Inverse Haar DWT via filter banks. Performs this
        transformation along the last axis. The elements of the first 2 axes are
        treated completely independently. The level of the IDWT is determined
        by self.level

        Parameters
        ----------
        x : pytorch Tensor
            Assumed to be of size (batch_size, self.width, self.input_len)

        Returns
        -------
        pytorch Tensor
            output size (batch_size, self.width, self.input_len).

        """
        for l in range(self.level-1, -1, -1):
            level_i_arr_idx = int(self.input_len / (2 ** l))
            x_out = x.clone()
            x_in = x[:,:,:level_i_arr_idx]
            x_out[:,:,:level_i_arr_idx] = self.unfilter(x_in)
            x = x_out
        return x


class WaveletBlock1d(nn.Module):
    def __init__(self, width, input_len, keep, device=None):
        super(WaveletBlock1d, self).__init__()
        self.input_len = input_len
        self.width = width
        self.keep = keep

        self.DWT = HaarDWT(width=width, input_len=input_len, device=device)
        self.IDWT = IHaarDWT(width=width, input_len=input_len, device=device)

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
