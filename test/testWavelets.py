import os
import unittest
import numpy as np
import torch
import pywt #PyWavelets is treated as the reference implementation

import Wavelets

class TestHaarDWT(unittest.TestCase):
    def setUp(self):
        # d_arrays = deterministic arrays. As opposed to tests on random data
        self.d_arrays = {}
        self.d_arrays['000_ones_even'] = np.ones(100)
        self.d_arrays['001_ones_odd'] = np.ones(101)
        self.d_arrays['002_zeros_even'] = np.zeros(100)
        self.d_arrays['003_zeros_odd'] = np.zeros(101)
        self.d_arrays['004_linspace_even'] = np.linspace(0, 10, 100)
        self.d_arrays['005_linspace_odd'] = np.linspace(0, 10, 101)
        self.d_arrays['006_sin'] = np.sin(np.linspace(0, 10, 100))
        self.d_arrays['007_cos'] = np.cos(np.linspace(0, 10, 100))
        self.d_arrays['008_abs'] = np.abs(np.linspace(-33, 33, 100))

        # r_arrays = random data arrays
        self.r_arrays = {}
        self.r_arrays['009_runif'] = np.random.uniform(size=100)
        self.r_arrays['010_rnorm'] = np.random.normal(size=100)


    @staticmethod
    def torch_to_numpy(ten):
        s = ten.size()[-1]
        return ten.numpy().reshape(s)

    @staticmethod
    def numpy_to_torch(arr):
        s = arr.shape[-1]
        return torch.tensor(arr).reshape((1,1,s))

    @staticmethod
    def pywt_reference_HaarDWT_level1(arr):
        c,d = pywt.dwt(arr, 'haar')
        return np.concatenate((c,d))

    def test_init(self):
        DWT_obj = Wavelets.HaarDWT()
        self.assertIsInstance(DWT_obj, Wavelets.HaarDWT)

    def testHaarDWTLevel1_deterministic(self):
        DWT_obj = Wavelets.HaarDWT()
        for k, v in self.d_arrays.items():
            with self.subTest(k=k):
                v_ten = self.numpy_to_torch(v)
                ans = self.torch_to_numpy(DWT_obj(v_ten))
                ref = self.pywt_reference_HaarDWT_level1(v)
                self.assertTrue(np.allclose(ref, ans), k)

    def testHaarDWTLevel1_random(self):
        DWT_obj = Wavelets.HaarDWT()
        for k, v in self.r_arrays.items():
            with self.subTest(k=k):
                v_ten = self.numpy_to_torch(v)
                ans = self.torch_to_numpy(DWT_obj(v_ten))
                ref = self.pywt_reference_HaarDWT_level1(v)
                self.assertTrue(np.allclose(ref, ans), k)
