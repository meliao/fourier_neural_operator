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
        self.d_arrays['000_ones'] = np.ones(128)
        self.d_arrays['001_ones'] = np.ones(256)
        self.d_arrays['002_zeros'] = np.zeros(128)
        self.d_arrays['003_zeros'] = np.zeros(256)
        self.d_arrays['004_linspace'] = np.linspace(0, 10, 512)
        self.d_arrays['005_linspace'] = np.linspace(0, 10, 1024)
        self.d_arrays['006_sin'] = np.sin(np.linspace(0, 10, 512))
        self.d_arrays['007_cos'] = np.cos(np.linspace(0, 10, 1024))
        self.d_arrays['008_abs'] = np.abs(np.linspace(-33, 33, 1024))

        # r_arrays = random data arrays
        self.r_arrays = {}
        self.r_arrays['009_runif'] = np.random.uniform(size=1024)
        self.r_arrays['010_rnorm'] = np.random.normal(size=1024)


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

    @staticmethod
    def pywt_reference_HaarDWT_leveln(arr, n, verbose=False):
        coeffs = pywt.wavedec(arr, 'haar', level=n)
        return np.concatenate(coeffs)

    @staticmethod
    def pywt_reference_IHaarDWT_level1(arr):
        l = arr.shape[-1]
        c = arr[:int(l/2)]
        d = arr[int(l/2):]
        out = pywt.idwt(c, d, 'haar')
        return out

    def test_DWTinit(self):
        DWT_obj = Wavelets.HaarDWT()
        self.assertIsInstance(DWT_obj, Wavelets.HaarDWT)

    def test_IDWTinit(self):
        IDWT_obj = Wavelets.IHaarDWT()
        self.assertIsInstance(IDWT_obj, Wavelets.IHaarDWT)

    def test_bad_input_length_DWT(self):
        DWT_obj = Wavelets.HaarDWT()
        x = torch.zeros((1,1,1+1024))
        with self.assertRaises(ValueError):
            y = DWT_obj(x)
    

    def testHaarDWTLevel1_deterministic(self):
        DWT_obj = Wavelets.HaarDWT()
        for k, v in self.d_arrays.items():
            with self.subTest(k=k):
                v_ten = self.numpy_to_torch(v)
                ans = self.torch_to_numpy(DWT_obj(v_ten))
                ref = self.pywt_reference_HaarDWT_level1(v)
                self.assertTrue(np.allclose(ref, ans), k)

    def testHaarDWTLeveln_deterministic(self):
        for k, v in self.d_arrays.items():
            for level in [1,2,3,4]:
                with self.subTest(k=k, level=level):
                    DWT_obj = Wavelets.HaarDWT(level=level)
                    v_ten = self.numpy_to_torch(v)
                    ans = self.torch_to_numpy(DWT_obj(v_ten))
                    ref = self.pywt_reference_HaarDWT_leveln(v, n=level)
                    self.assertTrue(np.allclose(ref, ans), k)

    def testHaarDWTLeveln_random(self):
        for k, v in self.r_arrays.items():
            for level in [1,2,3,4]:
                with self.subTest(k=k, level=level):
                    DWT_obj = Wavelets.HaarDWT(level=level)
                    v_ten = self.numpy_to_torch(v)
                    ans = self.torch_to_numpy(DWT_obj(v_ten))
                    ref = self.pywt_reference_HaarDWT_leveln(v, n=level)
                    self.assertTrue(np.allclose(ref, ans), k)

    def testHaarDWTLevel1_random(self):
        DWT_obj = Wavelets.HaarDWT()
        for k, v in self.r_arrays.items():
            with self.subTest(k=k):
                v_ten = self.numpy_to_torch(v)
                ans = self.torch_to_numpy(DWT_obj(v_ten))
                ref = self.pywt_reference_HaarDWT_level1(v)
                self.assertTrue(np.allclose(ref, ans), k)

    def testIHaarDWTLevel1_deterministic(self):
        IDWT_obj = Wavelets.IHaarDWT()
        for k, v in self.d_arrays.items():
            with self.subTest(k=k):
                dwt = self.pywt_reference_HaarDWT_level1(v)
                dwt_ten = self.numpy_to_torch(dwt)
                ans = self.torch_to_numpy(IDWT_obj(dwt_ten))
                ref = self.pywt_reference_IHaarDWT_level1(dwt)
                self.assertTrue(np.allclose(ref, ans), k)

    def testIHaarDWTLevel1_random(self):
        IDWT_obj = Wavelets.IHaarDWT()
        for k, v in self.r_arrays.items():
            with self.subTest(k=k):
                dwt = self.pywt_reference_HaarDWT_level1(v)
                dwt_ten = self.numpy_to_torch(dwt)
                ans = self.torch_to_numpy(IDWT_obj(dwt_ten))
                ref = self.pywt_reference_IHaarDWT_level1(dwt)
                self.assertTrue(np.allclose(ref, ans), k)

    def testIHaarDWTLeveln_deterministic(self):
        for k, v in self.d_arrays.items():
            for level in [1,2,3,4]:
                with self.subTest(k=k, level=level):
                    v_ten = self.numpy_to_torch(v)
                    DWT_obj = Wavelets.HaarDWT(level=level)
                    IDWT_obj = Wavelets.IHaarDWT(level=level)
                    dwt_ten = DWT_obj(v_ten)
                    ans = self.torch_to_numpy(IDWT_obj(dwt_ten))
                    self.assertTrue(np.allclose(v, ans), k)

    def testIHaarDWTLeveln_random(self):
        for k, v in self.r_arrays.items():
            for level in [1,2,3,4]:
                with self.subTest(k=k, level=level):
                    v_ten = self.numpy_to_torch(v)
                    DWT_obj = Wavelets.HaarDWT(level=level)
                    IDWT_obj = Wavelets.IHaarDWT(level=level)
                    dwt_ten = DWT_obj(v_ten)
                    ans = self.torch_to_numpy(IDWT_obj(dwt_ten))
                    self.assertTrue(np.allclose(v, ans), k)
