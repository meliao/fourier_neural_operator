import os
import unittest
import numpy as np
import torch
import pywt #PyWavelets is treated as the reference implementation

import Wavelets

class TestHaarDWT(unittest.TestCase):
    def setUp(self):
        # a_arrays = tested against pywt reference for DWT and IDWT
        self.a_arrays = {}
        self.a_arrays['000_ones'] = np.ones((1,1,128))
        self.a_arrays['001_ones'] = np.ones((1,1,256))
        self.a_arrays['002_zeros'] = np.zeros((1,1,128))
        self.a_arrays['003_zeros'] = np.zeros((1,1,256))
        self.a_arrays['004_linspace'] = np.linspace(0, 10, 512).reshape((1,1,512))
        self.a_arrays['005_linspace'] = np.linspace(0, 10, 1024).reshape((1,1,1024))
        self.a_arrays['006_sin'] = np.sin(np.linspace(0, 10, 512)).reshape((1,1,512))
        self.a_arrays['007_cos'] = np.cos(np.linspace(0, 10, 1024)).reshape((1,1,1024))
        self.a_arrays['008_abs'] = np.abs(np.linspace(-33, 33, 1024)).reshape((1,1,1024))

        # b_arrays = random data arrays
        self.b_arrays = {}
        self.b_arrays['009_runif'] = np.random.uniform(size=10240).reshape((2,5,1024))
        self.b_arrays['010_runif'] = np.random.uniform(size=5120).reshape((5,2,512))
        self.b_arrays['011_runif'] = np.random.uniform(size=11520).reshape((9,10,128))
        self.b_arrays['010_rnorm'] = np.random.normal(size=10240).reshape((10,1,1024))


    @staticmethod
    def torch_to_numpy(ten):
        return ten.numpy()

    @staticmethod
    def numpy_to_torch(arr):
        return torch.from_numpy(arr).float()

    @staticmethod
    def pywt_reference_HaarDWT_level1(arr):
        original_arr_shape = arr.shape
        arr = arr.flatten()
        c,d = pywt.dwt(arr, 'haar')
        return np.concatenate((c,d)).reshape(original_arr_shape)

    @staticmethod
    def pywt_reference_HaarDWT_leveln(arr, n, verbose=False):
        original_arr_shape = arr.shape
        arr = arr.flatten()
        coeffs = pywt.wavedec(arr, 'haar', level=n)
        return np.concatenate(coeffs).reshape(original_arr_shape)

    @staticmethod
    def pywt_reference_IHaarDWT_level1(arr):
        original_arr_shape = arr.shape
        arr = arr.flatten()
        l = arr.shape[-1]
        c = arr[:int(l/2)]
        d = arr[int(l/2):]
        out = pywt.idwt(c, d, 'haar').reshape(original_arr_shape)
        return out

    def assert_close_and_show_diff(self, ref, ans):
        errors = ref - ans
        errors = errors.flatten()
        l_inf = np.linalg.norm(errors, ord=np.inf)
        l_2 = np.linalg.norm(errors, ord=2)
        s = "Errors: L_2 = {:.03e} L_inf = {:.03e}".format(l_2, l_inf)
        self.assertTrue(np.allclose(ref, ans, atol=5e-06), s)

    def get_ref_ans_DWT(self, arr, level=1):
        DWT_obj = Wavelets.HaarDWT(arr.shape[1], arr.shape[2], level=level)
        dwt_ref = self.pywt_reference_HaarDWT_leveln(arr, n=level)
        arr_ten = self.numpy_to_torch(arr)
        dwt_ans_ten = DWT_obj(arr_ten)
        dwt_ans = self.torch_to_numpy(dwt_ans_ten)
        return dwt_ref, dwt_ans

    def get_ref_ans_IDWT(self, arr):
        IDWT_obj = Wavelets.IHaarDWT(arr.shape[1], arr.shape[2], level=1)
        dwt_ref = self.pywt_reference_HaarDWT_level1(arr)
        idwt_ref = self.pywt_reference_IHaarDWT_level1(dwt_ref)
        dwt_ten = self.numpy_to_torch(dwt_ref)
        idwt_ans_ten = IDWT_obj(dwt_ten)
        idwt_ans = self.torch_to_numpy(idwt_ans_ten)

        return idwt_ref, idwt_ans

    def get_DWT_IDWT_identity(self, arr):
        DWT_obj = Wavelets.HaarDWT(arr.shape[1], arr.shape[2])
        IDWT_obj = Wavelets.IHaarDWT(arr.shape[1], arr.shape[2])
        arr_ten = self.numpy_to_torch(arr)
        dwt_ten = DWT_obj(arr_ten)
        tensor_print(dwt_ten, "DWT_TEN")
        idwt_ten = IDWT_obj(dwt_ten)
        return self.torch_to_numpy(idwt_ten)


    def test_DWTinit(self):
        DWT_obj = Wavelets.HaarDWT(width=4, input_len=4)
        self.assertIsInstance(DWT_obj, Wavelets.HaarDWT)

    def test_IDWTinit(self):
        IDWT_obj = Wavelets.IHaarDWT(width=4, input_len=4)
        self.assertIsInstance(IDWT_obj, Wavelets.IHaarDWT)

    def test_bad_input_length_DWT(self):
        with self.assertRaises(ValueError):
            DWT_obj = Wavelets.HaarDWT(1, 1+1024)

    def test_dwt_against_reference(self):
        for k, v in self.a_arrays.items():
            for level in [1,2,3,4]:
                with self.subTest(k=k, level=level):
                    dwt_ref, dwt_ans = self.get_ref_ans_DWT(v, level=level)
                    self.assert_close_and_show_diff(dwt_ref, dwt_ans)

    def test_idwt_against_reference(self):
        for k, v in self.a_arrays.items():
            with self.subTest(k=k):
                idwt_ref, idwt_ans = self.get_ref_ans_IDWT(v)
                self.assert_close_and_show_diff(idwt_ref, idwt_ans)

    def test_invertibility(self):
        for k, v in self.b_arrays.items():
            for level in [1,2,3,4,None]:
                with self.subTest(k=k, level=level):
                    DWT_obj = Wavelets.HaarDWT(v.shape[1], v.shape[2], level=level)
                    IDWT_obj = Wavelets.IHaarDWT(v.shape[1], v.shape[2], level=level)
                    v_ten = self.numpy_to_torch(v)
                    dwt_ten = DWT_obj(v_ten)
                    idwt_ten = IDWT_obj(dwt_ten)
                    idwt = self.torch_to_numpy(idwt_ten)
                    self.assert_close_and_show_diff(idwt, v)
