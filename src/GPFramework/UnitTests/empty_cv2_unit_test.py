"""
Programmed by Austin Dunn
Contains the unit tests for opencv methods using empty arrays
"""
from GPFramework.data import EmadeDataInstance, EmadeDataPair, EmadeData, EmadeDataInstance, FeatureData, StreamData
import GPFramework.data as data
import os
import unittest
import numpy as np
np.random.seed(117)
import timeout_decorator
import GPFramework.spatial_methods as sp
import cv2

# Primitives now need to be called out of pset so we need these imports
from GPFramework.constants import FEATURES_TO_FEATURES, STREAM_TO_STREAM, STREAM_TO_FEATURES, Axis
import GPFramework.gp_framework_helper as gp_framework_helper
from deap import gp
import copy as cp
# This is outside the class because we only want it to happen once, not once per test
# Build a pset we can use in each primitve
pset = gp.PrimitiveSetTyped('MAIN', [EmadeDataPair], EmadeDataPair)
# Add the primitives and terminals to the pset
# Primitives must be added BEFORE terminals for STGP (strongly typed genetic programming)
gp_framework_helper.addPrimitives(pset)
gp_framework_helper.addTerminals(pset)

class EmptyUnitTest(unittest.TestCase):  #pylint: disable=R0904
    """
    This class contains the unit tests for OpenCV methods taking in empty arrays
    """
    @classmethod
    def setUpClass(cls):
        cls.data = np.zeros((4,0))
        stream = StreamData(cls.data)
        instances = [EmadeDataInstance(stream=stream)]
        data = EmadeData(instances)
        cls.data_pair = EmadeDataPair((cp.deepcopy(data), None), (cp.deepcopy(data), None))

        cls.data_8u = np.zeros((4,0)).astype(np.uint8)
        stream = StreamData(cls.data_8u)
        instances = [EmadeDataInstance(stream=stream)]
        data = EmadeData(instances)
        cls.data_pair_8u = EmadeDataPair((cp.deepcopy(data), None), (cp.deepcopy(data), None))

        cls.data_32f = np.zeros((4,0)).astype(np.float32)
        stream = StreamData(cls.data_32f)
        instances = [EmadeDataInstance(stream=stream)]
        data = EmadeData(instances)
        cls.data_pair_32f = EmadeDataPair((cp.deepcopy(data), None), (cp.deepcopy(data), None))

    def setUp(self):
        """
        Dereferences the self.<DataPair> object attributes from the cls.<DataPair> class attributes.
        """
        self.data_pair = cp.deepcopy(self.data_pair)
        self.data_pair_8u = cp.deepcopy(self.data_pair_8u)
        self.data_pair_32f = cp.deepcopy(self.data_pair_32f)

    def test_minimum_to_zero(self):
        """
        Test minimum_to_zero method
        :return:
        """
        print("minimum_to_zero")
        try:
            sp.minimum_to_zero(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
        except ValueError as e:
            self.assertEqual(str(e), "zero-size array to reduction operation minimum which has no identity")


    def test_edge_detection_canny(self):
        """
        Test edge_detection_canny method
        :return:
        """
        print("edge_detection_canny")
        try:
            sp.edge_detection_canny(self.data_pair, STREAM_TO_STREAM, Axis.FULL, t1=50, t2=150, apertureSize=3)
        except cv2.error:
            pass

    def test_corner_detection_harris(self):
        """
        Test corner_detection_harris method
        :return:
        """
        print("corner_detection_harris")
        try:
            sp.corner_detection_harris(self.data_pair, STREAM_TO_STREAM, Axis.FULL, 10, 5, 0.04)
        except cv2.error:
            pass

    def test_corner_detection_min_eigen_val(self):
        """
        Test corner_detection_min_eigen_val method
        :return:
        """
        print("corner_detection_min_eigen_val")
        try:
            sp.corner_detection_min_eigen_val(self.data_pair, STREAM_TO_STREAM, Axis.FULL, 10, 5)
        except cv2.error:
            pass

    def test_highpass_fourier_ellipsoid(self):
        """
        Test highpass_fourier_ellipsoid method
        :return:
        """
        print("highpass_fourier_ellipsoid")
        try:
            sp.highpass_fourier_ellipsoid(self.data_pair, STREAM_TO_STREAM, Axis.FULL, size=3)
        except ValueError as e:
            orig_str = str(e) == "invalid number of data points ([4 0]) specified"
            new_str = str(e) == "Invalid number of FFT data points (0) specified."
            self.assertTrue(orig_str | new_str)

    def test_highpass_irst(self):
        """
        Test highpass_irst method
        :return:
        """
        print("highpass_irst")
        try:
            sp.highpass_irst(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
        except ValueError as e:
            self.assertEqual(str(e), "Data array cannot be empty.")

    def test_median_filter(self):
        """
        Test median_filter method
        :return:
        """
        print("median_filter")
        result = sp.median_filter(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_size=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_lowpass_fourier_shift(self):
        """
        Test lowpass_fourier_shift method
        :return:
        """
        print("lowpass_fourier_shift")
        try:
            sp.lowpass_fourier_shift(self.data_pair, STREAM_TO_STREAM, Axis.FULL, shift=3)
        except ValueError as e:
            orig_str = str(e) == "invalid number of data points ([4 0]) specified"
            new_str = str(e) == "Invalid number of FFT data points (0) specified."
            self.assertTrue(orig_str | new_str)

    def test_highpass_fourier_shift(self):
        """
        Test highpass_fourier_shift method
        :return:
        """
        print("highpass_fourier_shift")
        try:
            sp.highpass_fourier_shift(self.data_pair, STREAM_TO_STREAM, Axis.FULL, shift=3)
        except ValueError as e:
            orig_str = str(e) == "invalid number of data points ([4 0]) specified"
            new_str = str(e) == "Invalid number of FFT data points (0) specified."
            self.assertTrue(orig_str | new_str)

    def test_highpass_fourier_gaussian(self):
        """
        Test highpass_fourier_gaussian method
        :return:
        """
        print("highpass_fourier_gaussian")
        try:
            sp.highpass_fourier_gaussian(self.data_pair, STREAM_TO_STREAM, Axis.FULL, sigma=3)
        except ValueError as e:
            orig_str = str(e) == "invalid number of data points ([4 0]) specified"
            new_str = str(e) == "Invalid number of FFT data points (0) specified."
            self.assertTrue(orig_str | new_str)

    def test_highpass_fourier_uniform(self):
        """
        Test highpass_fourier_uniform method
        :return:
        """
        print("highpass_fourier_uniform")
        try:
            sp.highpass_fourier_uniform(self.data_pair, STREAM_TO_STREAM, Axis.FULL, size=3)
        except ValueError as e:
            orig_str = str(e) == "invalid number of data points ([4 0]) specified"
            new_str = str(e) == "Invalid number of FFT data points (0) specified."
            self.assertTrue(orig_str | new_str)

    def test_highpass_unsharp_mask(self):
        """
        Test highpass_unsharp_mask method
        :return:
        """
        print("highpass_unsharp_mask")
        try:
            sp.highpass_unsharp_mask(self.data_pair, STREAM_TO_STREAM, Axis.FULL, 9, 10, 10)
        except ValueError as e:
            self.assertEqual(str(e), "Data array cannot be empty.")

    @timeout_decorator.timeout(10, use_signals=False)
    def test_highpass_laplacian(self):
        """
        TODO: Fix this unit test
        THIS METHOD HANGS. IT WILL NOT WORK.

        Test highpass_laplacian method
        :return:
        """
        print("highpass_laplacian")
        result = sp.highpass_laplacian(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_size=1, scale=1, delta=0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_highpass_sobel_derivative(self):
        """
        Test highpass_sobel_derivative method
        :return:
        """
        print("highpass_sobel_derivative")
        try:
            sp.highpass_sobel_derivative(self.data_pair, STREAM_TO_STREAM, Axis.FULL, dx=1, dy=1, ksize=5, scale=1, delta=0)
        except ValueError as e:
            self.assertEqual(str(e), "Data array cannot be empty.")

    def test_lowpass_filter_median(self):
        """
        Test lowpass_filter_median method
        :return:
        """
        print("lowpass_filter_median")
        try:
            result = sp.lowpass_filter_median(self.data_pair, STREAM_TO_STREAM, Axis.FULL, filter_size=5)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_lowpass_filter_average(self):
        """
        Test lowpass_filter_average method
        :return:
        """
        print("lowpass_filter_average")
        try:
            sp.lowpass_filter_average(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_size=5)
        except ValueError as e:
            self.assertEqual(str(e), "Data array cannot be empty.")

    def test_lowpass_filter_gaussian(self):
        """
        Test lowpass_filter_gaussian method
        :return:
        """
        print("lowpass_filter_gaussian")
        try:
            sp.lowpass_filter_gaussian(self.data_pair, STREAM_TO_STREAM, Axis.FULL, 3, 3, 0.5, 0.5)
        except ValueError as e:
            self.assertEqual(str(e), "Data array cannot be empty.")

    def test_lowpass_filter_bilateral(self):
        """
        TODO: Fix this unit test
        Test lowpass_filter_bilateral method
        :return:
        """
        print("lowpass_filter_bilateral")
        try:
            result = sp.lowpass_filter_bilateral(self.data_pair, STREAM_TO_STREAM, Axis.FULL, filter_diameter=9, sigma_color=75, sigma_space=75)
            self.assertIsInstance(result, data.EmadeDataPair)
        except cv2.error as e:
            condition = 'Bilateral filtering is only implemented for 8u and 32f' in str(e)
            self.assertTrue(condition)
        except ValueError as e:
            condition_0 = 'Primitive produced 0-d array' in str(e)
            condition_1 = 'Data array cannot be empty.' in str(e)
            condition = condition_0 | condition_1
            self.assertTrue(condition)
        # # Takes Forever, Figure out why
        # result = sp.lowpass_filter_bilateral(self.data_pair_8u, STREAM_TO_STREAM, Axis.FULL, filter_diameter=9, sigma_color=75, sigma_space=75)
        # self.assertIsInstance(result, data.EmadeDataPair)
        #
        # # Takes Forever, Figure out why
        # result = sp.lowpass_filter_bilateral(self.data_pair_32f, STREAM_TO_STREAM, Axis.FULL, filter_diameter=9, sigma_color=75, sigma_space=75)
        # self.assertIsInstance(result, data.EmadeDataPair)


    def test_lowpass_fourier_ellipsoid(self):
        """
        Test lowpass_fourier_ellipsoid method
        :return:
        """
        print("lowpass_fourier_ellipsoid")
        try:
            sp.lowpass_fourier_ellipsoid(self.data_pair, STREAM_TO_STREAM, Axis.FULL, size=3)
        except ValueError as e:
            orig_str = str(e) == "invalid number of data points ([4 0]) specified"
            new_str = str(e) == "Invalid number of FFT data points (0) specified."
            self.assertTrue(orig_str | new_str)

    def test_lowpass_fourier_gaussian(self):
        """
        Test lowpass_fourier_gaussian method
        :return:
        """
        print("lowpass_fourier_gaussian")
        try:
            sp.lowpass_fourier_gaussian(self.data_pair, STREAM_TO_STREAM, Axis.FULL, sigma=3)
        except ValueError as e:
            orig_str = str(e) == "invalid number of data points ([4 0]) specified"
            new_str = str(e) == "Invalid number of FFT data points (0) specified."
            self.assertTrue(orig_str | new_str)

    def test_lowpass_fourier_uniform(self):
        """
        Test lowpass_fourier_uniform method
        :return:
        """
        print("lowpass_fourier_uniform")
        try:
            sp.lowpass_fourier_uniform(self.data_pair, STREAM_TO_STREAM, Axis.FULL, size=3)
        except ValueError as e:
            orig_str = str(e) == "invalid number of data points ([4 0]) specified"
            new_str = str(e) == "Invalid number of FFT data points (0) specified."
            self.assertTrue(orig_str | new_str)

    def test_threshold_binary(self):
        """
        Test threshold_binary method
        :return:
        """
        print("threshold_binary")
        try:
            result = sp.threshold_binary(self.data_pair, STREAM_TO_STREAM, Axis.FULL, threshold=128)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_threshold_binary_max(self):
        """
        Test threshold_binary max method
        :return:
        """
        print("threshold_binary")
        try:
            sp.threshold_binary_max(self.data_pair, STREAM_TO_STREAM, Axis.FULL, maxvalue=255, ratio=1.0)
        except ValueError:
            pass

    def test_threshold_binary_float(self):
        """
        Test threshold_binary_float method
        :return:
        """
        print("threshold_binary_float")
        try:
            result = sp.threshold_binary_float(self.data_pair, STREAM_TO_STREAM, Axis.FULL, threshold=128, maxvalue = 255)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_threshold_to_zero(self):
        """
        Test threshold_to_zero method
        :return:
        """
        print("threshold_to_zero")
        try:
            result = sp.threshold_to_zero(self.data_pair, STREAM_TO_STREAM, Axis.FULL, threshold=128)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_threshold_to_zero_by_pixel_float(self):
        """
        Test threshold_to_zero_by_pixel_float method
        :return:
        """
        print("threshold_to_zero_by_pixel_float")
        try:
            result = sp.threshold_to_zero_by_pixel_float(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_threshold_to_zero_float(self):
        """
        Test threshold_to_zero_float method
        :return:
        """
        print("threshold_to_zero_float")
        try:
            result = sp.threshold_to_zero_float(self.data_pair, STREAM_TO_STREAM, Axis.FULL, threshold=0.5)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_threshold_binary_inverse(self):
        """
        Test threshold_binary_inverse method
        :return:
        """
        print("threshold_binary_inverse")
        try:
            result = sp.threshold_binary_inverse(self.data_pair, STREAM_TO_STREAM, Axis.FULL, threshold=128.0)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_threshold_binary_inverse_mask(self):
        """
        Test threshold_binary_inverse_mask method
        :return:
        """
        print("threshold_binary_inverse_mask")
        try:
            result = sp.threshold_binary_inverse_mask(self.data_pair, STREAM_TO_STREAM, Axis.FULL, threshold=128.0)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_morph_erosion_rect(self):
        """
        Test morph_erosion_rect method
        :return:
        """
        print("morph_erosion_rect")
        try:
            sp.morph_erosion_rect(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5, iterations=1)
        except cv2.error:
            pass

    def test_morph_erosion_ellipse(self):
        """
        Test morph_erosion_ellipse method
        :return:
        """
        print("morph_erosion_ellipse")
        try:
            sp.morph_erosion_ellipse(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5, iterations=1)
        except cv2.error:
            pass

    def test_morph_erosion_cross(self):
        """
        Test morph_erosion_cross method
        :return:
        """
        print("morph_erosion_cross")
        try:
            sp.morph_erosion_cross(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5, iterations=1)
        except cv2.error:
            pass

    def test_morph_dilate_rect(self):
        """
        Test morph_dilate_rect method
        :return:
        """
        print("morph_dilate_rect")
        try:
            sp.morph_dilate_rect(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5, iterations=1)
        except cv2.error:
            pass

    def test_morph_dilate_ellipse(self):
        """
        Test morph_dilate_ellipse method
        :return:
        """
        print("morph_dilate_ellipse")
        try:
            sp.morph_dilate_ellipse(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5, iterations=1)
        except cv2.error:
            pass

    def test_morph_dilate_cross(self):
        """
        Test morph_dilate_cross method
        :return:
        """
        print("morph_dilate_cross")
        try:
            sp.morph_dilate_cross(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5, iterations=1)
        except cv2.error:
            pass

    def test_morph_open_rect(self):
        """
        Test morph_open_rect method
        :return:
        """
        print("morph_open_rect")
        try:
            sp.morph_open_rect(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_open_ellipse(self):
        """
        Test morph_open_ellipse method
        :return:
        """
        print("morph_open_ellipse")
        try:
            sp.morph_open_ellipse(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_open_cross(self):
        """
        Test morph_open_cross method
        :return:
        """
        print("morph_open_cross")
        try:
            sp.morph_open_cross(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_close_rect(self):
        """
        Test morph_close_rect method
        :return:
        """
        print("morph_close_rect")
        try:
            sp.morph_close_rect(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_close_ellipse(self):
        """
        Test morph_close_ellipse method
        :return:
        """
        print("morph_close_ellipse")
        try:
            sp.morph_close_ellipse(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_close_cross(self):
        """
        Test morph_close_cross method
        :return:
        """
        print("morph_close_cross")
        try:
            sp.morph_close_cross(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_gradient_rect(self):
        """
        Test morph_gradient_rect method
        :return:
        """
        print("morph_gradient_rect")
        try:
            sp.morph_gradient_rect(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_gradient_ellipse(self):
        """
        Test morph_gradient_ellipse method
        :return:
        """
        print("morph_gradient_ellipse")
        try:
            sp.morph_gradient_ellipse(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_gradient_cross(self):
        """
        Test morph_gradient_cross method
        :return:
        """
        print("morph_gradient_cross")
        try:
            sp.morph_gradient_cross(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_tophat_rect(self):
        """
        Test morph_tophat_rect method
        :return:
        """
        print("morph_tophat_rect")
        try:
            sp.morph_tophat_rect(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_tophat_ellipse(self):
        """
        Test morph_tophat_ellipse method
        :return:
        """
        print("morph_tophat_ellipse")
        try:
            sp.morph_tophat_ellipse(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_tophat_cross(self):
        """
        Test morph_tophat_cross method
        :return:
        """
        print("morph_tophat_cross")
        try:
            sp.morph_tophat_cross(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_blackhat_rect(self):
        """
        Test morph_blackhat_rect method
        :return:
        """
        print("morph_blackhat_rect")
        try:
            sp.morph_blackhat_rect(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_blackhat_ellipse(self):
        """
        Test morph_blackhat_ellipse method
        :return:
        """
        print("morph_blackhat_ellipse")
        try:
            sp.morph_blackhat_ellipse(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_morph_blackhat_cross(self):
        """
        Test morph_blackhat_cross method
        :return:
        """
        print("morph_blackhat_cross")
        try:
            sp.morph_blackhat_cross(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        except cv2.error:
            pass

    def test_scalar_add(self):
        """
        Test scalar_add method
        :return:
        """
        print("scalar_add")
        try:
            sp.scalar_add(self.data_pair, STREAM_TO_STREAM, Axis.FULL, 1.0)
        except Exception as e:
            self.assertEqual(str(e), "OpenCV method returned None (Invalid Data)")

    def test_scalar_subtract(self):
        """
        Test scalar_subtract method
        :return:
        """
        print("scalar_subtract")
        try:
            sp.scalar_subtract(self.data_pair, STREAM_TO_STREAM, Axis.FULL, 2.0)
        except Exception as e:
            self.assertEqual(str(e), "OpenCV method returned None (Invalid Data)")

    def test_scalar_multiply(self):
        """
        Test scalar_multiply method
        :return:
        """
        print("scalar_multiply")
        try:
            sp.scalar_multiply(self.data_pair, STREAM_TO_STREAM, Axis.FULL, 1.0)
        except Exception as e:
            self.assertEqual(str(e), "OpenCV method returned None (Invalid Data)")

    def test_scalar_divide(self):
        """
        Test scalar_divide method
        :return:
        """
        print("scalar_divide")
        try:
            sp.scalar_divide(self.data_pair, STREAM_TO_STREAM, Axis.FULL, 1.0)
        except Exception as e:
            self.assertEqual(str(e), "OpenCV method returned None (Invalid Data)")

    def test_bitwise_and(self):
        """
        Test bitwise_and method
        :return:
        """
        print("bitwise_and")
        try:
            result = sp.bitwise_and(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_bitwise_not(self):
        """
        Test bitwise_not method
        :return:
        """
        print("bitwise_not")
        try:
            result = sp.bitwise_not(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_bitwise_or(self):
        """
        Test bitwise_or method
        :return:
        """
        print("bitwise_or")
        try:
            result = sp.bitwise_or(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_bitwise_xor(self):
        """
        Test bitwise_xor method
        :return:
        """
        print("bitwise_xor")
        try:
            result = sp.bitwise_xor(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_cv2_absdiff(self):
        """
        Test cv2_absdiff method
        :return:
        """
        print("cv2_absdiff")
        try:
            result = sp.cv2_absdiff(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_absdiff(self):
        """
        Test absdiff method
        :return:
        """
        print("absdiff")
        try:
            result = sp.absdiff(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_cv2_add(self):
        """
        Test cv2_add method
        :return:
        """
        print("cv2_add")
        try:
            result = sp.cv2_add(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_cv2_add_weighted(self):
        """
        Test cv2_add_weighted method
        :return:
        """
        print("cv2_add_weighted")
        try:
            result = sp.cv2_add_weighted(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL, alpha=1, beta=0)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_cv2_subtract(self):
        """
        Test cv2_subtract method
        :return:
        """
        print("cv2_subtract")
        try:
            result = sp.cv2_subtract(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_subtract_saturate(self):
        """
        Test subtract_saturate method
        :return:
        """
        print("subtract_saturate")
        try:
            result = sp.subtract_saturate(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_cv2_multiply(self):
        """
        Test cv2_multiply method
        :return:
        """
        print("cv2_multiply")
        try:
            result = sp.cv2_multiply(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_multiply_transposed(self):
        """
        Test multiply_transposed method
        :return:
        """
        print("multiply_transposed")
        try:
            result = sp.multiply_transposed(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_random_uniform(self):
        """
        Test random_uniform method
        :return:
        """
        print("random_uniform")
        try:
            sp.random_uniform(self.data_pair, STREAM_TO_STREAM, Axis.FULL, low=0, high=255)
        except cv2.error:
            pass

    def test_random_normal(self):
        """
        Test random_normal method
        :return:
        """
        print("random_normal")
        try:
            sp.random_normal(self.data_pair, STREAM_TO_STREAM, Axis.FULL, normal_mean=128, std_dev=1)
        except cv2.error:
            pass

    def test_random_shuffle(self):
        """
        Test random_shuffle method
        :return:
        """
        print("random_shuffle")
        result = sp.random_shuffle(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_sqrt(self):
        """
        Test cv2_sqrt method
        :return:
        """
        print("cv2_sqrt")
        try:
            result = sp.cv2_sqrt(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_cv2_divide(self):
        """
        Test cv2_divide method
        :return:
        """
        print("cv2_divide")
        try:
            result = sp.cv2_divide(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_cv2_pow(self):
        """
        Test cv2_pow method
        :return:
        """
        print("cv2_pow")
        try:
            result = sp.cv2_pow(self.data_pair, STREAM_TO_STREAM, Axis.FULL, power=1)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_cv2_rms(self):
        """
        Test cv2_rms method
        :return:
        """
        print("cv2_rms")
        try:
            sp.cv2_rms(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_size=5)
        except cv2.error:
            pass

    def test_cv2_dct(self):
        """
        Test cv2_dct method
        :return:
        """
        print("cv2_dct")
        try:
            sp.cv2_dct(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
        except ValueError as e:
            self.assertEqual(str(e), "Data array cannot be empty.")

    def test_cv2_idct(self):
        """
        Test cv2_idct method
        :return:
        """
        print("cv2_idct")
        try:
            sp.cv2_idct(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
        except ValueError as e:
            self.assertEqual(str(e), "Data array cannot be empty.")

    def test_cv2_dft_real(self):
        """
        Test cv2_dft_real method
        :return:
        """
        print("cv2_dft_real")
        try:
            sp.cv2_dft_real(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
        except ValueError as e:
            self.assertEqual(str(e), "Data array cannot be empty.")

    def test_cv2_idft(self):
        """
        Test cv2_idft method
        :return:
        """
        print("cv2_idft")
        try:
            sp.cv2_idft(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
        except ValueError as e:
            self.assertEqual(str(e), "Data array cannot be empty.")

    def test_cv2_transpose(self):
        """
        Test cv2_transpose method
        :return:
        """
        print("cv2_transpose")
        try:
            result = sp.cv2_transpose(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_cv2_log(self):
        """
        Test cv2_log method
        :return:
        """
        print("cv2_log")
        try:
            result = sp.cv2_log(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_cv2_max(self):
        """
        Test cv2_max method
        :return:
        """
        print("cv2_max")
        try:
            result = sp.cv2_max(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_cv2_min(self):
        """
        Test cv2_min method
        :return:
        """
        print("cv2_min")
        try:
            result = sp.cv2_min(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_scalar_max(self):
        """
        Test scalar_max method
        :return:
        """
        print("scalar_max")
        try:
            result = sp.scalar_max(self.data_pair, STREAM_TO_STREAM, Axis.FULL, scalar=1)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_scalar_min(self):
        """
        Test scalar_min method
        :return:
        """
        print("scalar_min")
        try:
            result = sp.scalar_min(self.data_pair, STREAM_TO_STREAM, Axis.FULL, scalar=1)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_multiply_spectrum(self):
        """
        Test multiply_spectrum method
        :return:
        """
        print("multiply_spectrum")
        try:
            sp.multiply_spectrum(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        except ValueError as e:
            self.assertEqual(str(e), "Data array cannot be empty.")

    def test_in_range(self):
        """
        Test in_range method
        :return:
        """
        print("in_range")
        try:
            sp.in_range(self.data_pair, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=128)
        except cv2.error:
            pass

    def test_std_deviation(self):
        """
        Test std_deviation method
        :return:
        """
        print("std_deviation")
        try:
            sp.std_deviation(self.data_pair, STREAM_TO_STREAM, Axis.FULL, kernel_size=5)
        except ValueError as e:
            self.assertEqual(str(e), "Data array cannot be empty.")

    def test_threshold_nlargest(self):
        """
        Test threshold_nlargest method
        :return:
        """
        print("threshold_nlargest")
        try:
            sp.threshold_nlargest(self.data_pair, STREAM_TO_STREAM, Axis.FULL, n=5)
        except ValueError:
            pass

    def test_threshold_nlargest_binary(self):
        """
        Test threshold_nlargest_binary method
        :return:
        """
        print("threshold_nlargest_binary")
        try:
            sp.threshold_nlargest_binary(self.data_pair, STREAM_TO_STREAM, Axis.FULL, n=5)
        except ValueError:
            pass

    def test_scale_abs(self):
        """
        Test scale_abs method
        :return:
        """
        print("scale_abs")
        try:
            result = sp.scale_abs(self.data_pair, STREAM_TO_STREAM, Axis.FULL, alpha=1, beta=0)
            self.assertIsInstance(result, data.EmadeDataPair)
        except ValueError as e:
            self.assertEqual(str(e), 'Primitive produced 0-d array')

    def test_contours_all(self):
        """
        Test contours_all method
        :return:
        """
        print("contours_all")
        try:
            sp.contours_all(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
        except cv2.error:
            pass

    def test_contours_min_area(self):
        """
        Test contours_min_area method
        :return:
        """
        print("contours_min_area")
        try:
            sp.contours_min_area(self.data_pair, STREAM_TO_STREAM, Axis.FULL, area=10)
        except cv2.error:
            pass

    def test_contours_max_area(self):
        """
        Test contours_max_area method
        :return:
        """
        print("contours_max_area")
        try:
            sp.contours_max_area(self.data_pair, STREAM_TO_STREAM, Axis.FULL, area=10)
        except cv2.error:
            pass

    def test_contours_convex_concave(self):
        """
        Test contours_convex_concave method
        :return:
        """
        print("contours_convex_concave")
        try:
            sp.contours_convex_concave(self.data_pair, STREAM_TO_STREAM, Axis.FULL, convex=True)
        except cv2.error:
            pass

    def test_contours_min_length(self):
        """
        Test contours_min_length method
        :return:
        """
        print("contours_min_length")
        try:
            sp.contours_min_length(self.data_pair, STREAM_TO_STREAM, Axis.FULL, length=10)
        except cv2.error:
            pass

    def test_contours_max_length(self):
        """
        Test contours_max_length method
        :return:
        """
        print("contours_max_length")
        try:
            sp.contours_max_length(self.data_pair, STREAM_TO_STREAM, Axis.FULL, length=10)
        except cv2.error:
            pass

    def test_contour_mask(self):
        """
        Test contour_mask method
        :return:
        """
        print("contour_mask")
        try:
            sp.contour_mask(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
        except cv2.error:
            pass

    def test_contour_mask_min_area(self):
        """
        Test contour_mask_min_area method
        :return:
        """
        print("contour_mask_min_area")
        try:
            sp.contour_mask_min_area(self.data_pair, STREAM_TO_STREAM, Axis.FULL, area=10)
        except cv2.error:
            pass

    def test_contour_mask_max_area(self):
        """
        Test contour_mask_max_area method
        :return:
        """
        print("contour_mask_max_area")
        try:
            sp.contour_mask_max_area(self.data_pair, STREAM_TO_STREAM, Axis.FULL, area=10)
        except cv2.error:
            pass

    def test_contour_mask_convex(self):
        """
        Test contour_mask_convex method
        :return:
        """
        print("contour_mask_convex")
        try:
            sp.contour_mask_convex(self.data_pair, STREAM_TO_STREAM, Axis.FULL, convex=True)
        except cv2.error:
            pass

    def test_contour_mask_min_length(self):
        """
        Test contour_mask_min_length method
        :return:
        """
        print("contour_mask_min_length")
        try:
            sp.contour_mask_min_length(self.data_pair, STREAM_TO_STREAM, Axis.FULL, length=10)
        except cv2.error:
            pass

    def test_contour_mask_max_length(self):
        """
        Test contour_mask_max_length method
        :return:
        """
        print("contour_mask_max_length")
        try:
            sp.contour_mask_max_length(self.data_pair, STREAM_TO_STREAM, Axis.FULL, length=10)
        except cv2.error:
            pass

    def test_contour_mask_range_length(self):
        """
        Test contour_mask_range_length method
        :return:
        """
        print("contour_mask_range_length")
        try:
            sp.contour_mask_range_length(self.data_pair, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=128)
        except cv2.error:
            pass

    def test_contour_mask_min_enclosing_circle(self):
        """
        Test contour_mask_min_enclosing_circle method
        :return:
        """
        print("contour_mask_min_enclosing_circle")
        try:
            sp.contour_mask_min_enclosing_circle(self.data_pair, STREAM_TO_STREAM, Axis.FULL, area=10)
        except cv2.error:
            pass

    def test_contour_mask_max_enclosing_circle(self):
        """
        Test contour_mask_max_enclosing_circle method
        :return:
        """
        print("contour_mask_max_enclosing_circle")
        try:
            sp.contour_mask_max_enclosing_circle(self.data_pair, STREAM_TO_STREAM, Axis.FULL, area=10)
        except cv2.error:
            pass

    def test_contour_mask_range_enclosing_circle(self):
        """
        Test contour_mask_range_enclosing_circle method
        :return:
        """
        print("contour_mask_range_enclosing_circle")
        try:
            sp.contour_mask_range_enclosing_circle(self.data_pair, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=128)
        except cv2.error:
            pass

    def test_contour_mask_min_extent_enclosing_circle(self):
        """
        Test contour_mask_min_extent_enclosing_circle method
        :return:
        """
        print("contour_mask_min_extent_enclosing_circle")
        try:
            sp.contour_mask_min_extent_enclosing_circle(self.data_pair, STREAM_TO_STREAM, Axis.FULL, ratio=0.5)
        except cv2.error:
            pass

    def test_contour_mask_max_extent_enclosing_circle(self):
        """
        Test contour_mask_max_extent_enclosing_circle method
        :return:
        """
        print("contour_mask_max_extent_enclosing_circle")
        try:
            sp.contour_mask_max_extent_enclosing_circle(self.data_pair, STREAM_TO_STREAM, Axis.FULL, ratio=0.5)
        except cv2.error:
            pass

    def test_contour_mask_range_extent_enclosing_circle(self):
        """
        Test contour_mask_range_extent_enclosing_circle method
        :return:
        """
        print("contour_mask_range_extent_enclosing_circle")
        try:
            sp.contour_mask_range_extent_enclosing_circle(self.data_pair, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=128)
        except cv2.error:
            pass

    def test_contour_mask_min_aspect_ratio(self):
        """
        Test contour_mask_min_aspect_ratio method
        :return:
        """
        print("contour_mask_min_aspect_ratio")
        try:
            sp.contour_mask_min_aspect_ratio(self.data_pair, STREAM_TO_STREAM, Axis.FULL, ratio=0.5)
        except cv2.error:
            pass

    def test_contour_mask_max_aspect_ratio(self):
        """
        Test contour_mask_max_aspect_ratio method
        :return:
        """
        print("contour_mask_max_aspect_ratio")
        try:
            sp.contour_mask_max_aspect_ratio(self.data_pair, STREAM_TO_STREAM, Axis.FULL, ratio=0.5)
        except cv2.error:
            pass

    def test_contour_mask_range_aspect_ratio(self):
        """
        Test contour_mask_range_aspect_ratio method
        :return:
        """
        print("contour_mask_range_aspect_ratio")
        try:
            sp.contour_mask_range_aspect_ratio(self.data_pair, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=128)
        except cv2.error:
            pass

    def test_contour_mask_min_extent(self):
        """
        Test contour_mask_min_extent method
        :return:
        """
        print("contour_mask_min_extent")
        try:
            sp.contour_mask_min_extent(self.data_pair, STREAM_TO_STREAM, Axis.FULL, boundary=128)
        except cv2.error:
            pass

    def test_contour_mask_max_extent(self):
        """
        Test contour_mask_max_extent method
        :return:
        """
        print("contour_mask_max_extent")
        try:
            sp.contour_mask_max_extent(self.data_pair, STREAM_TO_STREAM, Axis.FULL, boundary=128)
        except cv2.error:
            pass

    def test_contour_mask_range_extent(self):
        """
        Test contour_mask_range_extent method
        :return:
        """
        print("contour_mask_range_extent")
        try:
            sp.contour_mask_range_extent(self.data_pair, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=128)
        except cv2.error:
            pass

    def test_contour_mask_min_solidity(self):
        """
        Test contour_mask_min_solidity method
        :return:
        """
        print("contour_mask_min_solidity")
        try:
            sp.contour_mask_min_solidity(self.data_pair, STREAM_TO_STREAM, Axis.FULL, boundary=128)
        except cv2.error:
            pass

    def test_contour_mask_max_solidity(self):
        """
        Test contour_mask_max_solidity method
        :return:
        """
        print("contour_mask_max_solidity")
        try:
            sp.contour_mask_max_solidity(self.data_pair, STREAM_TO_STREAM, Axis.FULL, boundary=0.5)
        except cv2.error:
            pass

    def test_contour_mask_range_solidity(self):
        """
        Test contour_mask_range_solidity method
        :return:
        """
        print("contour_mask_range_solidity")
        try:
            sp.contour_mask_range_solidity(self.data_pair, STREAM_TO_STREAM, Axis.FULL, lower_bound=0.5, upper_bound=1)
        except cv2.error:
            pass

    def test_contour_mask_min_equ_diameter(self):
        """
        Test contour_mask_min_equ_diameter method
        :return:
        """
        print("contour_mask_min_equ_diameter")
        try:
            sp.contour_mask_min_equ_diameter(self.data_pair, STREAM_TO_STREAM, Axis.FULL, boundary=10)
        except cv2.error:
            pass

    def test_contour_mask_max_equ_diameter(self):
        """
        Test contour_mask_max_equ_diameter method
        :return:
        """
        print("contour_mask_max_equ_diameter")
        try:
            sp.contour_mask_max_equ_diameter(self.data_pair, STREAM_TO_STREAM, Axis.FULL, boundary=10)
        except cv2.error:
            pass

    def test_contour_mask_range_equ_diameter(self):
        """
        Test contour_mask_range_equ_diameter method
        :return:
        """
        print("contour_mask_range_equ_diameter")
        try:
            sp.contour_mask_range_equ_diameter(self.data_pair, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=10)
        except cv2.error:
            pass

    def test_correlation(self):
        """
        Test correlation method
        :return:
        """
        print("correlation")
        result = sp.correlation(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_equal(self):
        """
        Test cv2 equal
        :return:
        """
        print("testing cv2_equal")
        result = sp.cv2_equal(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_greater_than(self):
        """
        Test cv2 greater than
        :return:
        """
        print("testing cv2_greater_than")
        result = sp.cv2_greater_than(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_greater_than_equal(self):
        """
        Test cv2 greater than equal
        :return:
        """
        print("testing cv2_greater_than_or_equal")
        result = sp.cv2_greater_than_or_equal(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_less_than(self):
        """
        Test cv2 less than
        :return:
        """
        print("testing cv2_less_than")
        result = sp.cv2_less_than(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_less_than_equal(self):
        """
        Test cv2 less than equal
        :return:
        """
        print("testing cv2_less_than_equal")
        result = sp.cv2_less_than_or_equal(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_not_equal(self):
        """
        Test cv2 not equal
        :return:
        """
        print("testing cv2_not_equal")
        result = sp.cv2_not_equal(self.data_pair, self.data_pair, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_gradient_magnitude(self):
        """
        Test gradient magnitude
        :return:
        """
        print("testing gradient_magnitude")
        try:
            sp.gradient_magnitude(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
        except ValueError as e:
            self.assertEqual(str(e), "Data array cannot be empty.")

    def test_otsu_binary_threshold(self):
        """
        Test Otsu's binarization of an image
        :return:
        """
        print("testing otsu_binary_threshold")
        try:
            sp.otsu_binary_threshold(self.data_pair, STREAM_TO_STREAM, Axis.FULL)
        except cv2.error:
            pass

if __name__ == '__main__':
    unittest.main()
