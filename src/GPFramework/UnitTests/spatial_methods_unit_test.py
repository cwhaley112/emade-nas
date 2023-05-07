"""
Programmed by Jason Zutty
Modified by Austin Dunn
Contains the unit tests for the spatial methods
"""
from GPFramework.data import EmadeDataInstance, EmadeDataPair, EmadeData, EmadeDataInstance, FeatureData, StreamData
import GPFramework.data as data
import os
import unittest
import numpy as np
np.random.seed(117)
#import matplotlib.pyplot as plt
import GPFramework.spatial_methods as sp
import GPFramework.operator_methods as op
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

class SpatialUnitTest(unittest.TestCase):  #pylint: disable=R0904
    """
    This class contains the unit tests for the new EMADE data classes
    """
    @classmethod
    def setUpClass(cls):
        # cls.image_data will hold the image
        # cls.img = cv2.imread('./data/combined_red_035.png')
        cls.feature_data_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/train_data_v2_suit_1-5.csv.gz')
        cls.many_to_some_data_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/test_many_to_some.csv.gz')
        cls.img_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/cokecangray.jpg')
        cls.color_img = cv2.imread(cls.img_path)
        cls.img = cv2.cvtColor(cls.color_img, cv2.COLOR_BGR2GRAY)

        stream = StreamData(cls.img)
        instances = [EmadeDataInstance(stream=stream)]
        image_data = EmadeData(instances)
        cls.image_data = EmadeDataPair((cp.deepcopy(image_data), None), (cp.deepcopy(image_data), None))
        cls.image_hex = hex(id(cls.image_data))

        stream = StreamData(cls.color_img)
        instances = [EmadeDataInstance(stream=stream)]
        image_data = EmadeData(instances)
        cls.image_data_color = EmadeDataPair((cp.deepcopy(image_data), None), (cp.deepcopy(image_data), None))

        stream = StreamData(np.array([cls.img, cls.img]))
        instances = [EmadeDataInstance(stream=stream)]
        image_data = EmadeData(instances)
        cls.video_data = EmadeDataPair((cp.deepcopy(image_data), None), (cp.deepcopy(image_data), None))

        # Feature data from chemical companion
        feature_data = data.load_feature_data_from_file(cls.feature_data_path)
        feature_data[0].set_instances(np.random.choice(feature_data[0].get_instances(), size=100, replace=False))
        cls.feature_data = EmadeDataPair(cp.deepcopy(feature_data), cp.deepcopy(feature_data))
        print("Reduced feature_data:", feature_data[0].get_numpy().shape, end=" ", flush=True)

        many_to_some_data = data.load_many_to_some_from_file(cls.many_to_some_data_path)
        many_to_some_data[0].set_instances(np.random.choice(many_to_some_data[0].get_instances(), size=100, replace=False))
        cls.many_to_some = EmadeDataPair(cp.deepcopy(many_to_some_data), cp.deepcopy(many_to_some_data))
        print("Reduced many_to_some:", many_to_some_data[0].get_numpy().shape, end=" ", flush=True)

    def setUp(self):
        """
        Dereferences the self.<DataPair> object attributes from the cls.<DataPair> class attributes.
        """
        self.image_data = cp.deepcopy(self.image_data)
        self.image_data_color = cp.deepcopy(self.image_data_color)
        self.video_data = cp.deepcopy(self.video_data)
        self.feature_data = cp.deepcopy(self.feature_data)
        self.many_to_some = cp.deepcopy(self.many_to_some)

    def test_center_of_mass(self):
        """
        Test center of mass method
        :return:
        """

        result = sp.center_of_mass(self.many_to_some, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_threshold_then_center_of_mass(self):
        """
        Convert datapair to binary then use center of mass method
        :return:
        """

        thresholded = sp.threshold_binary(self.many_to_some, STREAM_TO_STREAM, Axis.FULL, 0.2, 255)
        result = sp.center_of_mass(thresholded, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_check_kernel_size(self):
        """
        Test check_kernel_size method
        :return:
        """
        print("check_kernel_size")
        result = sp.check_kernel_size(k=5)
        self.assertIsInstance(result, int)

    def test_minimum_to_zero(self):
        """
        Test minimum_to_zero method
        :return:
        """
        print("minimum_to_zero")
        result = sp.minimum_to_zero(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_to_uint8(self):
        """
        Test to_uint8 method
        :return:
        """
        print("to_uint8")
        result = sp.to_uint8(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_to_uint8_scale(self):
        """
        Test to_uint8_scale method
        :return:
        """
        print("to_uint8_scale")
        result = sp.to_uint8_scale(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_to_float(self):
        """
        Test to_float method
        :return:
        """
        print("to_float")
        result = sp.to_float(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_to_float_normalize(self):
        """
        Test to_float_normalize method
        :return:
        """
        print("to_float_normalize")
        result = sp.to_float_normalize(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_edge_detection_canny(self):
        """
        Test edge_detection_canny method
        :return:
        """
        print("edge_detection_canny")
        result = sp.edge_detection_canny(self.image_data, STREAM_TO_STREAM, Axis.FULL, t1=50, t2=150, apertureSize=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_corner_detection_harris(self):
        """
        Test corner_detection_harris method
        :return:
        """
        print("corner_detection_harris")
        result = sp.corner_detection_harris(self.image_data, STREAM_TO_STREAM, Axis.FULL, 10, 5, 0.04)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_corner_detection_min_eigen_val(self):
        """
        Test corner_detection_min_eigen_val method
        :return:
        """
        print("corner_detection_min_eigen_val")
        result = sp.corner_detection_min_eigen_val(self.image_data, STREAM_TO_STREAM, Axis.FULL, 10, 5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_highpass_fourier_ellipsoid(self):
        """
        Test highpass_fourier_ellipsoid method
        :return:
        """
        print("highpass_fourier_ellipsoid")
        result = sp.highpass_fourier_ellipsoid(self.image_data, STREAM_TO_STREAM, Axis.FULL, size=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_highpass_irst(self):
        """
        Test highpass_irst method
        :return:
        """
        print("highpass_irst")
        result = sp.highpass_irst(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_median_filter(self):
        """
        Test median_filter method
        :return:
        """
        print("median_filter")
        result = sp.median_filter(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_size=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_lowpass_fourier_shift(self):
        """
        Test lowpass_fourier_shift method
        :return:
        """
        print("lowpass_fourier_shift")
        result = sp.lowpass_fourier_shift(self.image_data, STREAM_TO_STREAM, Axis.FULL, shift=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_highpass_fourier_shift(self):
        """
        Test highpass_fourier_shift method
        :return:
        """
        print("highpass_fourier_shift")
        result = sp.highpass_fourier_shift(self.image_data, STREAM_TO_STREAM, Axis.FULL, shift=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_highpass_fourier_gaussian(self):
        """
        Test highpass_fourier_gaussian method
        :return:
        """
        print("highpass_fourier_gaussian")
        result = sp.highpass_fourier_gaussian(self.image_data, STREAM_TO_STREAM, Axis.FULL, sigma=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_highpass_fourier_uniform(self):
        """
        Test highpass_fourier_uniform method
        :return:
        """
        print("highpass_fourier_uniform")
        result = sp.highpass_fourier_uniform(self.image_data, STREAM_TO_STREAM, Axis.FULL, size=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_highpass_unsharp_mask(self):
        """
        Test highpass_unsharp_mask method
        :return:
        """
        print("highpass_unsharp_mask")
        result = sp.highpass_unsharp_mask(self.image_data, STREAM_TO_STREAM, Axis.FULL, 9, 10, 10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_highpass_laplacian(self):
        """
        Test highpass_laplacian method
        :return:
        """
        print("highpass_laplacian")
        result = sp.highpass_laplacian(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_size=3, scale=1, delta=0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_highpass_sobel_derivative(self):
        """
        Test highpass_sobel_derivative method
        :return:
        """
        print("highpass_sobel_derivative")
        result = sp.highpass_sobel_derivative(self.image_data, STREAM_TO_STREAM, Axis.FULL, dx=1, dy=1, ksize=5, scale=1, delta=0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_lowpass_filter_median(self):
        """
        Test lowpass_filter_median method
        :return:
        """
        print("lowpass_filter_median")
        result = sp.lowpass_filter_median(self.image_data, STREAM_TO_STREAM, Axis.FULL, filter_size=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_lowpass_filter_average(self):
        """
        Test lowpass_filter_average method
        :return:
        """
        print("lowpass_filter_average")
        result = sp.lowpass_filter_average(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_size=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_lowpass_filter_gaussian(self):
        """
        Test lowpass_filter_gaussian method
        :return:
        """
        print("lowpass_filter_gaussian")
        result = sp.lowpass_filter_gaussian(self.image_data, STREAM_TO_STREAM, Axis.FULL, 3, 3, 0.5, 0.5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_lowpass_filter_bilateral(self):
        """
        Test lowpass_filter_bilateral method
        :return:
        """
        print("lowpass_filter_bilateral")
        result = sp.lowpass_filter_bilateral(self.image_data, STREAM_TO_STREAM, Axis.FULL, filter_diameter=9, sigma_color=75, sigma_space=75)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_lowpass_fourier_ellipsoid(self):
        """
        Test lowpass_fourier_ellipsoid method
        :return:
        """
        print("lowpass_fourier_ellipsoid")
        result = sp.lowpass_fourier_ellipsoid(self.image_data, STREAM_TO_STREAM, Axis.FULL, size=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_lowpass_fourier_gaussian(self):
        """
        Test lowpass_fourier_gaussian method
        :return:
        """
        print("lowpass_fourier_gaussian")
        result = sp.lowpass_fourier_gaussian(self.image_data, STREAM_TO_STREAM, Axis.FULL, sigma=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_lowpass_fourier_uniform(self):
        """
        Test lowpass_fourier_uniform method
        :return:
        """
        print("lowpass_fourier_uniform")
        result = sp.lowpass_fourier_uniform(self.image_data, STREAM_TO_STREAM, Axis.FULL, size=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_threshold_binary(self):
        """
        Test threshold_binary method
        :return:
        """
        print("threshold_binary")
        result = sp.threshold_binary(self.image_data, STREAM_TO_STREAM, Axis.FULL, threshold=128)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_threshold_binary_max(self):
        """
        Test threshold_binary max method
        :return:
        """
        print("threshold_binary")
        result = sp.threshold_binary_max(self.image_data, STREAM_TO_STREAM, Axis.FULL, maxvalue=255, ratio=1.0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_threshold_binary_float(self):
        """
        Test threshold_binary_float method
        :return:
        """
        print("threshold_binary_float")
        result = sp.threshold_binary_float(self.image_data, STREAM_TO_STREAM, Axis.FULL, threshold=128, maxvalue = 255)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_threshold_to_zero(self):
        """
        Test threshold_to_zero method
        :return:
        """
        print("threshold_to_zero")
        result = sp.threshold_to_zero(self.image_data, STREAM_TO_STREAM, Axis.FULL, threshold=128)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_threshold_to_zero_by_pixel_float(self):
        """
        Test threshold_to_zero_by_pixel_float method
        :return:
        """
        print("threshold_to_zero_by_pixel_float")
        result = sp.threshold_to_zero_by_pixel_float(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_threshold_to_zero_float(self):
        """
        Test threshold_to_zero_float method
        :return:
        """
        print("threshold_to_zero_float")
        result = sp.threshold_to_zero_float(self.image_data, STREAM_TO_STREAM, Axis.FULL, threshold=0.5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_threshold_binary_inverse(self):
        """
        Test threshold_binary_inverse method
        :return:
        """
        print("threshold_binary_inverse")
        result = sp.threshold_binary_inverse(self.image_data, STREAM_TO_STREAM, Axis.FULL, threshold=128.0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_threshold_binary_inverse_mask(self):
        """
        Test threshold_binary_inverse_mask method
        :return:
        """
        print("threshold_binary_inverse_mask")
        result = sp.threshold_binary_inverse_mask(self.image_data, STREAM_TO_STREAM, Axis.FULL, threshold=128.0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_set_to_zero_if_greater_than_data_and_factor(self):
        """
        Test set_to_zero_if_greater_than_data_and_factor method
        :return:
        """
        print("set_to_zero_if_greater_than_data_and_factor")
        result = sp.set_to_zero_if_greater_than_data_and_factor(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL, alpha=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_set_to_zero_if_less_than_data_and_factor(self):
        """
        Test set_to_zero_if_less_than_data_and_factor method
        :return:
        """
        print("set_to_zero_if_less_than_data_and_factor")
        result = sp.set_to_zero_if_less_than_data_and_factor(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL, alpha=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_erosion_rect(self):
        """
        Test morph_erosion_rect method
        :return:
        """
        print("morph_erosion_rect")
        result = sp.morph_erosion_rect(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5, iterations=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_erosion_ellipse(self):
        """
        Test morph_erosion_ellipse method
        :return:
        """
        print("morph_erosion_ellipse")
        result = sp.morph_erosion_ellipse(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5, iterations=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_erosion_cross(self):
        """
        Test morph_erosion_cross method
        :return:
        """
        print("morph_erosion_cross")
        result = sp.morph_erosion_cross(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5, iterations=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_dilate_rect(self):
        """
        Test morph_dilate_rect method
        :return:
        """
        print("morph_dilate_rect")
        result = sp.morph_dilate_rect(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5, iterations=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_dilate_ellipse(self):
        """
        Test morph_dilate_ellipse method
        :return:
        """
        print("morph_dilate_ellipse")
        result = sp.morph_dilate_ellipse(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5, iterations=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_dilate_cross(self):
        """
        Test morph_dilate_cross method
        :return:
        """
        print("morph_dilate_cross")
        result = sp.morph_dilate_cross(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5, iterations=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_open_rect(self):
        """
        Test morph_open_rect method
        :return:
        """
        print("morph_open_rect")
        result = sp.morph_open_rect(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_open_ellipse(self):
        """
        Test morph_open_ellipse method
        :return:
        """
        print("morph_open_ellipse")
        result = sp.morph_open_ellipse(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_open_cross(self):
        """
        Test morph_open_cross method
        :return:
        """
        print("morph_open_cross")
        result = sp.morph_open_cross(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_close_rect(self):
        """
        Test morph_close_rect method
        :return:
        """
        print("morph_close_rect")
        result = sp.morph_close_rect(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_close_ellipse(self):
        """
        Test morph_close_ellipse method
        :return:
        """
        print("morph_close_ellipse")
        result = sp.morph_close_ellipse(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_close_cross(self):
        """
        Test morph_close_cross method
        :return:
        """
        print("morph_close_cross")
        result = sp.morph_close_cross(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_gradient_rect(self):
        """
        Test morph_gradient_rect method
        :return:
        """
        print("morph_gradient_rect")
        result = sp.morph_gradient_rect(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_gradient_ellipse(self):
        """
        Test morph_gradient_ellipse method
        :return:
        """
        print("morph_gradient_ellipse")
        result = sp.morph_gradient_ellipse(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_gradient_cross(self):
        """
        Test morph_gradient_cross method
        :return:
        """
        print("morph_gradient_cross")
        result = sp.morph_gradient_cross(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_tophat_rect(self):
        """
        Test morph_tophat_rect method
        :return:
        """
        print("morph_tophat_rect")
        result = sp.morph_tophat_rect(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_tophat_ellipse(self):
        """
        Test morph_tophat_ellipse method
        :return:
        """
        print("morph_tophat_ellipse")
        result = sp.morph_tophat_ellipse(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_tophat_cross(self):
        """
        Test morph_tophat_cross method
        :return:
        """
        print("morph_tophat_cross")
        result = sp.morph_tophat_cross(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_blackhat_rect(self):
        """
        Test morph_blackhat_rect method
        :return:
        """
        print("morph_blackhat_rect")
        result = sp.morph_blackhat_rect(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_blackhat_ellipse(self):
        """
        Test morph_blackhat_ellipse method
        :return:
        """
        print("morph_blackhat_ellipse")
        result = sp.morph_blackhat_ellipse(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_morph_blackhat_cross(self):
        """
        Test morph_blackhat_cross method
        :return:
        """
        print("morph_blackhat_cross")
        result = sp.morph_blackhat_cross(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_x=5, kernel_y=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_scalar_add(self):
        """
        Test scalar_add method
        :return:
        """
        print("scalar_add")
        result = sp.scalar_add(self.image_data, STREAM_TO_STREAM, Axis.FULL, 1.0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_scalar_subtract(self):
        """
        Test scalar_subtract method
        :return:
        """
        print("scalar_subtract")
        result = sp.scalar_subtract(self.image_data, STREAM_TO_STREAM, Axis.FULL, 2.0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_scalar_multiply(self):
        """
        Test scalar_multiply method
        :return:
        """
        print("scalar_multiply")
        result = sp.scalar_multiply(self.image_data, STREAM_TO_STREAM, Axis.FULL, 1.0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_scalar_divide(self):
        """
        Test scalar_divide method
        :return:
        """
        print("scalar_divide")
        result = sp.scalar_divide(self.image_data, STREAM_TO_STREAM, Axis.FULL, 1.0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_bitwise_and(self):
        """
        Test bitwise_and method
        :return:
        """
        print("bitwise_and")
        result = sp.bitwise_and(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_bitwise_not(self):
        """
        Test bitwise_not method
        :return:
        """
        print("bitwise_not")
        result = sp.bitwise_not(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_bitwise_or(self):
        """
        Test bitwise_or method
        :return:
        """
        print("bitwise_or")
        result = sp.bitwise_or(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_bitwise_xor(self):
        """
        Test bitwise_xor method
        :return:
        """
        print("bitwise_xor")
        result = sp.bitwise_xor(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_absdiff(self):
        """
        Test cv2_absdiff method
        :return:
        """
        print("cv2_absdiff")
        result = sp.cv2_absdiff(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_absdiff(self):
        """
        Test absdiff method
        :return:
        """
        print("absdiff")
        result = sp.absdiff(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_add(self):
        """
        Test cv2_add method
        :return:
        """
        print("cv2_add")
        result = sp.cv2_add(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_add_weighted(self):
        """
        Test cv2_add_weighted method
        :return:
        """
        print("cv2_add_weighted")
        result = sp.cv2_add_weighted(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL, alpha=1, beta=0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_subtract(self):
        """
        Test cv2_subtract method
        :return:
        """
        print("cv2_subtract")
        result = sp.cv2_subtract(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_subtract_saturate(self):
        """
        Test subtract_saturate method
        :return:
        """
        print("subtract_saturate")
        result = sp.subtract_saturate(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_multiply(self):
        """
        Test cv2_multiply method
        :return:
        """
        print("cv2_multiply")
        result = sp.cv2_multiply(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_multiply_transposed(self):
        """
        Test multiply_transposed method
        :return:
        """
        print("multiply_transposed")
        result = sp.multiply_transposed(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_random_uniform(self):
        """
        Test random_uniform method
        :return:
        """
        print("random_uniform")
        result = sp.random_uniform(self.image_data, STREAM_TO_STREAM, Axis.FULL, low=0, high=255)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_random_normal(self):
        """
        Test random_normal method
        :return:
        """
        print("random_normal")
        result = sp.random_normal(self.image_data, STREAM_TO_STREAM, Axis.FULL, normal_mean=128, std_dev=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_random_shuffle(self):
        """
        Test random_shuffle method
        :return:
        """
        print("random_shuffle")
        result = sp.random_shuffle(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_sqrt(self):
        """
        Test cv2_sqrt method
        :return:
        """
        print("cv2_sqrt")
        img = sp.to_float(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        result = sp.cv2_sqrt(img, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_divide(self):
        """
        Test cv2_divide method
        :return:
        """
        print("cv2_divide")
        result = sp.cv2_divide(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_np_divide(self):
        """
        Test np_divide method
        :return:
        """
        print("np_divide")
        result = sp.np_divide(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_pow(self):
        """
        Test cv2_pow method
        :return:
        """
        print("cv2_pow")
        result = sp.cv2_pow(self.image_data, STREAM_TO_STREAM, Axis.FULL, power=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_rms(self):
        """
        Test cv2_rms method
        :return:
        """
        print("cv2_rms")
        img = sp.to_float(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        result = sp.cv2_rms(img, STREAM_TO_STREAM, Axis.FULL, kernel_size=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_dct(self):
        """
        Test cv2_dct method
        :return:
        """
        print("cv2_dct")
        result = sp.cv2_dct(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_idct(self):
        """
        Test cv2_idct method
        :return:
        """
        print("cv2_idct")
        img = sp.to_float(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        result = sp.cv2_idct(img, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_dft_real(self):
        """
        Test cv2_dft_real method
        :return:
        """
        print("cv2_dft_real")
        img = sp.to_float(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        result = sp.cv2_dft_real(img, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_idft(self):
        """
        Test cv2_idft method
        :return:
        """
        print("cv2_idft")
        img = sp.to_float(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        result = sp.cv2_idft(img, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_transpose(self):
        """
        Test cv2_transpose method
        :return:
        """
        print("cv2_transpose")
        result = sp.cv2_transpose(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_log(self):
        """
        Test cv2_log method
        :return:
        """
        print("cv2_log")
        img = sp.to_float(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        result = sp.cv2_log(img, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_max(self):
        """
        Test cv2_max method
        :return:
        """
        print("cv2_max")
        result = sp.cv2_max(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_min(self):
        """
        Test cv2_min method
        :return:
        """
        print("cv2_min")
        result = sp.cv2_min(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_scalar_max(self):
        """
        Test scalar_max method
        :return:
        """
        print("scalar_max")
        result = sp.scalar_max(self.image_data, STREAM_TO_STREAM, Axis.FULL, scalar=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_scalar_min(self):
        """
        Test scalar_min method
        :return:
        """
        print("scalar_min")
        result = sp.scalar_min(self.image_data, STREAM_TO_STREAM, Axis.FULL, scalar=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_multiply_spectrum(self):
        """
        Test multiply_spectrum method
        :return:
        """
        print("multiply_spectrum")
        img = sp.to_float(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        result = sp.multiply_spectrum(img, img, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_in_range(self):
        """
        Test in_range method
        :return:
        """
        print("in_range")
        result = sp.in_range(self.image_data, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=128)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_std_deviation(self):
        """
        Test std_deviation method
        :return:
        """
        print("std_deviation")
        img = sp.to_float(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        result = sp.std_deviation(img, STREAM_TO_STREAM, Axis.FULL, kernel_size=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_threshold_nlargest(self):
        """
        Test threshold_nlargest method
        :return:
        """
        print("threshold_nlargest")
        result = sp.threshold_nlargest(self.image_data, STREAM_TO_STREAM, Axis.FULL, n=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_threshold_nlargest_binary(self):
        """
        Test threshold_nlargest_binary method
        :return:
        """
        print("threshold_nlargest_binary")
        result = sp.threshold_nlargest_binary(self.image_data, STREAM_TO_STREAM, Axis.FULL, n=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_nlargest(self):
        """
        Test nlargest method
        :return:
        """
        print("nlargest")
        result = sp.nlargest(self.image_data, n=5)
        self.assertIsInstance(result, list)

    def test_scale_abs(self):
        """
        Test scale_abs method
        :return:
        """
        print("scale_abs")
        result = sp.scale_abs(self.image_data, STREAM_TO_STREAM, Axis.FULL, alpha=1, beta=0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_get_data_items(self):
        """
        Test get_data_items method
        :return:
        """
        print("get_data_items")
        result = sp.get_data_items(self.image_data)
        self.assertIsInstance(result, list)

    def test_get_circular_extent(self):
        contours = sp.get_contours(self.image_data)
        result = sp.get_circular_extent(self.img,contours[0][0])
        self.assertIsInstance(result, float)

    def test_aspect_ratio(self):
        img = sp.edge_detection_canny(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        contours = sp.get_contours(img)
        result = sp.aspect_ratio(contours[0][0])
        self.assertIsInstance(result, float)

    def test_extent(self):
        contours = sp.get_contours(self.image_data)
        result = sp.extent(contours[0][0])
        self.assertIsInstance(result, float)

    def test_solidity(self):
        contours = sp.get_contours(self.image_data)
        result = sp.solidity(contours[0][0])
        self.assertIsInstance(result, float)

    def test_equ_diameter(self):
        contours = sp.get_contours(self.image_data)
        result = sp.equ_diameter(contours[0][0])
        self.assertIsInstance(result, float)

    def test_get_contours(self):
        """
        Test get_contours method
        :return:
        """
        print("get_contours")
        result = sp.get_contours(self.image_data)
        self.assertIsInstance(result, list)

    def test_contours_all(self):
        """
        Test contours_all method
        :return:
        """
        print("contours_all")
        result = sp.contours_all(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contours_min_area(self):
        """
        Test contours_min_area method
        :return:
        """
        print("contours_min_area")
        result = sp.contours_min_area(self.image_data, STREAM_TO_STREAM, Axis.FULL, area=10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contours_max_area(self):
        """
        Test contours_max_area method
        :return:
        """
        print("contours_max_area")
        result = sp.contours_max_area(self.image_data, STREAM_TO_STREAM, Axis.FULL, area=10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contours_convex_concave(self):
        """
        Test contours_convex_concave method
        :return:
        """
        print("contours_convex_concave")
        result = sp.contours_convex_concave(self.image_data, STREAM_TO_STREAM, Axis.FULL, convex=True)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contours_min_length(self):
        """
        Test contours_min_length method
        :return:
        """
        print("contours_min_length")
        result = sp.contours_min_length(self.image_data, STREAM_TO_STREAM, Axis.FULL, length=10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contours_max_length(self):
        """
        Test contours_max_length method
        :return:
        """
        print("contours_max_length")
        result = sp.contours_max_length(self.image_data, STREAM_TO_STREAM, Axis.FULL, length=10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask(self):
        """
        Test contour_mask method
        :return:
        """
        print("contour_mask")
        result = sp.contour_mask(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_min_area(self):
        """
        Test contour_mask_min_area method
        :return:
        """
        print("contour_mask_min_area")
        result = sp.contour_mask_min_area(self.image_data, STREAM_TO_STREAM, Axis.FULL, area=10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_max_area(self):
        """
        Test contour_mask_max_area method
        :return:
        """
        print("contour_mask_max_area")
        result = sp.contour_mask_max_area(self.image_data, STREAM_TO_STREAM, Axis.FULL, area=10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_convex(self):
        """
        Test contour_mask_convex method
        :return:
        """
        print("contour_mask_convex")
        result = sp.contour_mask_convex(self.image_data, STREAM_TO_STREAM, Axis.FULL, convex=True)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_min_length(self):
        """
        Test contour_mask_min_length method
        :return:
        """
        print("contour_mask_min_length")
        result = sp.contour_mask_min_length(self.image_data, STREAM_TO_STREAM, Axis.FULL, length=10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_max_length(self):
        """
        Test contour_mask_max_length method
        :return:
        """
        print("contour_mask_max_length")
        result = sp.contour_mask_max_length(self.image_data, STREAM_TO_STREAM, Axis.FULL, length=10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_range_length(self):
        """
        Test contour_mask_range_length method
        :return:
        """
        print("contour_mask_range_length")
        result = sp.contour_mask_range_length(self.image_data, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=128)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_min_enclosing_circle(self):
        """
        Test contour_mask_min_enclosing_circle method
        :return:
        """
        print("contour_mask_min_enclosing_circle")
        result = sp.contour_mask_min_enclosing_circle(self.image_data, STREAM_TO_STREAM, Axis.FULL, area=10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_max_enclosing_circle(self):
        """
        Test contour_mask_max_enclosing_circle method
        :return:
        """
        print("contour_mask_max_enclosing_circle")
        result = sp.contour_mask_max_enclosing_circle(self.image_data, STREAM_TO_STREAM, Axis.FULL, area=10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_range_enclosing_circle(self):
        """
        Test contour_mask_range_enclosing_circle method
        :return:
        """
        print("contour_mask_range_enclosing_circle")
        result = sp.contour_mask_range_enclosing_circle(self.image_data, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=128)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_min_extent_enclosing_circle(self):
        """
        Test contour_mask_min_extent_enclosing_circle method
        :return:
        """
        print("contour_mask_min_extent_enclosing_circle")
        result = sp.contour_mask_min_extent_enclosing_circle(self.image_data, STREAM_TO_STREAM, Axis.FULL, ratio=0.5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_max_extent_enclosing_circle(self):
        """
        Test contour_mask_max_extent_enclosing_circle method
        :return:
        """
        print("contour_mask_max_extent_enclosing_circle")
        result = sp.contour_mask_max_extent_enclosing_circle(self.image_data, STREAM_TO_STREAM, Axis.FULL, ratio=0.5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_range_extent_enclosing_circle(self):
        """
        Test contour_mask_range_extent_enclosing_circle method
        :return:
        """
        print("contour_mask_range_extent_enclosing_circle")
        result = sp.contour_mask_range_extent_enclosing_circle(self.image_data, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=128)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_min_aspect_ratio(self):
        """
        Test contour_mask_min_aspect_ratio method
        :return:
        """
        print("contour_mask_min_aspect_ratio")
        result = sp.contour_mask_min_aspect_ratio(self.image_data, STREAM_TO_STREAM, Axis.FULL, ratio=0.5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_max_aspect_ratio(self):
        """
        Test contour_mask_max_aspect_ratio method
        :return:
        """
        print("contour_mask_max_aspect_ratio")
        result = sp.contour_mask_max_aspect_ratio(self.image_data, STREAM_TO_STREAM, Axis.FULL, ratio=0.5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_range_aspect_ratio(self):
        """
        Test contour_mask_range_aspect_ratio method
        :return:
        """
        print("contour_mask_range_aspect_ratio")
        result = sp.contour_mask_range_aspect_ratio(self.image_data, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=128)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_min_extent(self):
        """
        Test contour_mask_min_extent method
        :return:
        """
        print("contour_mask_min_extent")
        result = sp.contour_mask_min_extent(self.image_data, STREAM_TO_STREAM, Axis.FULL, boundary=128)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_max_extent(self):
        """
        Test contour_mask_max_extent method
        :return:
        """
        print("contour_mask_max_extent")
        result = sp.contour_mask_max_extent(self.image_data, STREAM_TO_STREAM, Axis.FULL, boundary=128)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_range_extent(self):
        """
        Test contour_mask_range_extent method
        :return:
        """
        print("contour_mask_range_extent")
        result = sp.contour_mask_range_extent(self.image_data, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=128)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_min_solidity(self):
        """
        Test contour_mask_min_solidity method
        :return:
        """
        print("contour_mask_min_solidity")
        result = sp.contour_mask_min_solidity(self.image_data, STREAM_TO_STREAM, Axis.FULL, boundary=128)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_max_solidity(self):
        """
        Test contour_mask_max_solidity method
        :return:
        """
        print("contour_mask_max_solidity")
        result = sp.contour_mask_max_solidity(self.image_data, STREAM_TO_STREAM, Axis.FULL, boundary=0.5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_range_solidity(self):
        """
        Test contour_mask_range_solidity method
        :return:
        """
        print("contour_mask_range_solidity")
        result = sp.contour_mask_range_solidity(self.image_data, STREAM_TO_STREAM, Axis.FULL, lower_bound=0.5, upper_bound=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_min_equ_diameter(self):
        """
        Test contour_mask_min_equ_diameter method
        :return:
        """
        print("contour_mask_min_equ_diameter")
        result = sp.contour_mask_min_equ_diameter(self.image_data, STREAM_TO_STREAM, Axis.FULL, boundary=10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_max_equ_diameter(self):
        """
        Test contour_mask_max_equ_diameter method
        :return:
        """
        print("contour_mask_max_equ_diameter")
        result = sp.contour_mask_max_equ_diameter(self.image_data, STREAM_TO_STREAM, Axis.FULL, boundary=10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_contour_mask_range_equ_diameter(self):
        """
        Test contour_mask_range_equ_diameter method
        :return:
        """
        print("contour_mask_range_equ_diameter")
        result = sp.contour_mask_range_equ_diameter(self.image_data, STREAM_TO_STREAM, Axis.FULL, lower_bound=0, upper_bound=10)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_mean(self):
        """
        Test mean method
        :return:
        """
        print("mean")
        result = sp.mean(self.image_data)
        self.assertIsInstance(result, list)

    def test_my_sum(self):
        """
        Test my_sum method
        :return:
        """
        print("my_sum")
        result = sp.my_sum(self.image_data)
        self.assertIsInstance(result, list)

    def test_correlation(self):
        """
        Test correlation method
        This takes about 5 minutes to run
        :return:
        """
        print("correlation")
        print('Testing Correlation. Run time is approximatly 5 minutes')
        result = sp.correlation(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)
        pass

    def test_scalar_pixel_sum(self):
        """
        Test scalar_pixel_sum method
        :return:
        """
        print("scalar_pixel_sum")
        result = sp.scalar_pixel_sum(self.image_data)
        self.assertIsInstance(result, list)

    def test_median_filter_hole(self):
        """
        Test median_filter_hole method
        :return:
        """
        print("median_filter_hole")
        result = sp.median_filter_hole(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_size=7, hole_size=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_std_deviation_hole(self):
        """
        Test std_deviation_hole method
        :return:
        """
        print("std_deviation_hole")
        img = sp.to_float(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        result = sp.std_deviation_hole(img, STREAM_TO_STREAM, Axis.FULL, kernel_size=7, hole_size=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_prerejection(self):
        """
        Test prerejection method
        :return:
        """
        print("prerejection")
        result = sp.prerejection(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_size=7, hole_size=3, alpha=1)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_rms_hole(self):
        """
        Test rms_hole method
        :return:
        """
        print("rms_hole")
        result = sp.rms_hole(self.image_data, STREAM_TO_STREAM, Axis.FULL, kernel_size=7, hole_size=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_mark_targets(self):
        """
        Test mark_targets method
        :return:
        """
        print("mark_targets")
        result = sp.mark_targets(self.image_data, kernel_size=7)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_std_deviation_hole_custom(self):
        """
        Test std_deviation_hole method
        :return:
        """
        print("std_deviation_hole_custom")
        img1 = sp.to_float(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        img2 = sp.to_float(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        result = sp.std_deviation_hole_custom(img1, img2, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL, kernel_size=7, hole_size=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_spectral_filter(self):
        """
        Test spectral_filter method
        :return:
        """
        print("spectral_filter")
        img = sp.to_float(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        result = sp.spectral_filter(img, img, img, img, STREAM_TO_STREAM, STREAM_TO_STREAM, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_equal(self):
        """
        Test cv2 equal
        :return:
        """
        print("testing cv2_equal")
        result = sp.cv2_equal(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

        detection_data = data.load_many_to_some_from_file(self.many_to_some_data_path)

        self.many_to_some = EmadeDataPair(cp.deepcopy(detection_data), cp.deepcopy(detection_data))
        added = op.my_add(self.many_to_some, STREAM_TO_STREAM, Axis.FULL, 100)
        result = sp.cv2_equal(added, self.many_to_some, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_greater_than(self):
        """
        Test cv2 greater than
        :return:
        """
        print("testing cv2_greater_than")
        result = sp.cv2_greater_than(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

        detection_data = data.load_many_to_some_from_file(self.many_to_some_data_path)

        self.many_to_some = EmadeDataPair(cp.deepcopy(detection_data), cp.deepcopy(detection_data))
        added = op.my_add(self.many_to_some, STREAM_TO_STREAM, Axis.FULL, 100)
        result = sp.cv2_greater_than(added, self.many_to_some, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_greater_than_equal(self):
        """
        Test cv2 greater than equal
        :return:
        """
        print("testing cv2_greater_than_or_equal")
        result = sp.cv2_greater_than_or_equal(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

        detection_data = data.load_many_to_some_from_file(self.many_to_some_data_path)

        self.many_to_some = EmadeDataPair(cp.deepcopy(detection_data), cp.deepcopy(detection_data))
        added = op.my_add(self.many_to_some, STREAM_TO_STREAM, Axis.FULL, 100)
        result = sp.cv2_greater_than_or_equal(added, self.many_to_some, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_less_than(self):
        """
        Test cv2 less than
        :return:
        """
        print("testing cv2_less_than")
        result = sp.cv2_less_than(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

        detection_data = data.load_many_to_some_from_file(self.many_to_some_data_path)

        self.many_to_some = EmadeDataPair(cp.deepcopy(detection_data), cp.deepcopy(detection_data))
        added = op.my_add(self.many_to_some, STREAM_TO_STREAM, Axis.FULL, 100)
        result = sp.cv2_less_than(added, self.many_to_some, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_less_than_equal(self):
        """
        Test cv2 less than equal
        :return:
        """
        print("testing cv2_less_than_equal")
        result = sp.cv2_less_than_or_equal(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

        detection_data = data.load_many_to_some_from_file(self.many_to_some_data_path)

        self.many_to_some = EmadeDataPair(cp.deepcopy(detection_data), cp.deepcopy(detection_data))
        added = op.my_add(self.many_to_some, STREAM_TO_STREAM, Axis.FULL, 100)
        result = sp.cv2_less_than_or_equal(added, self.many_to_some, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_cv2_not_equal(self):
        """
        Test cv2 not equal
        :return:
        """
        print("testing cv2_not_equal")
        result = sp.cv2_not_equal(self.image_data, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

        added = op.my_add(self.many_to_some, STREAM_TO_STREAM, Axis.FULL, 100)
        result = sp.cv2_not_equal(added, self.many_to_some, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_background_subtraction(self):
        """
        Test sep background subtraction
        :return:
        """
        print("testing background_subtraction")
        result = sp.background_subtraction(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

        result = sp.background_subtraction(self.many_to_some, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_gradient_magnitude(self):
        """
        Test gradient magnitude
        :return:
        """
        print("testing gradient_magnitude")
        result = sp.gradient_magnitude(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

        result = sp.gradient_magnitude(self.many_to_some, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_snapshot(self):
        """
        Test snapshot
        :return:
        """
        print("testing my_snapshot")
        center = sp.center_of_mass(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        result = sp.my_snapshot(center, self.image_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL, precision=5)
        self.assertIsInstance(result, data.EmadeDataPair)

        center = sp.center_of_mass(self.many_to_some, STREAM_TO_STREAM, Axis.FULL)
        result = sp.my_snapshot(center, self.many_to_some, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL, precision=5)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_image_peak_finder(self):
        """
        Test Morphological reconstruction of an image
        :return:
        """
        print("testing image_peak_finder")
        result = sp.image_peak_finder(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

        result = sp.image_peak_finder(self.many_to_some, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_regional_maxima(self):
        """
        Test regional maxima of an image
        :return:
        """
        print("testing regional_maxima")
        result = sp.regional_maxima(self.image_data, STREAM_TO_STREAM, Axis.FULL, h=0.4)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_otsu_binary_threshold(self):
        """
        Test Otsu's binarization of an image
        :return:
        """
        print("testing otsu_binary_threshold")
        result = sp.otsu_binary_threshold(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_select_1d(self):
        """
        Test Select1D
        :return:
        """
        print("testing select_1d")
        result = sp.select_1d(self.image_data, STREAM_TO_STREAM, Axis.FULL, 0, 0)
        self.assertIsInstance(result, data.EmadeDataPair)

        result = sp.select_1d(self.many_to_some, STREAM_TO_STREAM, Axis.FULL, 0, 0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_select_2d(self):
        """
        Test Select2D
        :return:
        """
        print("testing select_2d")
        result = sp.select_2d(self.image_data, STREAM_TO_STREAM, Axis.FULL, 0, 1, 0, 0)
        self.assertIsInstance(result, data.EmadeDataPair)

        result = sp.select_2d(self.many_to_some, STREAM_TO_STREAM, Axis.FULL, 0, 1, 0, 0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_mean_filter(self):
        """
        Test MeanFilter
        :return:
        """
        print("testing mean_filter")
        result = sp.mean_filter(self.image_data, STREAM_TO_STREAM, Axis.FULL, dsize=3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_zero_one_norm(self):
        """
        Test ZeroOneNorm
        :return:
        """
        print("testing zero_one_norm")
        result = sp.zero_one_norm(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_standard_norm(self):
        """
        Test StandardNorm
        :return:
        """
        print("testing standard_norm")
        result = sp.standard_norm(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_equalize_disk(self):
        """
        Test EqualizeDisk
        :return:
        """
        print("testing equalize_disk")
        result = sp.equalize_disk(self.image_data, STREAM_TO_STREAM, Axis.FULL, dsize=30)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_equalize_adapthist(self):
        """
        Test EqualizeAdaptHist
        :return:
        """
        print("testing equalize_adapthist")
        result = sp.equalize_adapthist(self.image_data, STREAM_TO_STREAM, Axis.FULL, clip_limit=0.01)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_equalize_hist(self):
        """
        Test EqualizeHist
        :return:
        """
        print("testing equalize_hist")
        result = sp.equalize_hist(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_adjust_contrast_log(self):
        """
        Test AdjustContrastLog
        :return:
        """
        print("testing adjust_contrast_log")
        result = sp.adjust_contrast_log(self.image_data, STREAM_TO_STREAM, Axis.FULL, gain=1.0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_adjust_contrast_gamma(self):
        """
        Test AdjustContrastGamma
        :return:
        """
        print("testing adjust_contrast_gamma")
        result = sp.adjust_contrast_gamma(self.image_data, STREAM_TO_STREAM, Axis.FULL, gamma=2.0, gain=1.0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_local_pool_median(self):
        """
        Test LocalPoolingMedian
        :return:
        """
        print("testing local_pool_median")
        result = sp.local_pool_median(self.image_data, STREAM_TO_STREAM, Axis.FULL, 3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_local_pool_max(self):
        """
        Test LocalPoolingMax
        :return:
        """
        print("testing local_pool_max")
        result = sp.local_pool_max(self.image_data, STREAM_TO_STREAM, Axis.FULL, 3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_local_pool_mean(self):
        """
        Test LocalPoolingMean
        :return:
        """
        print("testing local_pool_mean")
        result = sp.local_pool_mean(self.image_data, STREAM_TO_STREAM, Axis.FULL, 3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_convert_to_counts(self):
        """
        Test ConvertToCounts
        :return:
        """
        print("testing convert_to_counts")
        result = sp.convert_to_counts(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_accumulate_weighted(self):
        """
        Test AccumulateWeighted
        :return:
        """
        print("testing convert_to_counts")
        result = sp.convert_to_counts(self.image_data, STREAM_TO_STREAM, Axis.FULL)
        result = sp.accumulate_weighted(result, STREAM_TO_STREAM, Axis.FULL, 0.01)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_image_alignment_ecc(self):
        """
        Test ImageAlignmentECC
        :return:
        """
        print("testing image_alignment_ecc")
        result = sp.image_alignment_ecc(self.video_data, self.video_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL, 50, 1e-10, 0, 3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_select_3d(self):
        """
        Test Select3d
        :return:
        """
        print("testing select_3d")
        result = sp.select_3d(self.image_data_color, STREAM_TO_STREAM, Axis.FULL, 0, 1, 2, 0, 0, 0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_convert_bw(self):
        """
        Test ConvertBW
        :return:
        """
        print("testing convert_bw")
        result = sp.convert_bw(self.image_data_color, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_gradient_weighted(self):
        """
        Test GradientWeighted
        :return:
        """
        print("testing gradient_weighted")
        result = sp.gradient_weighted(self.image_data, STREAM_TO_STREAM, Axis.FULL, 3, 0.7, 0.3)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_mean_with_hole(self):
        """
        Test MeanWithHole
        :return:
        """
        print("testing mean_with_hole")
        result = sp.mean_with_hole(self.image_data, STREAM_TO_STREAM, Axis.FULL, 7, 3)
        self.assertIsInstance(result, data.EmadeDataPair)

if __name__ == '__main__':
    unittest.main()
