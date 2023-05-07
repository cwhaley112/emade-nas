"""
Programmed by Jason Zutty
Modified by Austin Dunn
Contains the unit tests for the signal methods
"""
from GPFramework.data import EmadeDataInstance, EmadeDataPair
import GPFramework.data as data
import GPFramework.signal_methods as signal_methods
from GPFramework.constants import FEATURES_TO_FEATURES, STREAM_TO_STREAM, STREAM_TO_FEATURES, Axis
import os
import unittest
import numpy as np
np.random.seed(117)
import copy as cp


class SignalUnitTest(unittest.TestCase):  #pylint: disable=R0904
    """
    This class contains the unit tests for the new EMADE data classes
    """
    @classmethod
    def setUpClass(cls):
        # Time data from aglogica
        cls.time_data_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/test3axistrain.txt.gz')
        cls.time_data = data.load_many_to_one_from_file(cls.time_data_path)
        time_data_targets = np.stack([inst.get_target() for inst in cls.time_data[0].get_instances()])
        time_data_indices = np.array([]).astype(int)
        for target in np.unique(time_data_targets):
            indices = np.where(time_data_targets == target)[0]
            time_data_indices = np.hstack((time_data_indices, np.random.choice(indices, size=min(indices.shape[0], 10), replace=False))).astype(int)
        cls.time_data[0].set_instances(np.array(cls.time_data[0].get_instances())[time_data_indices])
        print("\nReduced time_data:", cls.time_data[0].get_numpy().shape, end=" ", flush=True)

        # Feature data from chemical companion
        cls.feature_data_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/train_data_v2_suit_1-5.csv.gz')
        cls.feature_data = data.load_feature_data_from_file(cls.feature_data_path)
        cls.feature_data[0].set_instances(np.random.choice(cls.feature_data[0].get_instances(), size=100, replace=False))
        print("Reduced feature_data:", cls.feature_data[0].get_numpy().shape, flush=True)

        # Lidar data for testing purposes
        cls.lidar_data = EmadeDataPair(cp.deepcopy(cls.time_data), cp.deepcopy(cls.time_data))

        # Time data from aglogica
        cls.time_data = EmadeDataPair(cp.deepcopy(cls.time_data), cp.deepcopy(cls.time_data))

        # Feature data from chemical companion
        cls.feature_data = EmadeDataPair(cp.deepcopy(cls.feature_data), cp.deepcopy(cls.feature_data))

    def setUp(self):
        """
        Dereferences the self.<DataPair> object attributes from the cls.<DataPair> class attributes.
        """
        self.lidar_data = cp.deepcopy(self.lidar_data)
        self.time_data = cp.deepcopy(self.time_data)
        self.feature_data = cp.deepcopy(self.feature_data)

    def test_select_range(self):
        """
        Test select range method
        """
        result = signal_methods.select_range(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 2, 4)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_my_concatenate(self):
        """
        Test both modes of concatenate
        """
        stream_to_stream = signal_methods.my_concatenate(self.time_data, self.time_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.my_concatenate(self.feature_data, self.feature_data, FEATURES_TO_FEATURES, FEATURES_TO_FEATURES, Axis.FULL, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_ecdf(self):
        """
        TODO: Fix Unit Test and/or method.
        Error:

        ======================================================================
        ERROR: test_ecdf (__main__.SignalUnitTest)
        ----------------------------------------------------------------------
        Traceback (most recent call last):
          File "signal_methods_unit_test.py", line 60, in test_ecdf
            stream_to_stream = signal_methods.my_ecdf(self.time_data, STREAM_TO_STREAM, Axis.FULL, n_components=2)
          File "/home/jrick6/repos/emade/src/GPFramework/wrapper_methods.py", line 673, in primitive_wrapper
            data = np.array(primitive_f(*data, *args, **helper_kwargs))
          File "/home/jrick6/repos/emade/src/GPFramework/signal_methods.py", line 1283, in my_ecdf_helper
            return ecdfRep(data, n_components)
          File "/home/jrick6/repos/emade/src/GPFramework/signal_methods.py", line 1494, in ecdfRep
            data = data.flatten(0)
        TypeError: order must be str, not int

        Test all three modes of the ecdf
        """
        stream_to_stream = signal_methods.my_ecdf(self.time_data, STREAM_TO_STREAM, Axis.FULL, n_components=2)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_features = signal_methods.my_ecdf(self.time_data, STREAM_TO_FEATURES, Axis.FULL, n_components=2)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)
        features_to_features = signal_methods.my_ecdf(stream_to_features, FEATURES_TO_FEATURES, Axis.FULL, n_components=2)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

        self.assertNotEquals(features_to_features, STREAM_TO_FEATURES, Axis.FULL)

    def test_fft(self):
        """
        Test all three modes of the fft
        """
        stream_to_stream = signal_methods.my_fft(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_features = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)
        features_to_features = signal_methods.my_fft(stream_to_features, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_dct(self):
        """
        Test all three modes of the dct
        """
        stream_to_stream = signal_methods.my_dct(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_features = signal_methods.my_dct(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)
        features_to_features = signal_methods.my_dct(stream_to_features, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_averager(self):
        """
        Test all two modes of my_averager
        """
        stream_to_stream = signal_methods.my_averager(self.time_data, STREAM_TO_STREAM, Axis.FULL, 3)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.my_averager(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 3)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_window_hann(self):
        """
        Test all two modes of window_hann
        """
        stream_to_stream = signal_methods.window_hann(self.time_data, STREAM_TO_STREAM, Axis.FULL, True)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.window_hann(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, True)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_window_hamming(self):
        """
        Test all two modes of window_hamming
        """
        stream_to_stream = signal_methods.window_hamming(self.time_data, STREAM_TO_STREAM, Axis.FULL, True)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.window_hamming(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, True)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_window_turkey(self):
        """
        Test all two modes of window_turkey
        """
        stream_to_stream = signal_methods.window_turkey(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.window_turkey(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_window_cosine(self):
        """
        Test all two modes of window_cosine
        """
        stream_to_stream = signal_methods.window_cosine(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.window_cosine(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_window_lanczos(self):
        """
        Test all two modes of window_lanczos
        """
        stream_to_stream = signal_methods.window_lanczos(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.window_lanczos(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_window_triangular(self):
        """
        Test all two modes of window_triangular
        """
        stream_to_stream = signal_methods.window_triangular(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.window_triangular(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_window_bartlett(self):
        """
        Test all two modes of window_bartlett
        """
        stream_to_stream = signal_methods.window_bartlett(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.window_bartlett(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_window_gaussian(self):
        """
        Test all two modes of window_gaussian
        """
        stream_to_stream = signal_methods.window_gaussian(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.window_gaussian(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_window_bartlett_hann(self):
        """
        Test all two modes of window_bartlett_hann
        """
        stream_to_stream = signal_methods.window_bartlett_hann(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.window_bartlett_hann(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_window_blackman(self):
        """
        Test all two modes of window_blackman
        """
        stream_to_stream = signal_methods.window_blackman(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.window_blackman(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def window_test(self, function):
        """
        Generic tests for windows
        """
        stream_to_stream = function(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = function(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_window_kaiser(self):
        """
        Test all two modes of window_kaiser
        """
        self.window_test(signal_methods.window_kaiser)

    def test_window_planck_taper(self):
        """
        Test all two modes of window_planck_taper
        """
        self.window_test(signal_methods.window_planck_taper)

    def test_window_nutall(self):
        """
        Test all two modes of window_nuttall
        """
        self.window_test(signal_methods.window_nuttall)

    def test_window_blackman_harris(self):
        """
        Test all two modes of window_blackman_harris
        """
        self.window_test(signal_methods.window_blackman_harris)

    def test_window_blackman_nuttall(self):
        """
        Test all two modes of window_blackman_nuttall
        """
        self.window_test(signal_methods.window_blackman_nuttall)

    def test_window_flat_top(self):
        """
        Test all two modes of window_flat_top
        """
        self.window_test(signal_methods.window_flat_top)

    def test_cascade_of_methods(self):
        """
        Test an fft for stream to features, a dct of features to features, and then a hanning window on the features
        """
        stream_to_features = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)
        features_to_features = signal_methods.my_dct(stream_to_features, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)
        # result = signal_methods.my_dwt(features_to_features, FEATURES_TO_FEATURES, Axis.FULL)
        # self.assertIsInstance(result, data.EmadeDataPair)
        result = signal_methods.window_hann(features_to_features, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_my_diff(self):
        """
        Test all two modes of my_diff
        """
        stream_to_stream = signal_methods.my_diff(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.my_diff(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_if_then_else(self):
        """
        Test if then else method
        """
        result = signal_methods.my_if_then_else(self.feature_data, self.time_data, self.time_data, FEATURES_TO_FEATURES, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL, Axis.FULL, column=0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_remove_feature(self):
        """
        Test remove feature method
        """
        result = signal_methods.remove_feature(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 0)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_my_auto_corr(self):
        """
        Test all two modes of my_auto_corr
        """
        stream_to_stream = signal_methods.my_auto_corr(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.my_auto_corr(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_cross_corr(self):
        """
        Test all two modes of my_auto_corr
        """
        stream_to_stream = signal_methods.my_cross_corr(self.time_data, self.time_data, STREAM_TO_STREAM, STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.my_cross_corr(self.feature_data, self.feature_data, FEATURES_TO_FEATURES, FEATURES_TO_FEATURES, Axis.FULL, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_cut_data_lead(self):
        """
        Test all two modes of cut_data_lead
        """
        stream_to_stream = signal_methods.cut_data_lead(self.time_data, STREAM_TO_STREAM, Axis.FULL, samples_to_cut=2)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        features_to_features = signal_methods.cut_data_lead(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, samples_to_cut=2)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_dwt(self):
        """
        Test all three modes of the dwt
        """
        stream_to_stream = signal_methods.my_dwt(self.time_data, STREAM_TO_STREAM, Axis.FULL, axis=0)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_features = signal_methods.my_dwt(self.time_data, STREAM_TO_FEATURES, Axis.FULL, axis=0)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)
        features_to_features = signal_methods.my_dwt(stream_to_features, FEATURES_TO_FEATURES, Axis.FULL, axis=0)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_norm(self):
        """
        Test all three modes of my_norm
        """
        stream_to_stream = signal_methods.my_norm(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_features = signal_methods.my_norm(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)
        features_to_features = signal_methods.my_norm(stream_to_features, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_rms_2d(self):
        """
        Test both modes of my_rms_2d
        """
        stream_to_stream = signal_methods.my_rms_2d(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_features = signal_methods.my_rms_2d(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)
        stream_to_stream = signal_methods.my_rms_2d(self.time_data, STREAM_TO_STREAM, Axis.FULL, axis=1)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_features = signal_methods.my_rms_2d(self.time_data, STREAM_TO_FEATURES, Axis.FULL, axis=1)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)

    def test_my_sum(self):
        """
        Test all three modes of my_sum
        """
        stream_to_stream = signal_methods.my_sum(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_features = signal_methods.my_sum(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)
        features_to_features = signal_methods.my_sum(stream_to_features, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_sub_sample_data(self):
        """
        Test only mode of sub_sample_data
        """
        stream_to_stream = signal_methods.sub_sample_data(self.time_data,
                                                        window_size=10,
                                                        step_size=3)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

    def test_my_prod(self):
        """
        Test all three modes of my_prod
        """
        stream_to_stream = signal_methods.my_prod(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_features = signal_methods.my_prod(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)
        features_to_features = signal_methods.my_prod(stream_to_features, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_cum_prod(self):
        """
        Test all three modes of my_cum_prod
        """
        stream_to_stream = signal_methods.my_cum_prod(self.feature_data, STREAM_TO_STREAM, Axis.FULL, -1)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_features = signal_methods.my_cum_prod(self.feature_data, STREAM_TO_FEATURES, Axis.FULL, -1)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)
        features_to_features = signal_methods.my_cum_prod(stream_to_features, FEATURES_TO_FEATURES, Axis.FULL, -1)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_cum_sum(self):
        """
        Test all three modes of my_cum_sum
        """
        stream_to_stream = signal_methods.my_cum_sum(self.time_data, STREAM_TO_STREAM, Axis.FULL, -1)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_features = signal_methods.my_cum_sum(self.time_data, STREAM_TO_FEATURES, Axis.FULL, -1)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)
        features_to_features = signal_methods.my_cum_sum(stream_to_features, FEATURES_TO_FEATURES, Axis.FULL, -1)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_abs(self):
        """
        Test all two modes of my_abs
        """
        stream_to_stream = signal_methods.my_abs(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

        features_to_features = signal_methods.my_abs(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_log(self):
        """
        Test all two modes of my_log
        """
        stream_to_stream = signal_methods.my_log(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

        features_to_features = signal_methods.my_log(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_round(self):
        """
        Test all two modes of my_round
        """
        stream_to_stream = signal_methods.my_round(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

        features_to_features = signal_methods.my_round(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_kalman_filter(self):
        """
        Test all two modes of my_kalman_filter
        """
        stream_to_stream = signal_methods.my_kalman_filter(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

        features_to_features = signal_methods.my_kalman_filter(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_linear_predictive_coding(self):
        """
        Test all three modes of the linear_predictive_coding
        """
        stream_to_stream = signal_methods.my_linear_predictive_coding(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_features = signal_methods.my_linear_predictive_coding(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)
        features_to_features = signal_methods.my_linear_predictive_coding(stream_to_features, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_wiener_filter(self):
        """
        Test all two modes of my_wiener_filter
        """
        stream_to_stream = signal_methods.my_wiener_filter(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

        features_to_features = signal_methods.my_wiener_filter(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_savitzky_golay_filter(self):
        """
        Test all two modes of my_savitzky_golay_filter
        """
        stream_to_stream = signal_methods.my_savitzky_golay_filter(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

        features_to_features = signal_methods.my_savitzky_golay_filter(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    """
    Issue with library hmmlearn.hmm
    """
    # def test_gaussian_hmm(self):
    #     """
    #     Test gaussian_hmm
    #     """
    #     output = signal_methods.gaussian_hmm(self.feature_data, n_components=4)
    #     self.assertIsInstance(output, data.EmadeDataPair)

    def test_my_rebase(self):
        """
        Test all two modes of my_rebase
        """
        stream_to_stream = signal_methods.my_rebase(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

        features_to_features = signal_methods.my_rebase(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_peak_finder(self):
        """
        Test all three modes of the peak_finder
        """
        stream_to_stream = signal_methods.my_peak_finder(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_features = signal_methods.my_peak_finder(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)
        features_to_features = signal_methods.my_peak_finder(stream_to_features, FEATURES_TO_FEATURES, Axis.FULL)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_informed_search(self):
        """
        Test all three modes of the informed_search
        """
        stream_to_stream_peaks = signal_methods.my_peak_finder(self.time_data,
                                                               STREAM_TO_STREAM, Axis.FULL,
                                                               start_delta=0.25)
        result = signal_methods.my_rebase(self.time_data, STREAM_TO_STREAM, Axis.FULL)
        stream_to_stream = signal_methods.my_informed_search(result,
                                                             stream_to_stream_peaks,
                                                             STREAM_TO_STREAM, STREAM_TO_STREAM,
                                                             Axis.FULL, Axis.FULL,
                                                             log_slope_thresh=0.0)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

        stream_to_features = signal_methods.my_informed_search(result,
                                                               stream_to_stream_peaks,
                                                               STREAM_TO_FEATURES, STREAM_TO_FEATURES,
                                                               Axis.FULL, Axis.FULL,
                                                               log_slope_thresh=0.0)
        self.assertIsInstance(stream_to_features, data.EmadeDataPair)

        features_to_features = signal_methods.my_peak_finder(stream_to_features,
                                                             FEATURES_TO_FEATURES, Axis.FULL,
                                                             start_delta=0.5)
        result = signal_methods.my_rebase(stream_to_features, FEATURES_TO_FEATURES, Axis.FULL)
        features_to_features = signal_methods.my_informed_search(result,
                                                                 features_to_features,
                                                                 FEATURES_TO_FEATURES, FEATURES_TO_FEATURES,
                                                                 Axis.FULL, Axis.FULL,
                                                                 log_slope_thresh=0.0)
        self.assertIsInstance(features_to_features, data.EmadeDataPair)

    def test_my_richardson_lucy(self):
        """
        Test my_richardson_lucy
        """
        stream_to_stream = signal_methods.my_richardson_lucy(self.lidar_data, STREAM_TO_STREAM, Axis.FULL, iterations=60)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

    def test_my_supersampling(self):
        """
        Test my_supersampling
        """
        stream_to_stream = signal_methods.my_supersampling(self.lidar_data, STREAM_TO_STREAM, Axis.FULL, 0.1, 0)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_stream = signal_methods.my_supersampling(self.lidar_data, STREAM_TO_STREAM, Axis.FULL, 0.1, 1)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_stream = signal_methods.my_supersampling(self.lidar_data, STREAM_TO_STREAM, Axis.FULL, 0.1, 2)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_stream = signal_methods.my_supersampling(self.lidar_data, STREAM_TO_STREAM, Axis.FULL, 0.1, 3)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

    def test_my_bspline(self):
        """
        Test my_bspline
        """
        stream_to_stream = signal_methods.my_bspline(self.lidar_data, STREAM_TO_STREAM, Axis.FULL, spline_degree=3)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

    def test_my_matched_filtering(self):
        """
        Test my_matched_filtering
        """
        stream_to_stream = signal_methods.my_matched_filtering(self.lidar_data, STREAM_TO_STREAM, Axis.FULL, 3)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_stream = signal_methods.my_matched_filtering(self.lidar_data, STREAM_TO_STREAM, Axis.FULL, 1)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)
        stream_to_stream = signal_methods.my_matched_filtering(self.lidar_data, STREAM_TO_STREAM, Axis.FULL, 2)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

    def test_gaussian_peak_lm(self):
        """
        TODO: Fix Unit Test and/or method.
        Error:

        ======================================================================
        ERROR: test_gaussian_peak_lm (__main__.SignalUnitTest)
        ----------------------------------------------------------------------
        Traceback (most recent call last):
          File "signal_methods_unit_test.py", line 563, in test_gaussian_peak_lm
            stream_to_stream = signal_methods.gaussian_peak_lm(self.lidar_data, STREAM_TO_STREAM, Axis.FULL)
          File "/home/jrick6/repos/emade/src/GPFramework/signal_methods.py", line 1716, in gaussian_peak_lm
            [peak_data_pair.get_train_data(), peak_data_pair.get_test_data()]):
        AttributeError: 'TriState' object has no attribute 'get_train_data'

        Test dual_gaussian_lm
        """
        stream_to_stream = signal_methods.gaussian_peak_lm(self.lidar_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

    def test_gaussian_peak_em(self):
        """
        TODO: Fix Unit Test and/or method.
        Error:

        ======================================================================
        ERROR: test_gaussian_peak_em (__main__.SignalUnitTest)
        ----------------------------------------------------------------------
        Traceback (most recent call last):
          File "signal_methods_unit_test.py", line 570, in test_gaussian_peak_em
            stream_to_stream = signal_methods.gaussian_peak_em(self.lidar_data, STREAM_TO_STREAM, Axis.FULL)
          File "/home/jrick6/repos/emade/src/GPFramework/wrapper_methods.py", line 673, in primitive_wrapper
            data = np.array(primitive_f(*data, *args, **helper_kwargs))
          File "/home/jrick6/repos/emade/src/GPFramework/signal_methods.py", line 1804, in gaussian_peak_em_helper
            data.append(np.random.multivariate_normal(mu[k + 1], Sigma[k + 1]))
        IndexError: index 2 is out of bounds for axis 0 with size 2

        Test dual_gaussian_em
        """
        stream_to_stream = signal_methods.gaussian_peak_em(self.lidar_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

    def test_my_lognormal_lm(self):
        """
        Test lognormal_lm
        """
        stream_to_stream = signal_methods.lognormal_lm(self.lidar_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

    def test_my_progressive_waveform_decomposition(self):
        """
        TODO: Fix Unit Test and/or method.
        Error:
        ======================================================================
        ERROR: test_my_progressive_waveform_decomposition (__main__.SignalUnitTest)
        ----------------------------------------------------------------------
        Traceback (most recent call last):
          File "signal_methods_unit_test.py", line 584, in test_my_progressive_waveform_decomposition
            stream_to_stream = signal_methods.my_progressive_waveform_decomposition(self.lidar_data, STREAM_TO_STREAM)
          File "/home/jrick6/repos/emade/src/GPFramework/signal_methods.py", line 2304, in my_progressive_waveform_decomposition
            transmit_region = search(transmitted, multiplier=10)
          File "/home/jrick6/repos/emade/src/GPFramework/signal_methods.py", line 2147, in search
            local_max = find_local_max(deriv, local_min, direction)
          File "/home/jrick6/repos/emade/src/GPFramework/signal_methods.py", line 2081, in find_local_max
            if deriv[start + i + j + 1] >= 0:
        IndexError: index 100 is out of bounds for axis 0 with size 100

        ----------------------------------------------------------------------

        Test my_progressive_waveform_decomposition
        """
        stream_to_stream = signal_methods.my_progressive_waveform_decomposition(self.lidar_data, STREAM_TO_STREAM)
        self.assertIsInstance(stream_to_stream, data.EmadeDataPair)

    def test_copy_stream_to_target(self):
        """
        Test CopyStreamToTarget method
        """
        result = signal_methods.copy_stream_to_target(self.lidar_data)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_my_exp(self):
        """
        Test Exponential method
        """
        result = signal_methods.my_exp(self.feature_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_my_tangent(self):
        """
        Test Tangent method
        """
        result = signal_methods.my_tangent(self.feature_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_my_cosine(self):
        """
        Test Cosine method
        """
        result = signal_methods.my_cosine(self.feature_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_my_sine(self):
        """
        Test Sine method
        """
        result = signal_methods.my_sine(self.feature_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_my_arctangent(self):
        """
        Test ArcTangent method
        """
        result = signal_methods.my_arctangent(self.feature_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_my_arccosine(self):
        """
        Test ArcCosine method
        """
        result = signal_methods.my_arccosine(self.feature_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_my_arcsine(self):
        """
        Test ArcSine method
        """
        result = signal_methods.my_arcsine(self.feature_data, STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

if __name__ == '__main__':
    unittest.main()
