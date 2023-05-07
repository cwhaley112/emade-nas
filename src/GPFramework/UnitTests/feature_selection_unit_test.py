"""
Programmed by Jason Zutty
Modified by Austin Dunn
Contains the unit tests for the feature selection methods
"""
from GPFramework.constants import FEATURES_TO_FEATURES, STREAM_TO_STREAM, STREAM_TO_FEATURES
from GPFramework.data import load_feature_data_from_file, EmadeDataPair
import GPFramework.feature_selection_methods as feature_selection
import os
import unittest
import numpy as np
np.random.seed(117)
import copy as cp

class FeatureSelectUnitTest(unittest.TestCase):  #pylint: disable=R0904
    """
    This class contains the unit tests for the new EMADE data classes
    """
    @classmethod
    def setUpClass(cls):
        """
        Create two EmadeDataInstance objects
        """
        cls.feature_data_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/train_data_v2_suit_1-5.csv.gz')
        feature_data = load_feature_data_from_file(cls.feature_data_path)
        cls.feature_data = EmadeDataPair(cp.deepcopy(feature_data), cp.deepcopy(feature_data))

    def setUp(self):
        """
        Dereferences the self.<DataPair> object attributes from the cls.<DataPair> class attributes.
        """
        self.feature_data = cp.deepcopy(self.feature_data)

    def test_select_k_best(self):
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        test = feature_selection.select_k_best_scikit(self.feature_data, FEATURES_TO_FEATURES, 0, 10)
        ending_shape = test.get_train_data().get_numpy().shape
        self.assertEqual(10, ending_shape[1])
        self.assertEqual(starting_shape[0], ending_shape[0])

    def test_select_percentile(self):
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        test = feature_selection.select_percentile_scikit(self.feature_data, FEATURES_TO_FEATURES, 0, 10)
        ending_shape = test.get_train_data().get_numpy().shape
        self.assertEqual(2, ending_shape[1])
        self.assertEqual(starting_shape[0], ending_shape[0])

    def test_select_fpr(self):
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        test = feature_selection.select_fpr_scikit(self.feature_data, FEATURES_TO_FEATURES, 0, 0.05)
        ending_shape = test.get_train_data().get_numpy().shape
        self.assertEqual(5, ending_shape[1])
        self.assertEqual(starting_shape[0], ending_shape[0])

    def test_select_fdr(self):
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        test = feature_selection.select_fdr_scikit(self.feature_data, FEATURES_TO_FEATURES, 0, 0.05)
        ending_shape = test.get_train_data().get_numpy().shape
        self.assertEqual(4, ending_shape[1])
        self.assertEqual(starting_shape[0], ending_shape[0])

    def test_select_generic_univariate(self):
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        test = feature_selection.select_generic_univariate_scikit(self.feature_data, FEATURES_TO_FEATURES, 0, 0, 0.05)
        ending_shape = test.get_train_data().get_numpy().shape
        self.assertEqual(1, ending_shape[1])
        self.assertEqual(starting_shape[0], ending_shape[0])

    def test_select_fwe(self):
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        test = feature_selection.select_fwe_scikit(self.feature_data, FEATURES_TO_FEATURES, 0, 0.05)
        ending_shape = test.get_train_data().get_numpy().shape
        self.assertEqual(4, ending_shape[1])
        self.assertEqual(starting_shape[0], ending_shape[0])

    def test_variance_threshold(self):
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # print(self.feature_data.get_train_data())
        test = feature_selection.variance_threshold_scikit(self.feature_data, FEATURES_TO_FEATURES, 5.0)
        ending_shape = test.get_train_data().get_numpy().shape
        self.assertEqual(8, ending_shape[1])
        self.assertEqual(starting_shape[0], ending_shape[0])



if __name__ == '__main__':
    unittest.main()
