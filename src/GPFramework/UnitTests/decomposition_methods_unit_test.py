"""
Programmed by Austin Dunn
Contains the unit tests for decomposition methods
"""
from GPFramework.data import load_feature_data_from_file, EmadeDataPair
from GPFramework.constants import FEATURES_TO_FEATURES, STREAM_TO_STREAM, STREAM_TO_FEATURES
import GPFramework.decomposition_methods as dm
import os
import unittest
import numpy as np
np.random.seed(117)
import copy as cp

class DecompositionUnitTest(unittest.TestCase):
    """
    This class contains the unit tests for clustering methods
    """
    @classmethod
    def setUpClass(cls):
        cls.feature_data_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/train_data_v2_suit_1-5.csv.gz')
        feature_data = load_feature_data_from_file(cls.feature_data_path)
        cls.feature_data = EmadeDataPair(cp.deepcopy(feature_data), cp.deepcopy(feature_data))

    def setUp(self):
        """
        Dereferences the self.<DataPair> object attributes from the cls.<DataPair> class attributes.
        """
        self.feature_data = cp.deepcopy(self.feature_data)

    def test_my_pca(self):
        """
        Test my_pca
        """
        output = dm.my_pca(self.feature_data, FEATURES_TO_FEATURES, 3)
        self.assertIsInstance(output, EmadeDataPair)

    def test_my_sparse_pca(self):
        """
        Test my_sparse_pca
        """
        output = dm.my_sparse_pca(self.feature_data, FEATURES_TO_FEATURES, 3)
        self.assertIsInstance(output, EmadeDataPair)

    def test_my_ica(self):
        """
        Test my_ica
        """
        output = dm.my_ica(self.feature_data, FEATURES_TO_FEATURES, 3)
        self.assertIsInstance(output, EmadeDataPair)

    def test_my_spectral_embedding(self):
        """
        Test my_spectral_embedding
        """
        output = dm.my_spectral_embedding(self.feature_data, FEATURES_TO_FEATURES, 3)
        self.assertIsInstance(output, EmadeDataPair)


if __name__ == '__main__':
    unittest.main()
