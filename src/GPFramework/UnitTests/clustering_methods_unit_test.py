"""
Contains the unit tests for clustering methods
"""
from GPFramework.data import load_many_to_many_from_file, EmadeDataPair
import GPFramework.clustering_methods as cm
import os
import unittest
import numpy as np
np.random.seed(117)
import copy as cp
import argparse as ap

class ClusterUnitTest(unittest.TestCase):
    """
    This class contains the unit tests for clustering methods
    """
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/cluster_data.csv.gz')
        data = load_many_to_many_from_file(cls.data_path)
        cls.data_pair = EmadeDataPair(cp.deepcopy(data), cp.deepcopy(data))

    def setUp(self):
        """
        Dereferences the self.<DataPair> object attributes from the cls.<DataPair> class attributes.
        """
        self.data_pair = cp.deepcopy(self.data_pair)

    def test_affinity_propagation(self):
        """
        Test affinity_propagation
        """
        output = cm.affinity_propagation(self.data_pair, 0.5)
        self.assertIsInstance(output, EmadeDataPair)

    def test_mean_shift(self):
        """
        Test mean_shift
        """
        output = cm.mean_shift(self.data_pair)
        self.assertIsInstance(output, EmadeDataPair)

    def test_db_scan(self):
        """
        Test affinity_propagation
        """
        output = cm.db_scan(self.data_pair, 0.5, 2)
        self.assertIsInstance(output, EmadeDataPair)

    def test_spectral_cluster(self):
        """
        Test spectral_cluster
        """
        output = cm.spectral_cluster(self.data_pair, 2, 1)
        self.assertIsInstance(output, EmadeDataPair)

    def test_k_means_cluster(self):
        """
        Test k_means_cluster
        """
        output = cm.k_means_cluster(self.data_pair, 2)
        self.assertIsInstance(output, EmadeDataPair)

    def test_agglomerative_cluster(self):
        """
        Test agglomerative_cluster
        """
        output = cm.agglomerative_cluster(self.data_pair, 2)
        self.assertIsInstance(output, EmadeDataPair)

    def test_birch_cluster(self):
        """
        Test birch_cluster
        """
        output = cm.birch_cluster(self.data_pair, 0.1, 50, 2)
        self.assertIsInstance(output, EmadeDataPair)


if __name__ == '__main__':
    unittest.main()
