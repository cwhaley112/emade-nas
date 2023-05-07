"""
Programmed by Jason Zutty
Modified by Austin Dunn
Contains the unit tests for the new data class
"""
from GPFramework.data import EmadeDataInstance, EmadeDataPair
import GPFramework.data as data
import os
import unittest
import numpy as np
np.random.seed(117)
import copy as cp
import matplotlib.pyplot as plt


class DataUnitTest(unittest.TestCase):  #pylint: disable=R0904
    """
    This class contains the unit tests for the new EMADE data classes
    """
    @classmethod
    def setUpClass(cls):
        cls.time_data_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/test3axistrain.txt.gz')
        cls.filter_data_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/filter_data.txt.gz')
        cls.feature_data_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/train_data_v2_suit_1-5.csv.gz')
        cls.many_to_some_data_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/test_many_to_some.csv.gz')

        # Time data from aglogica
        cls.time_data, _ = data.load_many_to_one_from_file(cls.time_data_path)
        time_data_targets = np.stack([inst.get_target() for inst in cls.time_data.get_instances()])
        time_data_indices = np.array([]).astype(int)
        for target in np.unique(time_data_targets):
            indices = np.where(time_data_targets == target)[0]
            time_data_indices = np.hstack((time_data_indices, np.random.choice(indices, size=min(indices.shape[0], 10), replace=False))).astype(int)
        cls.time_data.set_instances(np.array(cls.time_data.get_instances())[time_data_indices])
        print("\nReduced time_data:", cls.time_data.get_numpy().shape, end=" ", flush=True)

        # Filter data
        cls.filter_data, _ = data.load_many_to_many_from_file(cls.filter_data_path)
        cls.filter_data.set_instances(np.random.choice(cls.filter_data.get_instances(), size=100, replace=False))
        print("Reduced filter_data:", cls.filter_data.get_numpy().shape, end=" ", flush=True)

        # Feature data from chemical companion
        cls.feature_data, _ = data.load_feature_data_from_file(cls.feature_data_path)
        cls.feature_data.set_instances(np.random.choice(cls.feature_data.get_instances(), size=100, replace=False))
        print("Reduced feature_data:", cls.feature_data.get_numpy().shape, end=" ", flush=True)

        # Imagery with centroid locations
        cls.many_to_some, _ = data.load_many_to_some_from_file(cls.many_to_some_data_path)
        cls.many_to_some.set_instances(np.random.choice(cls.many_to_some.get_instances(), size=100, replace=False))
        print("Reduced many_to_some:", cls.many_to_some.get_numpy().shape, end=" ", flush=True)
    def setUp(self):
        """
        Dereferences the self.<DataPair> object attributes from the cls.<DataPair> class attributes.
        """
        self.time_data = cp.deepcopy(self.time_data)
        self.filter_data = cp.deepcopy(self.filter_data)
        self.feature_data = cp.deepcopy(self.feature_data)
        self.many_to_some = cp.deepcopy(self.many_to_some)

    def test_addition(self):
        """
        Test the addition of two EmadeDataInstance objects
        """
        var1 = self.time_data.get_instances()[0] + self.time_data.get_instances()[1]
        var2 = self.feature_data.get_instances()[0] + self.feature_data.get_instances()[1]
        var3 = self.filter_data.get_instances()[0] + self.filter_data.get_instances()[1]
        var4 = self.many_to_some.get_instances()[0] + self.many_to_some.get_instances()[1]
        #this plot can be used to manually check that addition is occuring for each element:
#        plt.plot(self.filter_data.get_instances()[0].get_stream().get_data()[0])
#        plt.plot(self.filter_data.get_instances()[1].get_stream().get_data()[0])
#        plt.plot(var3.get_stream().get_data()[0])
#        plt.plot(self.filter_data.get_instances()[0].get_target()[0])
#        plt.plot(self.filter_data.get_instances()[1].get_target()[0])
#        plt.plot(var3.get_target()[0])
#        print(var3.get_target())
#        plt.show()
#        print(var1)
#        print(var2)
#        print(var3)
        self.assertIsInstance(var1, data.EmadeDataInstance)
        self.assertIsInstance(var2, data.EmadeDataInstance)
        self.assertIsInstance(var3, data.EmadeDataInstance)
        self.assertIsInstance(var4, data.EmadeDataInstance)



    def test_addition_scalar(self):
        """
        Test the addition of one EmadeDataInstance objects and a scalar
        """
        var1 = self.time_data.get_instances()[0] + 5.0
        var2 = self.feature_data.get_instances()[0] + 5.0
        var3 = self.many_to_some.get_instances()[0] + 5.0

        self.assertIsInstance(var1, data.EmadeDataInstance)
        self.assertIsInstance(var2, data.EmadeDataInstance)
        self.assertIsInstance(var3, data.EmadeDataInstance)


    def test_subtraction(self):
        """
        Test the subtraction of two EmadeDataInstance objects
        """
        var1 = self.time_data.get_instances()[0] - self.time_data.get_instances()[1]
        var2 = self.feature_data.get_instances()[0] - self.feature_data.get_instances()[1]
        var3 = self.many_to_some.get_instances()[0] - self.many_to_some.get_instances()[1]

        self.assertIsInstance(var1, data.EmadeDataInstance)
        self.assertIsInstance(var2, data.EmadeDataInstance)
        self.assertIsInstance(var3, data.EmadeDataInstance)


    def test_subtraction_scalar(self):
        """
        Test the subtraction of one EmadeDataInstance objects and a scalar
        """
        var1 = self.time_data.get_instances()[0] - 5.0
        var2 = self.feature_data.get_instances()[0] - 5.0
        var3 = self.many_to_some.get_instances()[0] - 5.0

        self.assertIsInstance(var1, data.EmadeDataInstance)
        self.assertIsInstance(var2, data.EmadeDataInstance)
        self.assertIsInstance(var3, data.EmadeDataInstance)


    def test_multiplication(self):
        """
        Test the multiplication of two EmadeDataInstance objects
        """
        var1 = self.time_data.get_instances()[0] * self.time_data.get_instances()[1]
        var2 = self.feature_data.get_instances()[0] * self.feature_data.get_instances()[1]
        var3 = self.many_to_some.get_instances()[0] * self.many_to_some.get_instances()[1]

        self.assertIsInstance(var1, data.EmadeDataInstance)
        self.assertIsInstance(var2, data.EmadeDataInstance)
        self.assertIsInstance(var3, data.EmadeDataInstance)


    def test_multiplication_scalar(self):
        """
        Test the multiplication of one EmadeDataInstance objects and a scalar
        """
        var1 = self.time_data.get_instances()[0] * 5.0
        var2 = self.feature_data.get_instances()[0] * 5.0
        var3 = self.many_to_some.get_instances()[0] * 5.0

        self.assertIsInstance(var1, data.EmadeDataInstance)
        self.assertIsInstance(var2, data.EmadeDataInstance)
        self.assertIsInstance(var3, data.EmadeDataInstance)


    def test_division(self):
        """
        Test the division of two EmadeDataInstance objects
        """
        var1 = self.time_data.get_instances()[0] / (self.time_data.get_instances()[1] + 0.0000001)
        var2 = self.feature_data.get_instances()[0] / (self.feature_data.get_instances()[1] + 0.0000001)
        var3 = self.many_to_some.get_instances()[0] / (self.many_to_some.get_instances()[1] + 0.0000001)

        self.assertIsInstance(var1, data.EmadeDataInstance)
        self.assertIsInstance(var2, data.EmadeDataInstance)
        self.assertIsInstance(var3, data.EmadeDataInstance)


    def test_division_scalar(self):
        """
        Test the division of one EmadeDataInstance objects and a scalar
        """
        var1 = self.time_data.get_instances()[0] / 5.0
        var2 = self.feature_data.get_instances()[0] / 5.0
        var3 = self.many_to_some.get_instances()[0] / 5.0

        self.assertIsInstance(var1, data.EmadeDataInstance)
        self.assertIsInstance(var2, data.EmadeDataInstance)
        self.assertIsInstance(var3, data.EmadeDataInstance)

if __name__ == '__main__':
    unittest.main()
