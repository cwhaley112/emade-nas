"""
Programmed by Austin Dunn
Contains the unit tests for the operator methods
"""
from GPFramework.constants import FEATURES_TO_FEATURES, STREAM_TO_STREAM, STREAM_TO_FEATURES, Axis
from GPFramework.data import load_feature_data_from_file, EmadeDataPair
import GPFramework.operator_methods as op_methods
import os
import unittest
import numpy as np
np.random.seed(117)
import copy as cp

class OperatorUnitTest(unittest.TestCase):
    """
    This class contains the unit tests for operator primitives
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

    def test_my_frac_double(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_frac_double(self.feature_data, self.feature_data, FEATURES_TO_FEATURES, FEATURES_TO_FEATURES, Axis.FULL, Axis.FULL, 2)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_frac_triple(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_frac_triple(self.feature_data, self.feature_data, self.feature_data, FEATURES_TO_FEATURES, FEATURES_TO_FEATURES, FEATURES_TO_FEATURES,
                                                                                                      Axis.FULL, Axis.FULL, Axis.FULL, 2)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_fraction(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.fraction(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 2)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_add_pair(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_add_pair(self.feature_data, self.feature_data, FEATURES_TO_FEATURES, FEATURES_TO_FEATURES, Axis.FULL, Axis.FULL)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_add_pair_triple(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_add_pair_triple(self.feature_data, self.feature_data, self.feature_data, FEATURES_TO_FEATURES, FEATURES_TO_FEATURES, FEATURES_TO_FEATURES,
                                                                                                      Axis.FULL, Axis.FULL, Axis.FULL)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_add(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_add(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 2)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_add_float(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_add_float(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 2.0)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_subtract_pair(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_subtract_pair(self.feature_data, self.feature_data, FEATURES_TO_FEATURES, FEATURES_TO_FEATURES, Axis.FULL, Axis.FULL)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_subtract(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_subtract(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 2)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_subtract_float(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_subtract_float(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 2.0)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_divide_pair(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_divide_pair(self.feature_data, self.feature_data, FEATURES_TO_FEATURES, FEATURES_TO_FEATURES, Axis.FULL, Axis.FULL)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_divide(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_divide(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 2)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_divide_float(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_divide_float(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 2.0)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_divide_int(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_divide_int(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 2.0)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_divide_int_pair(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_divide_int_pair(self.feature_data, self.feature_data, FEATURES_TO_FEATURES, FEATURES_TO_FEATURES, Axis.FULL, Axis.FULL)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_multiply_pair(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_multiply_pair(self.feature_data, self.feature_data, FEATURES_TO_FEATURES, FEATURES_TO_FEATURES, Axis.FULL, Axis.FULL)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_multiply(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_multiply(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 2)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_multiply_float(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_multiply_float(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 2.0)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_np_multiply_pair(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_np_multiply_pair(self.feature_data, self.feature_data, FEATURES_TO_FEATURES, FEATURES_TO_FEATURES, Axis.FULL, Axis.FULL)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_np_multiply(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_np_multiply(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 2)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

    def test_my_np_multiply_float(self):
        # Store initial shape
        starting_shape = self.feature_data.get_train_data().get_numpy().shape
        # Run Primitive
        test = op_methods.my_np_multiply_float(self.feature_data, FEATURES_TO_FEATURES, Axis.FULL, 2.0)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_numpy().shape
        # Assert that the shape of the data did not change
        self.assertEqual(starting_shape, ending_shape)

if __name__ == '__main__':
    unittest.main()
