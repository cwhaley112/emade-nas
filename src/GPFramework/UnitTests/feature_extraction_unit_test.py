"""
Programmed by Austin Dunn
Contains the unit tests for the feature extraction methods
"""
from GPFramework.constants import FEATURES_TO_FEATURES, STREAM_TO_STREAM, STREAM_TO_FEATURES, Axis
from GPFramework.data import load_feature_data_from_file, EmadeDataPair, EmadeData, EmadeDataInstance, StreamData
import GPFramework.feature_extraction_methods as fe_methods
import os
import unittest
import numpy as np
np.random.seed(117)
import copy as cp
import cv2

class FeatureExtractionUnitTest(unittest.TestCase):
    """
    This class contains the unit tests for feature extraction primitives
    """
    @classmethod
    def setUpClass(cls):
        cls.img_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/cokecangray.jpg')
        cls.img = cv2.imread(cls.img_path)
        cls.img = cv2.cvtColor(cls.img, cv2.COLOR_BGR2GRAY)

        stream = StreamData(cls.img)
        instances = [EmadeDataInstance(stream=stream)]
        image_data = EmadeData(instances)
        cls.image_data = EmadeDataPair((cp.deepcopy(image_data), None), (cp.deepcopy(image_data), None))

    def setUp(self):
        """
        Dereferences the self.<DataPair> object attributes from the cls.<DataPair> class attributes.
        """
        self.image_data = cp.deepcopy(self.image_data)

    def test_hog_feature(self):
        # Run Primitive
        test = fe_methods.hog_feature(self.image_data, STREAM_TO_STREAM, Axis.FULL, False, 8, 8, 8)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_instances()[0].get_stream().get_data().shape
        print("HOG Output Shape:", ending_shape)
        # Assert that the shape of output is 1d
        self.assertEqual(len(ending_shape), 1)

    def test_daisy_feature(self):
        # Run Primitive
        test = fe_methods.daisy_feature(self.image_data, STREAM_TO_STREAM, Axis.FULL, 25, 10, 3, 8, 8)
        # Assert output is a EmadeDataPair
        self.assertIsInstance(test, EmadeDataPair)
        # Store ending shape
        ending_shape = test.get_train_data().get_instances()[0].get_stream().get_data().shape
        print("DAISY Output Shape:", ending_shape)
        # Assert that the shape of output is 3d
        self.assertEqual(len(ending_shape), 3)

if __name__ == '__main__':
    unittest.main()
