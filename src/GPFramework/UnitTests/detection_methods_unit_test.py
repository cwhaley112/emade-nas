"""
Programmed by Austin Dunn
Contains the unit tests for detection methods
"""
from GPFramework.data import EmadeDataInstance, EmadeDataPair, EmadeData, EmadeDataInstance, FeatureData, StreamData
import GPFramework.data as data
import os
import unittest
import numpy as np
np.random.seed(117)
import cv2

import GPFramework.detection_methods as dm
import GPFramework.spatial_methods as sm
import GPFramework.signal_methods as sim
import GPFramework.learner_methods as lm
import GPFramework.feature_extraction_methods as fm
from GPFramework.gp_framework_helper import LearnerType, EnsembleType
from GPFramework.constants import Axis, TriState
import GPFramework.gp_framework_helper as gp_framework_helper
from deap import gp
import copy as cp

import matplotlib.pyplot as plt

pset = gp.PrimitiveSetTyped('MAIN', [EmadeDataPair], EmadeDataPair)
gp_framework_helper.addPrimitives(pset)
gp_framework_helper.addTerminals(pset)

class DetectionUnitTest(unittest.TestCase):
    """
    This class contains the unit tests for the new EMADE data classes
    """
    @classmethod
    def setUpClass(cls):
        cls.detection_train_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/detection_unit_test.npz')
        cls.detection_test_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/detection_unit_test.npz')
        detection_train = data.load_pickle_from_file(cls.detection_train_path)
        detection_test = data.load_pickle_from_file(cls.detection_test_path)
        cls.detection_data = EmadeDataPair(detection_train, detection_test)

    def setUp(self):
        """
        Dereferences the self.<DataPair> object attributes from the cls.<DataPair> class attributes.
        """
        self.detection_data = cp.deepcopy(self.detection_data)
        self.detection_data.set_datatype('detectiondata')
    def test_conv_channel_merge(self):
        """
        Test ConvolveChannelMerge method
        :return:
        """
        result = dm.conv_channel_merge(self.detection_data, 3)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)

    def test_cv2_template_matching(self):
        """
        Test Cv2TemplateMatching method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        result = dm.cv2_template_matching(result, 3, 0)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)

    def test_label_objects(self):
        """
        Test LabelObjects method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        binary_threshold = dm.dew.get_registry()['MyBinaryThreshold']['function']
        result = binary_threshold(result, TriState.STREAM_TO_STREAM, Axis.FULL, 2000)
        result = dm.label_objects(result, TriState.STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, EmadeDataPair)

    def test_label_by_com(self):
        """
        Test LabelByCenterOfMass method
        :return:
        """
        binary_threshold = dm.dew.get_registry()['MyBinaryThreshold']['function']
        result = binary_threshold(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2000)
        result = dm.label_by_com(result, self.detection_data, TriState.STREAM_TO_STREAM, TriState.STREAM_TO_STREAM, Axis.FULL, Axis.FULL)
        self.assertIsInstance(result, EmadeDataPair)

    def test_normal_likelihood(self):
        """
        Test NormalLikelihood method
        :return:
        """
        normal_likelihood = dm.dewb.get_registry()['NormalLikelihood']['function']
        result = normal_likelihood(self.detection_data, 0)
        # for instance in result.get_train_data().get_instances():
        #     print(instance.get_stream().get_data().shape)
        #
        # for instance in result.get_test_data().get_instances():
        #     print(instance.get_stream().get_data().shape)

        self.assertIsInstance(result, EmadeDataPair)

    def test_matched_filter_2D(self):
        """
        Test matched_filtering_2d method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        matched_filtering_2d = dm.dew.get_registry()['MatchedFiltering2D']['function']
        result = matched_filtering_2d(result, TriState.STREAM_TO_STREAM, Axis.FULL, 11, 7.5)
        # plt.subplot(211)
        # plt.imshow(self.detection_data.get_train_data().get_instances()[0].get_stream().get_data()[:,:,0])
        # plt.subplot(212)
        # plt.imshow(result.get_train_data().get_instances()[0].get_stream().get_data())
        # plt.show()
        # for instance in result.get_train_data().get_instances():
        #     print(instance.get_stream().get_data().shape)
        #
        # for instance in result.get_test_data().get_instances():
        #     print(instance.get_stream().get_data().shape)


        self.assertIsInstance(result, EmadeDataPair)


    def test_minimum_filter(self):
        """
        Test minimum_filter method
        :return:
        """
        minimum_filter = dm.dew.get_registry()['MinimumFilter']['function']
        result = minimum_filter(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 7, 0.3)
        # plt.subplot(211)
        # plt.imshow(self.detection_data.get_train_data().get_instances()[0].get_stream().get_data()[:,:,0])
        # plt.subplot(212)
        # plt.imshow(result.get_train_data().get_instances()[0].get_stream().get_data()[:,:,0])
        # plt.show()
        self.assertIsInstance(result, EmadeDataPair)


    def test_maximum_filter(self):
        """
        Test maximum_filter method
        :return:
        """
        maximum_filter = dm.dew.get_registry()['MaximumFilter']['function']
        result = maximum_filter(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 7, 27.0)
        # plt.subplot(211)
        # plt.imshow(self.detection_data.get_train_data().get_instances()[0].get_stream().get_data()[:,:,0])
        # plt.subplot(212)
        # plt.imshow(result.get_train_data().get_instances()[0].get_stream().get_data()[:,:,0])
        # plt.show()
        self.assertIsInstance(result, EmadeDataPair)

    def test_gaussian_filter(self):
        """
        Test gaussian_filter method
        :return:
        """
        gaussian_filter = dm.dew.get_registry()['GaussianFilter']['function']
        result = gaussian_filter(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 3.0)
        # plt.subplot(211)
        # plt.imshow(self.detection_data.get_train_data().get_instances()[0].get_stream().get_data()[:,:,0])
        # plt.subplot(212)
        # plt.imshow(result.get_train_data().get_instances()[0].get_stream().get_data()[:,:,0])
        # plt.show()
        self.assertIsInstance(result, EmadeDataPair)


    def test_sobel_filter(self):
        """
        Test sobel_filter method
        :return:
        """
        sobel_filter = dm.dew.get_registry()['SobelFilter']['function']
        result = sobel_filter(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL)
        # plt.subplot(211)
        # plt.imshow(self.detection_data.get_train_data().get_instances()[0].get_stream().get_data()[:,:,0])
        # plt.subplot(212)
        # plt.imshow(result.get_train_data().get_instances()[0].get_stream().get_data()[:,:,0])
        # plt.show()
        self.assertIsInstance(result, EmadeDataPair)


    def test_binary_threshold(self):
        """
        Test binary_threshold method
        :return:
        """
        binary_threshold = dm.dew.get_registry()['MyBinaryThreshold']['function']
        result = binary_threshold(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2000)
        # plt.subplot(211)
        # plt.imshow(self.detection_data.get_train_data().get_instances()[0].get_stream().get_data()[:,:,0])
        # plt.subplot(212)
        # plt.imshow(result.get_train_data().get_instances()[0].get_stream().get_data()[:,:,0])
        # plt.show()
        self.assertIsInstance(result, EmadeDataPair)


    def test_ski_median_filter(self):
        """
        Test ski_median_filter method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        ski_median_filter = dm.dew.get_registry()['SkiMedianFilter']['function']
        result = ski_median_filter(result, TriState.STREAM_TO_STREAM, Axis.FULL, 12)
        # plt.subplot(211)
        # plt.imshow(self.detection_data.get_train_data().get_instances()[0].get_stream().get_data()[:,:,0])
        # plt.subplot(212)
        # plt.imshow(result.get_train_data().get_instances()[0].get_stream().get_data())
        # plt.show()
        self.assertIsInstance(result, EmadeDataPair)

    def test_doh_detection(self):
        """
        Test doh_detection method
        :return:
        """
        doh_detection = dm.dew.get_registry()['DohDetection']['function']
        result = doh_detection(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 1.0, 5.0, 5, 2000, 0.5)
        result = dm.get_centroids(result, TriState.STREAM_TO_STREAM, Axis.FULL)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)

    def test_dog_detection(self):
        """
        Test dog_detection method
        :return:
        """
        dog_detection = dm.dew.get_registry()['DogDetection']['function']
        result = dog_detection(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 1.0, 5.0, 1.6, 2000, 0.5)
        result = dm.get_centroids(result, TriState.STREAM_TO_STREAM, Axis.FULL)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)


    def test_log_detection(self):
        """
        Test log_detection method
        :return:
        """
        log_detection = dm.dew.get_registry()['LogDetection']['function']
        result = log_detection(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 1.0, 5.0, 5, 2000, 0.5)
        result = dm.get_centroids(result, TriState.STREAM_TO_STREAM, Axis.FULL)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)


    def test_anomaly_detection(self):
        """
        Test anomaly_detection method
        :return:
        """
        result = dm.rx_anomaly_detector(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 0.9)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)

    def test_object_detection(self):
        """
        Test basic matched filtering method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        result = dm.object_detection(result, 5, 5.0, 90.0)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)

    def test_sep_object_detection(self):
        """
        Test SEP matched filtering method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        result = dm.sep_object_detection(result, 5, 5.0, 70000.0)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)


    def test_sep_detection_window(self):
        """
        Test SEP detection window method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        result = dm.sep_detection_window(result, 0, 5, 5.0)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)

    def test_maximum_window(self):
        """
        Test Maximum window method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        result = dm.maximum_window(result, 0, 7, 1.0)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)

    def test_filter_centroids(self):
        """
        Test filter centroids method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        result = dm.maximum_window(result, 1, 7, 1.0)
        result = fm.hog_feature(result, TriState.FEATURES_TO_FEATURES, Axis.AXIS_0, 0, 8, 3, 1)
        learner_type = LearnerType("RAND_FOREST", {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0})
        ensemble_type = EnsembleType("SINGLE", None)
        result = lm.learner(result, learner_type, ensemble_type)
        result = dm.filter_centroids(result)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)

    def test_ccorr_object_filter(self):
        """
        Test cross-correlation object filter method
        :return:
        """
        result1 = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        result2 = dm.sep_object_detection(result1, 5, 5.0, 70000.0)
        result2 = sim.copy_stream_to_target(result2)
        result = dm.ccorr_object_filter(result2, result1, 0, 6.6e-3)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)

    def test_get_centroids(self):
        """
        Test get_centroid method
        :return:
        """
        log_detection = dm.dew.get_registry()['LogDetection']['function']
        result = log_detection(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 1.0, 5.0, 5, 3000, 0.5)
        get_centroids = dm.dew.get_registry()['GetCentroids']['function']
        result = get_centroids(result, TriState.STREAM_TO_STREAM, Axis.FULL)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_target())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)

    def test_max_loc(self):
        """
        Test max location method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        result = dm.max_loc(result, TriState.STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_min_loc(self):
        """
        Test min location method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        result = dm.min_loc(result, TriState.STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

    def test_create_bbox(self):
        """
        Test create bounding box method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        result = dm.max_loc(result, TriState.STREAM_TO_STREAM, Axis.FULL)
        self.assertIsInstance(result, data.EmadeDataPair)

        result = dm.create_bbox(result, self.detection_data, TriState.STREAM_TO_STREAM, TriState.STREAM_TO_STREAM, Axis.FULL, Axis.FULL, 3)
        self.assertIsInstance(result, data.EmadeDataPair)

if __name__ == '__main__':
    unittest.main()
