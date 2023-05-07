"""
Programmed by Jason Zutty
Modified by Austin Dunn
Contains the unit tests for the machine learning methods
"""
from GPFramework.constants import FEATURES_TO_FEATURES, STREAM_TO_STREAM, STREAM_TO_FEATURES, Axis
import GPFramework.data as data
from GPFramework.learner_methods import learner
from GPFramework.gp_framework_helper import LearnerType, EnsembleType
import GPFramework.signal_methods as signal_methods
import os
import unittest
import numpy as np
np.random.seed(117)
import copy as cp


class MethodsUnitTest(unittest.TestCase):  #pylint: disable=R0904
    """
    This class contains the unit tests for the new EMADE data classes
    """

    @classmethod
    def setUpClass(cls):
        cls.time_data_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/test3axistrain.txt.gz')
        cls.feature_data_path = os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/train_data_v2_suit_1-5.csv.gz')

        # Time data from aglogica
        time_data = data.load_many_to_one_from_file(cls.time_data_path)
        time_data_targets = np.stack([inst.get_target() for inst in time_data[0].get_instances()])
        time_data_indices = np.array([]).astype(int)
        for target in np.unique(time_data_targets):
            indices = np.where(time_data_targets == target)[0]
            time_data_indices = np.hstack((time_data_indices, np.random.choice(indices, size=min(indices.shape[0], 10), replace=False))).astype(int)
        time_data[0].set_instances(np.array(time_data[0].get_instances())[time_data_indices])
        print("Reduced Time Data", len(time_data[0].get_instances()), flush=True)
        # Construct a data pair
        cls.time_data = data.EmadeDataPair(cp.deepcopy(time_data), cp.deepcopy(time_data))
        # Feature data from chemical companion
        feature_data = data.load_feature_data_from_file(cls.feature_data_path)
        # Construct a data pair
        cls.feature_data = data.EmadeDataPair(cp.deepcopy(feature_data), cp.deepcopy(feature_data))

    def setUp(self):
        """
        Dereferences the self.<DataPair> object attributes from the cls.<DataPair> class attributes.
        """
        self.time_data = cp.deepcopy(self.time_data)
        self.feature_data = cp.deepcopy(self.feature_data)

    def test_knn(self):
        learner_type = LearnerType("KNN", {'K': 3, 'weights':0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_bayes(self):
        learner_type = LearnerType("BAYES", None)
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_omp(self):
        learner_type = LearnerType("OMP", None)
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_svm(self):
        learner_type = LearnerType("SVM", {'C':1.0, 'kernel':0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_trees(self):
        learner_type = LearnerType("DECISION_TREE", {'criterion':0, 'splitter':0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_random_forest(self):
        learner_type = LearnerType("RAND_FOREST", {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_boosting(self):
        learner_type = LearnerType("BOOSTING", {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_blup_learner(self):
        learner_type = LearnerType("BLUP", None)
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_bagging(self):
        learner_type = LearnerType("KNN", {'K': 3, 'weights':0})
        ensemble_type = EnsembleType("BAGGED", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_adaboost(self):
        learner_type = LearnerType("RAND_FOREST", {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0})
        ensemble_type = EnsembleType("ADABOOST", {'n_estimators': 50, 'learning_rate':1.0})
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_logistic_regression_scikit(self):
        learner_type = LearnerType("LOGR", {'penalty':0, 'C':1.0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_gmm_scikit(self):
        learner_type = LearnerType("GMM", {'n_components':2, 'covariance_type':0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_my_arg_max(self):
        learner_type = LearnerType("ARGMAX", {'sampling_rate':1})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_my_arg_min(self):
        learner_type = LearnerType("ARGMIN", {'sampling_rate':1})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_my_depth_estimate(self):
        learner_type = LearnerType("DEPTH_ESTIMATE", {'sampling_rate':1, 'off_nadir_angle':20.0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_extra_trees(self):
        learner_type = LearnerType("EXTRATREES", {'n_estimators':100, 'max_depth':6, 'criterion':0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

# -----------------------------------------  Regression Methods ------------------------------------------------------ #

    def test_xgboost(self):
        learner_type = LearnerType("XGBOOST", {
            'learning_rate': 0.1,
            'max_depth': 3,
            'n_estimators': 100,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_lightgbm(self):
        learner_type = LearnerType("LIGHTGBM", {
            'max_depth': -1,
            'learning_rate': 0.1,  # shrinkage_rate
            'boosting_type': 0,
            'num_leaves': 31})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_boosting_regression(self):
        learner_type = LearnerType("BOOSTING_REGRESSION", {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_adaboost_regression(self):
        learner_type = LearnerType("ADABOOST_REGRESSION", {'learning_rate': 0.1, 'n_estimators': 100})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_random_forest_regression(self):
        learner_type = LearnerType("RANDFOREST_REGRESSION", {'n_estimators': 100, 'criterion':0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_svm_regression(self):
        learner_type = LearnerType("SVM_REGRESSION", {'kernel':0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)

    def test_knn_regression(self):
        learner_type = LearnerType("KNN_REGRESSION", {'K': 3, 'weights':0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)

        stream_test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        stream_test = learner(stream_test, learner_type, ensemble_type)
        self.assertIsInstance(stream_test, data.EmadeDataPair)




if __name__ == '__main__':
    unittest.main()
