"""
Programmed by Jason Zutty
Modified by Austin Dunn
Contains the unit tests for the machine learning methods
"""
from GPFramework.constants import FEATURES_TO_FEATURES, STREAM_TO_STREAM, STREAM_TO_FEATURES, Axis
import GPFramework.data as data
from GPFramework.data import EmadeDataInstance, EmadeDataPair, EmadeData, EmadeDataInstance, FeatureData, StreamData
from GPFramework.learner_methods import learner
from GPFramework.gp_framework_helper import LearnerType, EnsembleType
import GPFramework.signal_methods as signal_methods
import unittest
import copy as cp
import GPFramework.eval_methods as em
import numpy as np
import GPFramework.detection_methods as dm
import GPFramework.spatial_methods as sp
import GPFramework.signal_methods as sm
import GPFramework.learner_methods as lm
import GPFramework.feature_extraction_methods as fm
import GPFramework.clustering_methods as cm
from GPFramework.constants import Axis, TriState

class MethodsUnitTest(unittest.TestCase):  #pylint: disable=R090
    """
    This class contains the unit tests for the new EMADE data classes
    """
    def setUp(self):
        """
        Create two EmadeDataPair objects
        """
       #eval methods
        self.fp = em.false_positive
        self.fn = em.false_negative

        import pathlib

        running_path = str(pathlib.Path(__file__).parent.absolute())
        working_path = str(pathlib.Path().absolute())

        pre_path = ""
        lrun = list(running_path)
        lwork = list(working_path)
        leftover = lrun[len(lwork) :]
        pre_path = pre_path.join(leftover)
        if len(pre_path) > 0:
            pre_path += "/"
            pre_path = pre_path[1:]

        feature_data = data.load_feature_data_from_file(pre_path + '../../../unitTestData/train_data_v2_suit_1-5.csv.gz')
        # feature_data_array = [feature_data]

        self.truth_data = feature_data[0].get_target() 

        # self.target_data = feature_data[0].get_target()
        #print("target data: ", self.target_data, "truth data: ", self.truth_data_array[0])
        self.feature_data = EmadeDataPair(cp.deepcopy(feature_data), cp.deepcopy(feature_data))


        detection_train = data.load_pickle_from_file(pre_path + '../../../unitTestData/detection_unit_test.npz')
        detection_test = data.load_pickle_from_file(pre_path + '../../../unitTestData/detection_unit_test.npz')

        self.detection_data = EmadeDataPair(detection_train, detection_test)
        self.detection_data.set_datatype('detectiondata')
        # self.detection_truth_data = detection_test[0].get_target()
        # self.detection_truth_data = np.array([instance.get_target()[0] for instance in detection_test[0].get_instances()])
        #import pdb; pdb.set_trace()
        self.detection_truth_data = detection_test[0].get_target()

        time_data = data.load_many_to_one_from_file(pre_path + '../../../unitTestData/test3axistrain.txt.gz')
        # Construct a data pair
        self.time_data = data.EmadeDataPair(cp.deepcopy(time_data), cp.deepcopy(time_data))
        self.time_data_truth_data = time_data[0].get_target()

        filter_data = data.load_many_to_many_from_file(pre_path + '../../../unitTestData/cluster_data.csv.gz')
        self.filter_data_pair = EmadeDataPair(cp.deepcopy(filter_data), cp.deepcopy(filter_data))
        self.filter_truth_data = filter_data[0].get_target()
        print(self.filter_truth_data.dtype)

        img_data = data.load_images_from_file(pre_path + '../../../datasets/image/small/train9/')
        self.img_data = EmadeDataPair(cp.deepcopy(img_data), cp.deepcopy(img_data))
        self.img_truth_data = img_data[0].get_target()

        #CopyStreamToTarget(CenterOfMass(ThresholdBinary(ARG0, 1, 0.2, 255), 1))
    def test_image(self):
        """
        Test center of mass method
        :return:
        """
        #CopyStreamToTarget(CenterOfMass(ThresholdBinary(ARG0, 1, 0.2, 255), 1))
        print(self.img_data.get_test_data().get_target())
        print(self.img_truth_data)
        #result = sp.center_of_mass(self.many_to_some, STREAM_TO_STREAM, Axis.FULL)
        result = sp.threshold_binary(self.img_data, STREAM_TO_STREAM, Axis.FULL, threshold=128)
        result = sp.center_of_mass(result, STREAM_TO_STREAM, Axis.FULL)
        result = sm.copy_stream_to_target(result)
        test_data_classes = result.get_test_data().get_target()
        print(test_data_classes)
        truth_data = self.truth_data #truth data to be passed in
        #print("test data before shape test in eval method:",test_data_classes, "truth data before shape check in eval method",truth_data) #input into the eval method
        fp = self.fp("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        self.assertIsNot(fp, np.inf)

        self.assertIsInstance(result, data.EmadeDataPair)


    def test_affinity_propagation(self):
        """
        Test affinity_propagation
        """
        output = self.filter_data_pair
        #print(self.filter_truth_data==output.get_test_data().get_target(), self.filter_truth_data==output.get_train_data().get_target())
        output = cm.affinity_propagation(self.filter_data_pair, 0.5)
        self.assertIsInstance(output, EmadeDataPair)
        #print(self.filter_truth_data==output.get_test_data().get_target(), self.filter_truth_data==output.get_train_data().get_target())
        test_data_classes = output.get_test_data().get_target()
        contvar = em.continuous_var(None, self.filter_truth_data, self.filter_truth_data,  name=None)
        print(np.equal(self.filter_truth_data, output.get_test_data().get_target(), dtype=np.object), np.equal(self.filter_truth_data, output.get_train_data().get_target(), dtype=np.object))
        print(self.filter_truth_data, test_data_classes)
        self.assertIsNot(contvar, np.inf)


    
    def test_filter_centroids(self):
        """
        Test filter centroids method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        result = dm.maximum_window(result, 1, 7, 1.0)
        result = fm.hog_feature(result, TriState.FEATURES_TO_FEATURES, Axis.AXIS_0, 0, 8, 3, 1)
        learner_type = LearnerType("RAND_FOREST", {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0})
        #learner_type = LearnerType("ARGMAX", {'sampling_rate':1})
        # learner_type = LearnerType("DEPTH_ESTIMATE", {'sampling_rate':1, 'off_nadir_angle':20.0})
        ensemble_type = EnsembleType("SINGLE", None)
        result = lm.learner(result, learner_type, ensemble_type)
        result = dm.filter_centroids(result)
        # print(result.get_train_data().get_instances()[0].get_stream().get_data())
        # print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)
        test_data_classes = result.get_test_data().get_target()
        #test_data_classes = [np.array(inst.get_target()) for inst in result.get_test_data().get_instances()]
        #test_data_classes = result.get_test_data().get_target()
        #import pdb; pdb.set_trace()
        # print(self.detection_truth_data.shape, test_data_classes)
        dft = em.distance_from_target(None, test_data_classes, self.detection_truth_data,  name=None)
        mdft =  em.mean_dist_from_target(None, test_data_classes, self.detection_truth_data,  name=None)
        fpc = em.false_positive_centroid(None, test_data_classes, self.detection_truth_data,  name=None)
        fnc = em.false_negative_centroid(None, test_data_classes, self.detection_truth_data,  name=None)
        fnc2 = em.false_negative_centroid2(None, test_data_classes, self.detection_truth_data,  name=None)
        self.assertIsNot(dft, np.inf)
        self.assertIsNot(mdft, np.inf)
        self.assertIsNot(fpc, np.inf)
        self.assertIsNot(fnc, np.inf)
        self.assertIsNot(fnc2, np.inf)
        # test_data_classes = [np.array(inst.get_target()) for inst in result.get_test_data().get_instances()]
        self.assertIsInstance(result, EmadeDataPair)
  
    def test_object_detection(self):
        """
        Test basic matched filtering method
        :return:
        """
        result = sm.select_1d(self.detection_data, TriState.STREAM_TO_STREAM, Axis.FULL, 2, 0)
        result = dm.object_detection(result, 5, 5.0, 90.0)
        print(result.get_train_data().get_instances()[0].get_stream().get_data())
        print(result.get_train_data().get_instances()[0].get_stream().get_data().shape)
        self.assertIsInstance(result, EmadeDataPair)

        test_data_classes = result.get_test_data().get_target()
        #test_data_classes = [np.array(inst.get_target()) for inst in result.get_test_data().get_instances()]
        #test_data_classes = result.get_test_data().get_target()
        #import pdb; pdb.set_trace()
        # print(self.detection_truth_data.shape, test_data_classes)
        dft = em.distance_from_target(None, test_data_classes, self.detection_truth_data,  name=None)
        mdft =  em.mean_dist_from_target(None, test_data_classes, self.detection_truth_data,  name=None)
        fpc = em.false_positive_centroid(None, test_data_classes, self.detection_truth_data,  name=None)
        fnc = em.false_negative_centroid(None, test_data_classes, self.detection_truth_data,  name=None)
        fnc2 = em.false_negative_centroid2(None, test_data_classes, self.detection_truth_data,  name=None)
        print(dft, mdft, fpc, fnc, fnc2)
        self.assertIsNot(dft, np.inf)
        self.assertIsNot(mdft, np.inf)
        self.assertIsNot(fpc, np.inf)
        self.assertIsNot(fnc, np.inf)
        self.assertIsNot(fnc2, np.inf)


    def test_trees(self):
        learner_type = LearnerType("DECISION_TREE", {'criterion':0, 'splitter':0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)
        # test_data_classes = [np.array(inst.get_target()) for inst in test.get_test_data().get_instances()] #test data to be passed in
        test_data_classes = test.get_test_data().get_target()
        truth_data = self.truth_data #truth data to be passed in
        #print("test data before shape test in eval method:",test_data_classes, "truth data before shape check in eval method",truth_data) #input into the eval method
        fp = self.fp("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        self.assertIsNot(fp, np.inf)
        # print(test_data_classes, truth_data)
        #This print below statement is for what is passed in before the shape chack that returns infinity: 
        # test_data = np.array(test_data_classes)
        # truth_data = np.array(truth_data)

    def test_trees_stream(self):
        learner_type = LearnerType("DECISION_TREE", {'criterion':0, 'splitter':0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = signal_methods.my_fft(self.time_data, STREAM_TO_FEATURES, Axis.FULL)
        test = learner(test, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)
        # test_data_classes = [np.array(inst.get_target()) for inst in test.get_test_data().get_instances()] #test data to be passed in
        test_data_classes = test.get_test_data().get_target()
        truth_data = self.time_data_truth_data #truth data to be passed in
        print(test_data_classes, truth_data)
        #print("test data before shape test in eval method:",test_data_classes, "truth data before shape check in eval method",truth_data) #input into the eval method
        rms = em.objective0EvalFunction("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        print(rms)
        self.assertIsNot(rms, np.inf)

        under = em.objective1EvalFunction("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        print(under)
        self.assertIsNot(under, np.inf)

        over = em.objective2EvalFunction("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        print(over)
        self.assertIsNot(over, np.inf)

        prob = em.objective4EvalFunction("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        print(prob)
        self.assertIsNot(prob, np.inf)

        percent = em.objective5EvalFunction("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        print(percent)
        self.assertIsNot(percent, np.inf)

        scratching = em.scratching_error_rate("", test_data_classes, truth_data, "test_vec") # eval method output if output is infinity then next print statment is for debugging that.
        print(scratching)
        self.assertIsNot(scratching, np.inf)

        scratching2 = em.scratching_false_alarm_rate("", test_data_classes, truth_data, "test_vec") # eval method output if output is infinity then next print statment is for debugging that.
        print(scratching2)
        self.assertIsNot(scratching2, np.inf)
        
        # print(test_data_classes, truth_data)
        #This print below statement is for what is passed in before the shape chack that returns infinity: 
        # test_data = np.array(test_data_classes)
        # truth_data = np.array(truth_data)
     
    def test_random_forest(self):
        learner_type = LearnerType("RAND_FOREST", {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)
        test_data_classes = test.get_test_data().get_target()
        truth_data = self.truth_data #truth data to be passed in
        #print("test data before shape test in eval method:",test_data_classes, "truth data before shape check in eval method",truth_data) #input into the eval method
        fn = self.fn("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        self.assertIsNot(fn, np.inf)
        #print(test_data_classes, truth_data)
        #This print below statement is for what is passed in before the shape chack that returns infinity: 
  


        
      

    def test_logistic_regression_scikit(self):
        learner_type = LearnerType("LOGR", {'penalty':0, 'C':1.0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)
        test_data_classes = test.get_test_data().get_target()
        truth_data = self.truth_data #truth data to be passed in
        #print("test data before shape test in eval method:",test_data_classes, "truth data before shape check in eval method",truth_data) #input into the eval method
        fp = self.fp("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        self.assertIsNot(fp,np.inf)
        #This print below statement is for what is passed in before the shape chack that returns infinity: 



    def test_boosting_regression(self):
        learner_type = LearnerType("BOOSTING_REGRESSION", {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)
        test_data_classes = test.get_test_data().get_target()
        truth_data = self.truth_data #truth data to be passed in
        #print("test data before shape test in eval method:",test_data_classes, "truth data before shape check in eval method",truth_data) #input into the eval method
        fn = self.fn("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        #This print below statement is for what is passed in before the shape chack that returns infinity: 
        self.assertIsNot(fn,np.inf)
  

  

    def test_random_forest_regression(self):
        learner_type = LearnerType("RANDFOREST_REGRESSION", {'n_estimators': 100, 'criterion':0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)
        test_data_classes = test.get_test_data().get_target()
        truth_data = self.truth_data #truth data to be passed in
        #print("test data before shape test in eval method:",test_data_classes, "truth data before shape check in eval method",truth_data) #input into the eval method
        fp = self.fp("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        self.assertIsNot(fp,np.inf)
        

    def test_my_arg_max(self):
        learner_type = LearnerType("ARGMAX", {'sampling_rate':1})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)
        test_data_classes = test.get_test_data().get_target()
        truth_data = self.truth_data #truth data to be passed in
        #print("test data before shape test in eval method:",test_data_classes, "truth data before shape check in eval method",truth_data) #input into the eval method
        fp = self.fp("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        self.assertIsNot(fp,np.inf)

        #This print below statement is for what is passed in before the shape chack that returns infinity: 

    def test_my_arg_min(self):
        learner_type = LearnerType("ARGMIN", {'sampling_rate':1})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)
        test_data_classes = test.get_test_data().get_target()
        truth_data = self.truth_data #truth data to be passed in
        #print("test data before shape test in eval method:",test_data_classes, "truth data before shape check in eval method",truth_data) #input into the eval method
        fp = self.fp("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        #print(test_data_classes, truth_data)
        self.assertIsNot(fp,np.inf)


    def test_my_depth_estimate(self):
        learner_type = LearnerType("DEPTH_ESTIMATE", {'sampling_rate':1, 'off_nadir_angle':20.0})
        ensemble_type = EnsembleType("SINGLE", None)
        test = learner(self.feature_data, learner_type, ensemble_type)
        self.assertIsInstance(test, data.EmadeDataPair)
        test_data_classes = test.get_test_data().get_target()
        truth_data = self.truth_data #truth data to be passed in
        #print("test data before shape test in eval method:",test_data_classes, "truth data before shape check in eval method",truth_data) #input into the eval method
        fp = self.fp("", test_data_classes, truth_data) # eval method output if output is infinity then next print statment is for debugging that.
        self.assertIsNot(fp,np.inf)


if __name__ == '__main__':
    unittest.main()
