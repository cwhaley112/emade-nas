import unittest
import sys
import numpy as np
import GPFramework.neural_network_methods as nnm
import GPFramework.eval_methods as em
from GPFramework.data import EmadeDataPair, EmadeData, EmadeDataInstance
from GPFramework.data import FeatureData, StreamData, TextData, EmadeDataObject
from GPFramework.data import load_text_data_from_file, load_pickle_from_file, load_feature_pickle_from_file
import os
from math import isnan

class Test(unittest.TestCase):
    """
    This class contains the unit tests for text processing methods
    """

    def setUp(self):
        """
        Create EmadeDataPair object 
        """
        #eval method
        self.acc = em.accuracy_score
        self.fp = em.false_positive
        self.fn = em.false_negative
        self.size = 100
        def reduce_instances(emadeDataTuple, size=self.size):
            emadeData, none = emadeDataTuple
            print(len(emadeData.get_instances()))
            subset = emadeData.get_instances()[:size]
            emadeData.set_instances(subset)
            return emadeData, none

        def check_cleared(emadeData):
            for inst in emadeData.get_instances():
                target = inst.get_target()
                self.assertIsInstance(target, np.ndarray)
                assert(isnan(target[0]))

                #array_equal compares NaN incorrectly
                #assert(np.array_equal(target, clear_val)), f"target: {target}, cleared: {clear_val}"


        # Read in full splits
        #train = load_text_data_from_file("datasets/toxicity/train.csv.gz")
        # train = load_text_data_from_file(os.path.join(os.path.dirname(__file__), "../../../datasets/wikidetox/train.csv.gz"))
        # test = load_text_data_from_file(os.path.join(os.path.dirname(__file__), "../../../datasets/wikidetox/test.csv.gz"))
        print('zeroth')
        train = load_feature_pickle_from_file(os.path.join(os.path.dirname(__file__), "../../../datasets/rec_data/train_rec_reg.npz"))
        test = load_feature_pickle_from_file(os.path.join(os.path.dirname(__file__), "../../../datasets/rec_data/test_rec_reg.npz"))

        print("first", train[0].get_target().shape)
        # Reduce the number of instances for quick test. currently 2 epochs of 67K instances.
        train = reduce_instances(train)
        test = reduce_instances(test)
        #self.assertEqual(len(train[0].get_instances()), self.size)
        #self.assertEqual(len(test[0].get_instances()), self.size)
        #print("second", train[0].get_target())
        train_data_array = [train]
        test_data_array = [test]
        #print(test[0].get_numpy())

        # Use the same process as EMADE.py
       


        print("before", train[0].get_target().shape)
        # for test_data in test_data_array:
        #     for instance in test_data[0].get_instances():
        #         print(instance.get_target())
        # [[instance.set_target(np.array([instance.get_target()]))
        #         for instance in train_data[0].get_instances()]
        #         for train_data in train_data_array]
        # [[instance.set_target(np.array([instance.get_target()]))
        #                 for instance in test_data[0].get_instances()]
        #                 for test_data in test_data_array]

        # print("after", train[0].get_target().shape)  
        # for test_data in test_data_array:
        #     for instance in test_data[0].get_instances():
        #         print(instance.get_target())              
        # Copy the truth data in to its own location


        self.truth_data_array = [test_data_array[0][0].get_target()]
        # Clear out the truth data from the test data
        # print(self.truth_data_array[0])

        # Create DataPair
        data_pair_list = [EmadeDataPair(train_data, test_data)
            for train_data, test_data in zip(train_data_array, test_data_array)]
        self.assertEqual(len(data_pair_list), 1)

        #store in datapair
        self.data = data_pair_list[0]
        self.data.set_truth_data()
        self.data.set_datatype('recdata') 
        self.data.set_regression(True)
        self.data.get_test_data().set_target(np.full(self.data.get_test_data().get_target().shape, np.nan)) # clear out test data after setting truth data in data_pair! 
        #[instance.set_target(np.array([float('NaN')]))
        #                    for instance in self.data.get_test_data().get_instances()]
        
        check_cleared(self.data.get_test_data())

    # def tearDown(self):
    #   del self.toxic_train_data
    #   del self.toxic_test_data
    #   del self.num_labels
    #   del self.data

    # def test_data(self):
        ## a method that looks at the organization of EmadeData.
        ##
        # self.assertEqual(self.data.get_train_data(), self.toxic_train_data[0])


        # train_data = self.data.get_train_data()
        # target = train_data.get_target()
        # sh = target.shape

        # train_inst = train_data.get_instances()
        # ti_sh = len(train_inst) #stored as a list in the back, so returned as a list.
        #                       #when we get target, it hstacks the np.arrays 
        #                       #that are returned from inst.get_target() for all inst in self._inst.

        # single_inst = train_inst[0].get_target()
        # si_sh = single_inst.shape

        # using assert False to force unittest to output to the console.
        # this was just faster than figuring out how logging module works.
        # #assert(False), f"get_target(): {sh}\ntrain_inst.shape: {ti_sh} {type(train_inst)}\n single_inst.shape: {si_sh}"
        # #assert(False), f"{type(self.toxic_train_data)}, {type(self.data)}"

    def check_filled(self, emadeInstances):
            for inst in emadeInstances:
                #convert to np.ndarray cuz that's what EMADE.handleWorker does.
                target = np.array(inst.get_target()) 
                self.assertIsInstance(target, np.ndarray)
                assert(len(target) == 1), f"target: {target}"
 
    def test_single_concat(self):
        # forward pass of our individual
        #ARG0 = self.data
        #llist = nnm.InputLayer()
        #llist = nnm.EmbeddingLayer(100, ARG0, nnm.WeightInitializer.RANDOMUNIFORM, llist)
        #llist = nnm.GRULayer(32,  nnm.WeightInitializer.RANDOMUNIFORM,10, True, llist)
        #llist = nnm.ELUActivationLayer(llist)
        #llist = nnm.DenseLayer(15,  nnm.WeightInitializer.RANDOMUNIFORM,10, llist)
        #llist = nnm.LeakyReLULayer(0.01, llist)
        #llist = nnm.DropoutLayer(0.1, llist)
        #llist = nnm.DenseLayer(15,  nnm.WeightInitializer.RANDOMUNIFORM,3,llist)
        #llist = nnm.LeakyReLULayer(0.01, llist)
        #llist = nnm.OutputLayer(ARG0, llist)
        #result = nnm.NNLearner(ARG0, llist, 100, nnm.Optimizer.ADAM)
        #self.assertIsInstance(result, EmadeDataPair)


        ARG0 = self.data
        llist = nnm.InputLayer(ARG0)
        llist = nnm.EmbeddingLayer(100, ARG0, nnm.WeightInitializer.RANDOMUNIFORM, llist)
        #llist = nnm.DenseLayer(10,nnm.Activation.RELU, 3,llist)
        llist1 = nnm.InputLayer(ARG0)
        llist1 = nnm.EmbeddingLayer(100, ARG0, nnm.WeightInitializer.RANDOMUNIFORM, llist1)
        #llist1 = nnm.DenseLayer(10,nnm.Activation.RELU, 3,llist1)
        llist3 = nnm.ConcatenateLayer(llist, llist1)
        llist5 = nnm.OutputLayer(ARG0, llist3)
        


        #print(llist5.mylist)

        #llist = nnm.InputLayer()
        #llist = nnm.EmbeddingLayer(100, ARG0, nnm.WeightInitializer.RANDOMUNIFORM, llist)
        #llist = nnm.Conv1DLayer(256, 3,1, nnm.WeightInitializer.RANDOMUNIFORM, 10, llist)
        #llist = nnm.ELUActivationLayer(llist)
        #llist = nnm.MaxPoolingLayer1D(4,1, llist)
        #llist = nnm.FlattenLayer(llist)
        ##llist = nnm.GlobalMaxPoolingLayer1D(llist)
        #llist = nnm.DenseLayer(15,  nnm.WeightInitializer.RANDOMUNIFORM,10, llist)
        #llist = nnm.LeakyReLULayer(0.01, llist)
        #llist = nnm.DropoutLayer(0.1, llist)
        #llist = nnm.DenseLayer(15,  nnm.WeightInitializer.RANDOMUNIFORM,3,llist)
        #llist = nnm.LeakyReLULayer(0.01, llist)
        #llist = nnm.OutputLayer(ARG0, llist)
        result = nnm.NNLearner(ARG0, llist5, 100, nnm.Optimizer.ADAM)

        #make sure that the data pair has a model param count
        #assert(result.get_num_params() == 39354), f"{result.get_num_params()} Num params should be 2" 

        # make sure that the test targets are filled in.
        pred_target = result.get_test_data().get_instances()
        self.assertIsInstance(pred_target, list)
        self.assertEqual(len(pred_target), self.size)
        self.check_filled(pred_target)

        # prepare arguments to evaluation method
        test_data_classes = result.get_test_data().get_target()
        truth_data_array = self.truth_data_array[0]
        individual = "NNLearner(ARG0, OutputLayer(ARG0, DenseLayer(15, 'relu', LSTMLayer(32,EmbeddingLayer(100, ARG0)))))"

        # call eval method
        # sidenote: whoever is testing eval methods should do it here.
        output_string = "test vs truth\n"
        for a, b in zip(test_data_classes, truth_data_array):
            output_string += f"{a} vs {b}" + "\n"
        obj1 = self.fn(individual, test_data_classes, truth_data_array)
        print(test_data_classes)
        print(truth_data_array)
        obj2 = self.fp(individual, test_data_classes, truth_data_array)
        obj3 = self.acc(individual, test_data_classes, truth_data_array)
        print(obj1, obj2, obj3)
        # force output to console with assert(False) when debugging.
        # otherwise, comment out the line below.
        assert(obj1 == 2), f"{output_string} false_negative should be 2"
        assert(obj2 == 0), f"{output_string} false_positive should be 0"
        assert(obj3 ==  0.4), f"{output_string} accuracy should be 0.4"
        #assert(False), f"{output_string}\nobjectives[fn, fp, acc]: {[obj1, obj2, obj3]}"
 
if __name__ == '__main__':
    # logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
    print("really done.")
