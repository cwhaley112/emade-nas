import GPFramework.data as data

import numpy as np
import matplotlib.pyplot as plt
import unittest
import scipy
import GPFramework.learner_methods as lm
import GPFramework.gp_framework_helper as gpf 
import GPFramework.eval_methods as ev
import GPFramework.neural_network_methods as nnm
import GPFramework.eval_methods as em
import os

import logging
import sys


class Test(unittest.TestCase):
    def setUp(self):
        #image_data = data.load_images_from_file('datasets/image/small/train9/')
        # image_data = data.load_pickle_from_file(os.path.join(os.path.dirname(__file__), '../../../datasets/atr_eo/small_400/train_eo0.npz'))
        self.size = 12
        def reduce_instances(emadeDataTuple, size=self.size):
            emadeData, none = emadeDataTuple
            print(len(emadeData.get_instances()))
            subset = emadeData.get_instances()[:size]
            emadeData.set_instances(subset)
            return emadeData, none
        image_data = data.load_pickle_from_file(os.path.join(os.path.dirname(__file__), '../../../datasets/chest_xrays/chest_val_14.npz'))
        print('worked!')
        self.image_data = data.EmadeDataPair(reduce_instances(image_data), reduce_instances(image_data))
        self.image_data.set_truth_data()
        print(self.image_data.get_truth_data()) 
        self.image_data.set_datatype('imagedata')
        self.image_data.set_multilabel(True)

    def test_nnlearner(self):

        llist = nnm.InputLayer()

        #llist = nnm.ConcatenateLayer(llist, llist2)
        #llist = nnm.ELUActivationLayer(llist)
        #llist = nnm.DenseLayer(15,  nnm.WeightInitializer.RANDOMUNIFORM,10, llist)
        #llist = nnm.LeakyReLULayer(0.01, llist)
        #llist = nnm.DropoutLayer(0.1, llist)
        #llist = nnm.DenseLayer(15,  nnm.WeightInitializer.RANDOMUNIFORM,3,llist)
        #llist = nnm.LeakyReLULayer(0.01, llist)
        ARG0 = self.image_data
        # llist = nnm.Conv2DLayer(256, nnm.Activation.RELU, 3, 1, False, 10, llist)
        # llist = nnm.BatchNormalizationLayer(llist)
        llist = nnm.ResBlock(64, llist)
        llist = nnm.ResBlock(128, llist)
        llist = nnm.ResBlock(256, llist)
        llist = nnm.ResBlock(512, llist)
        llist = nnm.OutputLayer(llist)
        result = nnm.NNLearner(ARG0, llist, 100, nnm.Optimizer.ADAM)
        self.assertIsInstance(result, data.EmadeDataPair)


        print(self.image_data.get_train_data().get_instances()[0].get_stream().get_data().shape)
        a = np.array([instance.get_stream().get_data() for instance in self.image_data.get_train_data().get_instances()])
        print(a.shape)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
    print("really done.")
