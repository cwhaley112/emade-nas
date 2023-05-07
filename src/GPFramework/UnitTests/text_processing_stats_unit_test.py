import numpy as np
import matplotlib.pyplot as plt
import unittest
import scipy
import GPFramework.eval_methods as em
import GPFramework.text_processing_stats as tps

import logging
import sys
import os

import nltk
from textblob import TextBlob


logging.basicConfig(level=logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
#stream_handler.setFormatter(logging.Formatter(fmt=logformat, datefmt=datefmt))

logger = logging.getLogger("app")
logger.addHandler(stream_handler)


class Test(unittest.TestCase):
    """
    This class contains the unit tests for text processing methods
    """

    def setUp(self):
        self.eps = 1e-8
        self.gens = 10
        self.samples = 50
        self.shape = (self.samples, self.gens)
        pass

    def tearDown(self):
        pass

    def test_mean_variance(self):
        """
        Item #2
        """
        print(f"test mean variance: gens {self.gens}")
        #test data
        aucs = np.random.rand(self.samples, self.gens) * 100
        #self.samples = 3
        #aucs = np.random.rand(3, 3) * 100

        #expected vs actual
        expected_mean = np.sum(aucs, axis=0) / self.samples
#        assert(False), f"expect shape {expected_mean.shape}"
        expected_var = np.sum(np.power(aucs, 2), axis=0) / self.samples 
        expected_var -= np.power(expected_mean, 2)
        actual_mean, actual_var = tps.compute_mean_variance(aucs)

        #assertion checks
        self.assertIsInstance(actual_mean, np.ndarray)
        self.assertIsInstance(actual_var, np.ndarray)
        assert(np.array_equal(actual_mean, expected_mean)), f"actual{actual_mean}\nexpect{expected_mean.shape}\n{expected_mean}"
        diff = np.abs(actual_var - expected_var)
        self.assertEqual(np.count_nonzero(diff < self.eps), self.gens)
        #assert(np.count_nonzero(diff < self.eps) == self.gens), f"{np.count_nonzero(diff < self.eps)}"
        #assert(np.array_equal(actual_var, expected_var)), f"actual{actual_var[:]}\nexpect{expected_var}"
        tps.plot_boxplot(aucs, show_mean=True)
        tps.plot_boxplot(aucs, show_mean=False)

    def test_pvalues(self):
        """
        Item #2
        """
        print(f"test pvalues: gens {self.gens}")
        #test data
        mu, sigma = 100, 10
        variable = np.random.gamma(1.0, 2.0, self.shape)
        control = np.random.normal(mu, sigma, self.shape)

        var_mu, var_sigma = tps.compute_mean_variance(variable)
        ctrl_mu, ctrl_sigma = tps.compute_mean_variance(control)
        pvalues = tps.compute_pvalues(var_mu, var_sigma, self.samples, ctrl_mu, ctrl_sigma, self.samples)

        print(f"{pvalues.shape}")
        self.assertEqual(pvalues.shape, (self.gens, ))
        self.assertEqual(np.count_nonzero(pvalues < 0.05), self.gens)
        print(np.count_nonzero(pvalues < 0.05) / self.gens)

        tps.plot_pvalues(pvalues)


        

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
    print("really done.")
