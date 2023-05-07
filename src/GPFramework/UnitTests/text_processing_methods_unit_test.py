import GPFramework.data as data
import GPFramework.text_processing_methods as tp
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
from textblob import TextBlob
import logging
import sys

import nltk

#import GPFramework.sparse_nlp_learners as snl
# text_data = data.load_text_data_from_file('data/train1.csv.gz')
# text_data = data.EmadeDataPair(text_data, text_data)
# #print(text_data.get_train_data().get_instances()[0].get_features().get_data()[0, :])

# logging.basicConfig(level=logging.INFO)
# stream_handler = logging.StreamHandler(sys.stdout)
# #stream_handler.setFormatter(logging.Formatter(fmt=logformat, datefmt=datefmt))

# logger = logging.getLogger("app")
# logger.addHandler(stream_handler)


class Test(unittest.TestCase):
    """
    This class contains the unit tests for text processing methods
    """

    def setUp(self):
        """
        Create EmadeDataPair object 
        """
        
            # Copy the truth data in to its own location

            # Clear out the truth data from the test data
        # [[instance.set_target(np.array([float('NaN')]))
        #                         for instance in test_data[0].get_instances()]
        #                         for test_data in test_data_array]

            # Stores DataPair object
        
        text_data = data.load_text_data_from_file(os.path.join(os.path.dirname(__file__), '../../../datasets/unit_test_data/movie_reviews_data.csv.gz'))
        #text_data = data.load_text_data_for_multilabel('src/UnitTests/data/test1.csv.gz')
        
        #feature_data =  data.load_feature_data_from_file('data/train_data_v2_suit_1-5.csv.gz')
        
        self.truth_data_array = np.array([instance.get_target()[0] for instance in text_data[0].get_instances()])
        print(self.truth_data_array)
        # self.feature_data = data.EmadeDataPair(feature_data, feature_data)
        # self.text_data = data.EmadeDataPair(text_data, text_data)

        
        #print(self.text_data.get_train_data().get_numpy()[23:25])
        #print("DIVIDER!!")

    # def tearDown(self):
    #     del self.text_data
        
    # def test_cv(self):
        # test = tp.count_vectorizer(self.text_data, True, 0, 0, 33)
        # self.assertIsInstance(test, data.EmadeDataPair)
        # #print(test.get_train_data().get_numpy().shape)
        # #print(np.array(test.get_train_data().get_instances()[0].get_target()))
        # #print(type(test.get_train_data().get_instances()[923].get_features().get_data()))
        # print(test.get_train_data().get_instances()[0].get_features().get_data())
        # self.assertEqual(test.get_train_data().get_numpy()[0].shape[0], 1)
        # self.assertIsInstance(test.get_train_data().get_numpy()[0], scipy.sparse.csr_matrix)
        # #self.assertEqual(self.text_data.get_train_data().get_numpy().shape[0], test.get_train_data().get_numpy().shape[0]) TAKES WAYY TOO LONG SMH
        # #print(self.text_data.get_train_data().get_instances()[1].get_features().get_data()[0, :])
        # #print(test.get_train_data().get_numpy()[0].shape)
        # #test = methods.logistic_regression_scikit(test)
        # lm.makeFeatureFromClass(test, name = 'LogR')

    def test_tv(self):
        test = tp.tfidf_vectorizer(self.text_data, False, 43,343,4357)
        test = lm.learner(test, gpf.LearnerType("LOGR",  {'penalty':0, 'C':1.0}, "SINGLE", None))
        print(test.get_train_data().get_target())
        num = em.false_positive(test, self.truth_data_array, test.get_train_data().get_target())
        #print(type(test.get_test_data().get_numpy()))
        #print(test.get_test_data().get_instances())
        self.assertIsInstance(test, data.EmadeDataPair)
        
        #self.assertIsInstance(test.get_train_data().get_numpy()[0], scipy.sparse.csr_matrix)

    def test_nnm(self):
        print("------------------------------------------------------------slkdjf------")
        #first = self.text_data.get_train_data().get_target()
        #print(type(first))
        test = tp.tokenizer(self.text_data, 100)
        nn = nnm.lstm(test, 0,0,0,200,0,20)
        second = nn.get_test_data().get_target()
        print(second)
        #print(second[0])
        #print(self.truth_data_array[0])
        #num = em.false_positive(nn, self.truth_data_array[0], second)
        #print(num)

    

       

    def tearDown(self):
        del self.text_data


    def test_sum(self):
        #print(self.summary_data.get_nump)
        test_data = self.summary_data.get_test_data()

        print('truth data', np.array([instance.get_target()
                                for instance in test_data.get_instances()]))
        truth_data = np.array([instance.get_target()
                                for instance in test_data.get_instances()])
    
        test2 = tp.tfidf_vectorizer_mod(self.summary_data, True, 0,1,4)
        #print(test2)
        print('done2')
        #test = tp.tfisf(self.summary_data, False, 343, 4357, 4)
        #
        #print(test2.get_train_data().get_numpy().shape)
        import time
        start = time.time()
        #print(scipy.sparse.vstack(list(test2.get_train_data().get_numpy()[:,0].reshape(-1))).shape)
        print('time', time.time() - start)
        # .vstack([instance.get_features().get_data()
        #                  for instance in  test2.get_train_data().get_numpy()[:,0]])
        test = lm.learner_mod(test2, gpf.LearnerType('LINSVC', {'penalty':0, 'C':1.0}, 'SINGLE', None))
        
        predicted_data = np.array([instance.get_target()
                                for instance in test.get_test_data().get_instances()])
        print('done', truth_data.shape, predicted_data.shape)
        print(em.false_positive('', predicted_data, truth_data))


    # def test_stemmatizer(self):
    #     log = logging.getLogger("TestLog")
    #     for learnerName, learnerParams in zip(self.names, self.params):
    #         log.debug(f"learnerName: {learnerName}")
    #         log.debug(f"learnerParams: {learnerParams}")
    #         for i in range(3):
    #             log.debug(i)
    #             test = tp.stemmatizer(self.text_data, i, 0)
    #             test = tp.word2vec(test, 0, 100, 100, 0, False)
    #             test = lm.learner(test, gpf.LearnerType(learnerName, learnerParams, 'SINGLE', None))
    #             self.assertIsInstance(test, data.EmadeDataPair)

    # def test_sentiment_learner(self):
    #     for learnerName, learnerParams in zip(self.names, self.params):
    #         logger.debug(f"learnerName: {learnerName}")
    #         logger.debug(f"learnerParams: {learnerParams}")
    #         sent = tp.sentiment(self.text_data, sentence_vec=True)
    #         test = lm.learner(sent, gpf.LearnerType(learnerName, learnerParams, 'SINGLE', None))
    #         self.assertIsInstance(test, data.EmadeDataPair)

    #         word = tp.sentiment(self.text_data, sentence_vec=False)
    #         test = lm.learner(word, gpf.LearnerType(learnerName, learnerParams, 'SINGLE', None))
    #         self.assertIsInstance(test, data.EmadeDataPair)


   

    def test_tv(self):
      test = tp.tfidf_vectorizer(self.text_data, False, 43,343,4357)
      test = lm.learner(test, gpf.LearnerType("LOGR",  {'penalty':0, 'C':1.0}, "SINGLE", None))
      print(test.get_train_data().get_target())
      #print(self.truth_data_array.shape, test.get_test_data())
      num = em.false_positive(test, self.truth_data_array, test.get_train_data().get_target())
      #print(type(test.get_test_data().get_numpy()))
      #print(test.get_test_data().get_instances())
      self.assertIsInstance(test, data.EmadeDataPair)
        
    
    def num_named_entities(self):
        #print(self.summary_data.get_nump)
        test_data = self.summary_data.get_test_data()

        print('truth data', np.array([instance.get_target()
                                for instance in test_data.get_instances()]))
        truth_data = np.array([instance.get_target()
                                for instance in test_data.get_instances()])
    
        test2 = tp.num_named_entities(self.summary_data)
        #print(test2)
        print('done2')
        #test = tp.tfisf(self.summary_data, False, 343, 4357, 4)
        #
        #print(test2.get_train_data().get_numpy().shape)
        import time
        start = time.time()
        #print(scipy.sparse.vstack(list(test2.get_train_data().get_numpy()[:,0].reshape(-1))).shape)
        print('time', time.time() - start)
        # .vstack([instance.get_features().get_data()
        #                  for instance in  test2.get_train_data().get_numpy()[:,0]])
        test = lm.learner_mod(test2, gpf.LearnerType('LINSVC', {'penalty':0, 'C':1.0}, 'SINGLE', None))
        
        predicted_data = np.array([instance.get_target()
                                for instance in test.get_test_data().get_instances()])
        print('done', truth_data.shape, predicted_data.shape)
        print(em.false_positive('', predicted_data, truth_data))
        
    def test_tfisf(self):
        import time
        from nltk.corpus import stopwords

        # get stopwords from nltk
        stop_words = stopwords.words('english')

        test_data = self.summary_data.get_test_data()
        print('truth data', np.array([instance.get_target()
                                for instance in test_data.get_instances()]))
        truth_data = np.array([instance.get_target()
                                for instance in test_data.get_instances()])
        test2 = tp.tfisf(self.summary_data, 1, 2, 5, 1)
        print('done2')

        start = time.time()
        print('time', time.time() - start)
        test = lm.learner_mod(test2, gpf.LearnerType('LINSVC', {'penalty':0, 'C':1.0}, 'SINGLE', None))
        
        predicted_data = np.array([instance.get_target()
                                for instance in test.get_test_data().get_instances()])
        print('done', truth_data.shape, predicted_data.shape)
        print(em.false_positive('', predicted_data, truth_data))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
    print("really done.")
