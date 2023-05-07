"""
Programmed by Jason Zutty
"""

import unittest
import numpy as np
np.random.seed(117)
import copy as cp
import random
import inspect
import pickle
import time
import sys
import ast
import gc
import os
import itertools

from GPFramework.EMADE import my_str, my_hash
from GPFramework.general_methods import parse_tree
from GPFramework.standalone_tree_evaluator import load_environment
from GPFramework.sql_connection_orm_base import IndividualStatus, ConnectionSetup
from GPFramework.sql_connection_orm_master import SQLConnectionMaster
from GPFramework.gp_framework_helper import LearnerType
from GPFramework.data import EmadeDataPair
from GPFramework import EMADE as emade
import GPFramework.eval_methods as em
import GPFramework.data as data
import GPFramework.gp_framework_helper as gpFrameworkHelper

from sklearn.metrics import mean_squared_error

from deap import gp
from deap import base
from deap import creator

from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from lxml import etree

import argparse as ap

'''
For some of these, I tested simple inputs and checked manually whether or not the methods outputted the correct values.
'''


class EvalMethodsUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #feature_data = data.load_feature_data_from_file('../../../datasets/unit_test_data/train_data_v2_suit_1-5.csv.gz')
        #self.feature_data = EmadeDataPair(cp.deepcopy(feature_data), cp.deepcopy(feature_data))

        random.seed(100)
        np.random.seed(100)

    def setup(self):
        pass

    def get_individual(self):
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
        print("pre_path: ", pre_path, flush=True)
        print()

        inputFile = os.path.abspath(pre_path + "../../../templates/input_titanic.xml")
        print("inputFile: ", inputFile, flush=True)
        seedingFile = "seeding_test_1"

        schema_doc = etree.parse(os.path.join(pre_path + "../../../templates", 'inputSchema.xsd'))
        schema = etree.XMLSchema(schema_doc)

        doc = etree.parse(inputFile)
        # Raise error if invalid XML
        try:
            schema.assertValid(doc)
        except:
            raise

        tree = ET.parse(inputFile)
        root = tree.getroot()

        database_str_info = root.find('dbConfig')
        server_ip = database_str_info.findtext('server')
        server_username = database_str_info.findtext('username')
        server_password = database_str_info.findtext('password')
        server_database = database_str_info.findtext('database')
        database_str = "mysql://" + server_username + ":" + server_password + "@" + server_ip + "/" + server_database
        print("database_str", database_str, flush=True)
        valid_types = {"int":int, "float":float, "str":str}

        # Initializes the dataset dictionary (Only need this to get the sname of the dataset)
        datasetDict = {}
        datasetList = root.iter('dataset')
        for datasetNum, dataset in enumerate(datasetList):
        # Iterate over each dataset and add to dictionary
            monte_carlo = dataset.iter('trial')
            datasetDict[datasetNum] = {'name': dataset.findtext('name'),
                                   'type': dataset.findtext('type'),
                                   'batchSize': None if dataset.findtext('batchSize') is None else int(dataset.findtext('batchSize')),
                                   'trainFilenames':[],
                                   'testFilenames':[]}
        # The data is already folded for K-fold cross validation. Add these to the trainFilenames
        # and testFilenames lists
        for trial in monte_carlo:
            datasetDict[datasetNum]['trainFilenames'].append(
                trial.findtext('trainFilename'))
            datasetDict[datasetNum]['testFilenames'].append(
                trial.findtext('testFilename'))

        # Initializes the objective dictionary
        objectiveDict = {}
        objectiveList = tree.iter('objective')
        evaluationInfo = tree.find('evaluation')
        #evaluationModule = import_module(evaluationInfo.findtext('module'))
        for objectiveNum, objective in enumerate(objectiveList):
            # Iterate over each objective and add to dictionary

            # convert all the arguments in the xml into their correct types
            o_args = {}
            args_dir = objective.find('args')
            if args_dir is not None:
                for i in args_dir.iterfind('arg'):
                    o_args[i.findtext('name')] = valid_types[i.findtext('type')](i.findtext('value'))

            objectiveDict[objectiveNum] = {'name': objective.findtext('name'), 'weight': float(objective.findtext('weight')),
            'achievable': float(objective.findtext('achievable')), 'goal': float(objective.findtext('goal'))}

            # 'evaluationFunction': getattr(evaluationModule, objective.findtext('evaluationFunction')),
            # 'lower': float(objective.findtext('lower')), 'upper': float(objective.findtext('upper')),
            # 'args': o_args}

        fitness_names = [objectiveDict[objective]['name'] for objective in objectiveDict]
        dataset_names = [datasetDict[dataset]['name'] for dataset in datasetDict]

        # Initialize pset
        pset = gp.PrimitiveSetTyped("MAIN", [EmadeDataPair], EmadeDataPair)
        gpFrameworkHelper.addTerminals(pset)
        ephemeral_methods = gpFrameworkHelper.addPrimitives(pset)

        # Determine whether the problem is a regression (True) or classification (False) problem
        if root.findtext('regression') is None:
            gpFrameworkHelper.set_regression(False)
        elif root.findtext('regression') == 1:
            gpFrameworkHelper.set_regression(True)
        else:
            gpFrameworkHelper.set_regression(False)

        terminals = {}
        primitives = {}
        ephemerals = {}
        for item in pset.mapping:
            if isinstance(pset.mapping[item], gp.Terminal):
                terminals[item] = pset.mapping[item]
            elif isinstance(pset.mapping[item], gp.Primitive):
                primitives[item] = pset.mapping[item]

        names = []
        methods = dir(gp)
        for method in methods:
            pointer = getattr(gp, method)
            if inspect.isclass(pointer) and issubclass(pointer, gp.Ephemeral):
                ephemerals[method] = pointer

        pset_info = {"primitives": primitives,
                      "terminals": terminals,
                      "ephemerals": ephemerals,
                      "ephemeral_methods": ephemeral_methods,
                      "context": pset.context}

        # Compute arrays of weights and thresholds from objective information
        weights = tuple([objectiveDict[objective]['weight'] for objective in objectiveDict])
        goals = tuple([objectiveDict[objective]['goal'] for objective in objectiveDict])
        achievable = tuple([objectiveDict[objective]['achievable'] for objective in objectiveDict])
        LROI = np.array(goals)
        creator.create("FitnessMin", base.Fitness, weights=weights)
        fitness_names = (datasetDict[dataset]['name'] for dataset in datasetDict)
        fitness_attr = dict(zip(fitness_names, itertools.repeat(creator.FitnessMin)))

        arr = np.random.randint(0, 2, 20)
        creator.create("Individual", list, pset=pset, fitness=creator.FitnessMin, age=0, hash_val=None, num_elements=0, test_vec=arr, **fitness_attr)

        fitness_names = [objectiveDict[objective]['name'] for objective in objectiveDict]
        dataset_names = [datasetDict[dataset]['name'] for dataset in datasetDict]


        ConnectionSetup()

        # Initialize database for storing information about each individual
        database = SQLConnectionMaster(connection_str=database_str, reuse=1, fitness_names=fitness_names,
                                                    dataset_names=dataset_names, statistics_dict={}, cache_dict={"timeout":"300"}, is_worker=True)

        print("connected to database")

        individual_list = []
        with open(pre_path + "../../../SeedingFiles/" + seedingFile, 'r') as input:
            for line in input:
                # only parse non-empty lines
                if line.strip():
                    # recursively build a list of a seed
                    # appends primitives, ephemerals, and terminals based on the given tree string
                    my_func, i = parse_tree("Learner(ARG0, LearnerType('RAND_FOREST', {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0}), EnsembleType('SINGLE', None))", pset_info)

                    # construct a deap primitive tree and individual from the given list
                    my_tree = gp.PrimitiveTree(my_func)
                    my_individual = creator.Individual([my_tree, my_tree, my_tree, my_tree])

                    individual_list.append(my_individual)
                    return my_individual

        #return individual_list

        '''
        xml_file = "../../../templates/input_titanic.xml"
        startup_file = "../../../myPickleFile5418.dat"
        train_file_name = "../../../datasets/titanic/train_0.csv.gz"
        test_file_name = "../../../datasets/titanic/test_0.csv.gz"

        # Start off by loading up the environment
        ind, database = load_environment(startup_file, xml_file, train_file=train_file_name, test_file=test_file_name)


        pset = gp.PrimitiveSetTyped("MAIN", [EmadeDataPair], EmadeDataPair)
        gpFrameworkHelper.addTerminals(pset)
        ephemeral_methods = gpFrameworkHelper.addPrimitives(pset)

        terminals = {}
        primitives = {}
        ephemerals = {}
        for item in pset.mapping:
            if isinstance(pset.mapping[item], gp.Terminal):
                terminals[item] = pset.mapping[item]
            elif isinstance(pset.mapping[item], gp.Primitive):
                primitives[item] = pset.mapping[item]

        names = []
        methods = dir(gp)
        for method in methods:
            pointer = getattr(gp, method)
            if inspect.isclass(pointer) and issubclass(pointer, gp.Ephemeral):
                ephemerals[method] = pointer

        pset_info = {"primitives": primitives,
                    "terminals": terminals,
                    "ephemerals": ephemerals,
                    "ephemeral_methods": ephemeral_methods,
                    "context": pset.context}


        strings_to_eval = ["Learner(ARG0, LearnerType('RAND_FOREST', {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0}), EnsembleType('SINGLE', None))",
                           "Learner(ARG0, ModifyLearnerInt(learnerType('RAND_FOREST', {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0}), 99, 0), ensembleType('SINGLE', None))"]

        my_individual = None

        # Evaluate each string individually
        for s in strings_to_eval:
            # Construct a list of nodes from a string
            my_func, i = parse_tree(s, pset_info)

            # Construct a deap primitive tree and individual from the given list
            my_tree = gp.PrimitiveTree(my_func)
            my_individual = ind.Individual([my_tree, my_tree, my_tree, my_tree])

            # Print the seeded individual to terminal
            print(my_str(my_individual))
        return my_individual
    '''


    def test_distance_from_target(self):

    #     arr1 = [[1], [2], [3], [4], [5]]
    #     arr2 = [[2], [3], [6], [8], [5]]

        arr3 = np.array([np.array([[1,2]]), np.array([[4,5],[6,7]]), np.array([[-1,-1]]), np.array([[-1,-1]]), np.array([[4,5],[6,7], [8,9]])])
        arr4 = np.array([np.array([[7,2]]), np.array([[8,4],[6,8]]), np.array([[5,5]]), np.array([[-1,-1]]), np.array([[8,4],[6,8], [10,12]])])

        test = em.distance_from_target(None, arr3, arr4, name=None)
        print("Distance from Target: ", test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)






    def test_mean_dist_from_target(self):
        arr3 = np.array([np.array([[1,2]]), np.array([[4,5],[6,7]]), np.array([[-1,-1]]), np.array([[-1,-1]]), np.array([[4,5],[6,7], [8,9]])])
        arr4 = np.array([np.array([[7,2]]), np.array([[8,4],[6,8]]), np.array([[5,5]]), np.array([[-1,-1]]), np.array([[8,4],[6,8], [10,12]])])

        test = em.mean_dist_from_target(None, arr3, arr4, name=None)
        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_false_positive_centroid(self):
        arr3 = np.array([np.array([[1,2]]), np.array([[4,5],[6,7]]), np.array([[-1,-1]]), np.array([[-1,-1]]), np.array([[4,5],[6,7], [8,9]])])
        arr4 = np.array([np.array([[7,2]]), np.array([[8,4],[6,8]]), np.array([[5,5]]), np.array([[-1,-1]]), np.array([[8,4],[6,8], [10,12]])])

        test = em.false_positive_centroid(None, arr3, arr4, name=None)
        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, int)

    def test_false_negative_centroid(self):
        arr3 = np.array([np.array([[1,2]]), np.array([[4,5],[6,7]]), np.array([[-1,-1]]), np.array([[-1,-1]]), np.array([[4,5],[6,7], [8,9]])])
        arr4 = np.array([np.array([[7,2]]), np.array([[8,4],[6,8]]), np.array([[5,5]]), np.array([[-1,-1]]), np.array([[8,4],[6,8], [10,12]])])

        test = em.false_negative_centroid(None, arr3, arr4, name=None)
        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, int)

    def test_false_negative_centroid2(self):
        arr3 = np.array([np.array([[1,2]]), np.array([[4,5],[6,7]]), np.array([[-1,-1]]), np.array([[-1,-1]]), np.array([[4,5],[6,7], [8,9]])])
        arr4 = np.array([np.array([[7,2]]), np.array([[8,4],[6,8]]), np.array([[5,5]]), np.array([[-1,-1]]), np.array([[8,4],[6,8], [10,12]])])

        test = em.false_negative_centroid2(None, arr3, arr4, name=None)
        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, int)

    def test_false_positive(self):

        arr1 = np.zeros(5)
        arr2 = np.ones(5)


        # Test Data input must be formatted such that each element is iterable
        #data = np.array([[1], [1], [1], [1], [1]])

        test = em.false_positive(None, arr2, arr1, name=None)

        self.assertIsNot(test, np.inf)
        self.assertEqual(test, 5)

    def test_false_negative(self):

        arr1 = np.zeros(5)
        arr2 = np.ones(5)

        # Test Data input must be formatted such that each element is iterable
        #data = np.array([[0], [0], [0], [0], [0]])

        test = em.false_negative(None, arr1, arr2, name=None)

        self.assertIsNot(test, np.inf)
        self.assertEqual(test, 5)

    def test_false_positive_rate(self):

        arr1 = np.zeros(5)
        arr2 = np.ones(5)

        # Test Data input must be formatted such that each element is iterable
        #data = np.array([[1], [1], [1], [1], [1]])

        test = em.false_positive_rate(None, arr2, arr1, name=None)

        self.assertIsNot(test, np.inf)
        self.assertEqual(test, 1)

    def test_false_negative_rate(self):

        arr1 = np.zeros(5)
        arr2 = np.ones(5)

        # Test Data input must be formatted such that each element is iterable
        #data = np.array([[0], [0], [0], [0], [0]])

        test = em.false_negative_rate(None, arr1, arr2, name=None)

        self.assertIsNot(test, np.inf)
        self.assertEqual(test, 1)

    def test_roc_auc(self):

        arr1 = np.random.randint(low=0, high=2, size=10)
        arr2 = np.random.rand(10)

        test = em.roc_auc(None, arr2, arr1, name=None)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_precision_auc(self):

        arr1 = np.random.randint(low=0, high=2, size=10)
        arr2 = np.random.randint(low=0, high=2, size=10)

        test = em.precision_auc(None, arr2, arr1, name=None)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_f1_score_min(self):

        arr1 = np.ones(10)
        arr2 = np.ones(10)

        test = em.f1_score_min(None, arr2, arr1, name=None)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)
        self.assertEqual(test, 0.0)

    def test_objective0EvalFunction(self):

        arr1 = np.random.randint(0, 2, 10)
        arr2 = np.random.randint(0, 2, 10)

        test = em.objective0EvalFunction(None, arr2, arr1, name=None)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

        ans = np.sqrt(mean_squared_error(arr1, arr2))
        self.assertEqual(test, ans)


    def test_objective1EvalFunction(self):

        arr1 = np.zeros(10)
        arr2 = np.ones(10)

        test = em.objective1EvalFunction(None, arr1, arr2, name=None)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)
        self.assertEqual(test, 0.0)


    def test_objective2EvalFunction(self):

        arr1 = np.zeros(10)
        arr2 = np.ones(10)

        test = em.objective2EvalFunction(None, arr2, arr1, name=None)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)
        self.assertEqual(test, 0.0)

    def test_objective3EvalFunction(self):
        print()
        #ind = "Learner(ARG0, LearnerType('RAND_FOREST', {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0}), EnsembleType('SINGLE', None))"

        ind = self.get_individual()
        # ind = base.Toolbox.individual()
        # print()
        # print(str(ind))

        arr1 = np.zeros(10)
        arr2 = np.ones(10)

        test = em.objective3EvalFunction(ind, arr2, arr1, name=None)

        print(test)
        # self.assertIsNot(test, np.inf)
        # self.assertEqual(test, 0.0)

    def test_objective4EvalFunction(self):

        arr1 = np.random.randint(0, 2, 10)
        arr2 = np.random.randint(0, 2, 10)

        test = em.objective4EvalFunction(None, arr1, arr2, name=None)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_objective5EvalFunction(self):

        arr1 = np.zeros(10)
        arr2 = np.ones(10)

        # arr1 = np.random.randint(0, 2, 10)
        # arr2 = np.random.randint(0, 2, 10)

        # Had to change np.mean to np.nanmean in eval_methods.py
        test = em.objective5EvalFunction(None, arr1, arr2, name=None)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)
        self.assertEqual(test, 1.0)

    def test_objective6EvalFunction(self):

        arr1 = np.zeros(10)
        arr2 = np.ones(10)

        # arr1 = np.random.randint(0, 2, 10)
        # arr2 = np.random.randint(0, 2, 10)

        test = em.objective6EvalFunction(None, arr1.astype(float), arr2.astype(float), name=None)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)
        self.assertEqual(test, 1.0)

    def test_objective7EvalFunction(self):

        # arr1 = np.zeros(10)
        # arr2 = np.ones(10)

        arr1 = np.random.randint(0, 2, 10)
        arr2 = np.random.randint(0, 2, 10)

        test = em.objective7EvalFunction(None, arr1.astype(float), arr2.astype(float), name=None)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)
        # self.assertEqual(test, 1.0)

    def test_objective8EvalFunction(self):

        arr1 = np.zeros(10)
        arr2 = np.ones(10)

        # arr1 = np.random.randint(0, 2, 10)
        # arr2 = np.random.randint(0, 2, 10)

        test = em.objective8EvalFunction(None, arr2, arr1, name=None)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)
        self.assertEqual(test, 1.0)

    def test_objective9EvalFunction(self):

        # arr1 = np.zeros(10)
        # arr2 = np.ones(10)

        arr1 = np.random.randint(0, 2, 10)
        arr2 = np.random.randint(0, 2, 10)

        test = em.objective9EvalFunction(None, arr2.astype(float), arr1.astype(float), name=None)

        # print()
        # print("Objective 9 Result: ", test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)
        # self.assertEqual(test, 1.0)

    def test_objective10EvalFunction(self):

        # arr1 = np.zeros(10)
        # arr2 = np.ones(10)

        arr1 = np.random.randint(0, 2, 10)
        arr2 = np.random.randint(0, 2, 10)

        test = em.objective10EvalFunction(None, arr1.astype(float), arr2.astype(float), name=None)

        self.assertIsNot(test, np.inf)
        # self.assertIsInstance(test, float)
        self.assertEqual(test, 0.0)

    def test_objective11EvalFunction(self):

        # arr1 = np.zeros(10)
        # arr2 = np.ones(10)

        arr1 = np.random.randint(0, 2, 10)
        arr2 = np.random.randint(0, 2, 10)

        test = em.objective11EvalFunction(None, arr1.astype(float), arr2.astype(float), name=None)

        self.assertIsNot(test, np.inf)
        # self.assertIsInstance(test, float)
        self.assertEqual(test, 0.0)

    def test_objective12EvalFunction(self):

        # arr1 = np.zeros(10)
        # arr2 = np.ones(10)

        arr1 = np.random.randint(0, 2, 10)
        arr2 = np.random.randint(0, 2, 10)

        test = em.objective12EvalFunction(None, arr1.astype(float), arr2.astype(float), name=None)

        self.assertIsNot(test, np.inf)
        # self.assertIsInstance(test, float)
        self.assertEqual(test, 0.0)

    def test_class0AccuracyEvalFunction(self):

        arr1 = np.random.randint(0, 5, 20)
        arr2 = np.random.randint(0, 5, 20)

        test = em.class0AccuracyEvalFunction(None, arr1, arr2, name=None)

        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)
        # self.assertEqual(test, 1.0)

    def test_class4AccuracyEvalFunction(self):

        arr1 = np.random.randint(0, 5, 20)
        arr2 = np.random.randint(0, 5, 20)

        test = em.class4AccuracyEvalFunction(None, arr1, arr2, name=None)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)
        # self.assertEqual(test, 1.0)

    def test_drinking_error_rate(self):

        arr1 = np.random.randint(0, 4, 20)
        arr2 = np.random.randint(0, 4, 20)

        # print()
        # print(arr1)
        # print(arr2)

        test = em.drinking_error_rate(None, arr1, arr2, name=None)

        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_drinking_false_alarm_rate(self):

        arr1 = np.random.randint(0, 4, 20)
        arr2 = np.random.randint(0, 4, 20)

        # print()
        # print(arr1)
        # print(arr2)

        test = em.drinking_false_alarm_rate(None, arr1, arr2, name=None)

        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_breadth_eval_function(self):
        # ind = "Learner(ARG0, LearnerType('RAND_FOREST', {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0}), EnsembleType('SINGLE', None))"
        ind = self.get_individual()
        arr1 = np.random.randint(0, 10, 50)
        arr2 = np.random.randint(0, 10, 50)
        test = em.breadth_eval_function(ind, arr1, arr2, name=None)
        print()
        print("Breadth: ", test)

       # test = em.breadth_eval_function(ind, self.feature_data.get_test_data, self.feature_data.get_train_data, name=None)
        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_depth_breadth_eval_function(self):
        ind = self.get_individual()
        arr1 = np.random.randint(0, 10, 50)
        arr2 = np.random.randint(0, 10, 50)
        test = em.depth_breadth_eval_function(ind, arr1, arr2, name=None)
        print()
        print("Depth Breadth: ", test)
        #test = em.depth_breadth_eval_function(individual, self.feature_data.get_test_data, self.feature_data.get_train_data, name=None)
        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_num_elements_eval_function(self):
        #from pprint import pprint
        ind = self.get_individual()
        #pprint(vars(ind))
        arr1 = np.ones(10)
        arr2 = np.zeros(10)
        test = em.num_elements_eval_function(ind, arr1, arr2, name="num_elements")
        print()
        print("Num Elements: ", test)
        # test = em.num_elements_eval_function(individual, self.feature_data.get_test_data, self.feature_data.get_train_data, name=None)
        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, np.int64)

    def test_num_elements_eval_function_capped(self):
        ind = self.get_individual()
        arr1 = np.random.randint(0, 10, 50)
        arr2 = np.random.randint(0, 10, 50)
        test = em.num_elements_eval_function_capped(ind, arr1, arr2, name=None)
        print()
        print("Num Capped: ", test)
        #test = em.num_elements_eval_function_capped(individual, self.feature_data.get_test_data, self.feature_data.get_train_data, name=None)
        self.assertIsNot(test, np.inf)
        # self.assertIsInstance(test, int)

    def test_shaking_error_rate(self):

        arr1 = np.random.randint(0, 10, 50)
        arr2 = np.random.randint(0, 10, 50)

        # print()
        # print(arr1)
        # print(arr2)

        test = em.shaking_error_rate(None, arr1, arr2, name=None)

        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_shaking_false_alarm_rate(self):

        arr1 = np.random.randint(0, 10, 50)
        arr2 = np.random.randint(0, 10, 50)

        # print()
        # print(arr1)
        # print(arr2)

        test = em.shaking_false_alarm_rate(None, arr1, arr2, name=None)

        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_scratching_error_rate(self):
        """
        TODO: Fix this error.

        ======================================================================
        ERROR: test_scratching_error_rate (__main__.EvalMethodsUnitTest)
        ----------------------------------------------------------------------
        Traceback (most recent call last):
          File "eval_methods_unit_test.py", line 745, in test_scratching_error_rate
            test = em.scratching_error_rate(ind, arr1, arr2, name="test_vec")
          File "/home/jrick6/repos/emade/src/GPFramework/eval_methods.py", line 1001, in scratching_error_rate
            test_data = np.array([elem[0] for elem in test_data])
          File "/home/jrick6/repos/emade/src/GPFramework/eval_methods.py", line 1001, in <listcomp>
            test_data = np.array([elem[0] for elem in test_data])
        IndexError: invalid index to scalar variable.

        """
        ind = self.get_individual()
        arr1 = np.random.randint(0, 2, 20)
        arr2 = np.random.randint(0, 2, 20)

    #     print()
    #     print(arr1)
    #     print(arr2)

        test = em.scratching_error_rate(ind, arr1, arr2, name="test_vec")
        print()
        print("Scratching Error: ", test)
        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_scratching_false_alarm_rate(self):
        """
        TODO: Fix this error.

        ======================================================================
        ERROR: test_scratching_false_alarm_rate (__main__.EvalMethodsUnitTest)
        ----------------------------------------------------------------------
        Traceback (most recent call last):
          File "eval_methods_unit_test.py", line 762, in test_scratching_false_alarm_rate
            test = em.scratching_false_alarm_rate(ind, arr1, arr2, name="test_vec")
          File "/home/jrick6/repos/emade/src/GPFramework/eval_methods.py", line 1041, in scratching_false_alarm_rate
            test_data = np.array([elem[0] for elem in test_data])
          File "/home/jrick6/repos/emade/src/GPFramework/eval_methods.py", line 1041, in <listcomp>
            test_data = np.array([elem[0] for elem in test_data])
        IndexError: invalid index to scalar variable.

        """
        ind = self.get_individual()
        arr1 = np.random.randint(0, 2, 20)
        arr2 = np.random.randint(0, 2, 20)

    #     print()
    #     print(arr1)
    #     print(arr2)

        test = em.scratching_false_alarm_rate(ind, arr1, arr2, name="test_vec")

        print("Scratching False Alarm: ", test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_get_over_predicted_inds(self):

        arr1 = np.random.randint(0, 4, 20)
        arr2 = np.random.randint(0, 4, 20)

        # print()
        # print(arr1)
        # print(arr2)

        test = em.get_over_predicted_inds(arr1, arr2, name=None, tolerance=1.5)

        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, np.ndarray)

    def test_get_under_predicted_inds(self):

        arr1 = np.random.randint(0, 4, 20)
        arr2 = np.random.randint(0, 4, 20)

        # print()
        # print(arr1)
        # print(arr2)

        test = em.get_under_predicted_inds(arr1, arr2, name=None, tolerance=1.5)

        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, np.ndarray)

    def test_count_over_predictions(self):

        arr1 = np.random.randint(0, 4, 20)
        arr2 = np.random.randint(0, 4, 20)

        # print()
        # print(arr1)
        # print(arr2)

        test = em.count_over_predictions(None, arr1, arr2, name=None, tolerance=1.5)

        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_count_under_predictions(self):

        arr1 = np.random.randint(0, 4, 20)
        arr2 = np.random.randint(0, 4, 20)

        # print()
        # print(arr1)
        # print(arr2)

        test = em.count_under_predictions(None, arr1, arr2, name=None, tolerance=1.5)

        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_overall_standard_deviation(self):

        arr1 = np.random.randint(0, 4, 20)
        arr2 = np.random.randint(0, 4, 20)

        test = em.overall_standard_deviation(None, arr1, arr2, name=None)

        # print()
        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test,float)

    def test_standard_deviation_over(self):

        arr1 = np.random.randint(0, 4, 20)
        arr2 = np.random.randint(0, 4, 20)

        test = em.standard_deviation_over(None, arr1, arr2, name=None, tolerance=1.0)

        # print()
        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test,float)

    def test_standard_deviation_under(self):

        arr1 = np.random.randint(0, 4, 20)
        arr2 = np.random.randint(0, 4, 20)

        test = em.standard_deviation_under(None, arr1, arr2, name=None, tolerance=1.0)

        # print()
        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test,float)

    def test_average_percent_error(self):

        arr1 = np.random.randint(1, 5, 20)
        arr2 = np.random.randint(1, 5, 20)

        test = em.average_percent_error(None, arr1, arr2, name=None)

        # print()
        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_average_percent_error_over(self):

        arr1 = np.random.randint(1, 5, 20)
        arr2 = np.random.randint(1, 5, 20)

        test = em.average_precent_error_over(None, arr1, arr2, name=None, tolerance=1.0)

        # print()
        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_average_percent_error_under(self):

        arr1 = np.random.randint(1, 5, 20)
        arr2 = np.random.randint(1, 5, 20)

        test = em.average_precent_error_under(None, arr1, arr2, name=None, tolerance=1.0)

        # print()
        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_max_over_prediction_error(self):

        arr1 = np.random.randint(1, 5, 20)
        arr2 = np.random.randint(1, 5, 20)

        test = em.max_over_prediction_error(None, arr1, arr2, name=None, tolerance=1.0)

        # print()
        # print(test)
        # print(type(test))

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, np.int64)

    def test_max_under_prediction_error(self):

        arr1 = np.random.randint(1, 5, 20)
        arr2 = np.random.randint(1, 5, 20)

        test = em.max_under_prediction_error(None, arr1, arr2, name=None, tolerance=1.0)

        # print()
        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, np.int64)

    def test_continuous_mse(self):

        arr1 = np.random.randint(1, 5, 20)
        arr2 = np.random.randint(1, 5, 20)

        test = em.continuous_mse(None, arr1, arr2, name=None)

        # print()
        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_continuous_bias(self):

        arr1 = np.random.randint(1, 5, 20)
        arr2 = np.random.randint(1, 5, 20)

        test = em.continuous_bias(None, arr1, arr2, name=None)

        # print()
        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_continuous_var(self):

        arr1 = np.random.randint(1, 5, 20)
        arr2 = np.random.randint(1, 5, 20)

        test = em.continuous_var(None, arr1, arr2, name=None)

        # print()
        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_cluster_partition_distance(self):

        arr1 = np.random.randint(1, 5, 20)
        arr2 = np.random.randint(1, 5, 20)

        test = em.cluster_partition_distance(None, [arr1], [arr2], name=None)

        # print()
        # print(test)

        self.assertIsNot(test, np.inf)
        self.assertIsInstance(test, float)

    def test_cluster_error1(self):

        arr1 = np.random.randint(1, 5, 20)
        arr2 = np.random.randint(1, 5, 20)

        test = em.cluster_error1(None, [arr1], [arr2], name=None)

        # print()
        # print(test)

        self.assertIsNot(test, np.inf)
        # self.assertIsInstance(test, float)

    def test_cluster_error2(self):

        arr1 = np.random.randint(1, 5, 20)
        arr2 = np.random.randint(1, 5, 20)

        test = em.cluster_error2(None, [arr1], [arr2], name=None)

        # print()
        # print(test)

        self.assertIsNot(test, np.inf)
        # self.assertIsInstance(test, float)



if __name__ == '__main__':
    unittest.main()
