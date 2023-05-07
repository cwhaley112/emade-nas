import unittest
import numpy as np
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
#from GPFramework.database_tree_evaluator import load_environment, evaluation_wrapper
from GPFramework.sql_connection_orm_base import IndividualStatus, ConnectionSetup
from GPFramework.sql_connection_orm_master import SQLConnectionMaster
from GPFramework.gp_framework_helper import LearnerType
from GPFramework.data import EmadeDataPair
from GPFramework import EMADE as emade
from GPFramework import eval_methods
import GPFramework.data as data
import GPFramework.gp_framework_helper as gpFrameworkHelper
from GPFramework.general_methods import str2bool

from sklearn.metrics import mean_squared_error

from deap import gp
from deap import base
from deap import creator

from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from lxml import etree

import argparse as ap

from GPFramework.general_methods import load_environment


def eval_file(data_file, pset_info):

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

    individual_list = []

    if data_file is not None:

        with open(pre_path + "../../unitTestData/" + data_file, 'r') as input:

            for line in input:

                if line.strip():

                    my_func, i = parse_tree(line, pset_info)

                    my_tree = gp.PrimitiveTree(my_func)
                    my_individual = creator.Individual([my_tree, my_tree, my_tree, my_tree])
                    my_individual.hash_val = my_hash(my_individual)

                    individual_list.append(my_individual)

    return individual_list


def eval_wrapper(individual):

    print('Evaluating Individual', emade.my_str(individual))
    # sys.stdout.flush()
    del individual.fitness.values
    print(individual.fitness.values)
    individual, run_time, error_string, retry = emade.evaluate_individual(individual, 0)
    print("Individual: ", emade.my_str(individual), " ran for ", run_time, "s\nand returned the error:"  ,error_string)
    print("With the following fitness values: ", individual.fitness.values)

    #individual, run_time, error_string, retry = emade.evaluate_individual(individual, 1)
    #print("Individual: ", emade.my_str(individual), " ran for ", run_time, "s\nand returned the error:"  ,error_string)
    #print("With the following fitness values: ", individual.fitness.values)
    return individual.fitness.values



class TitanicUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        database, pset_info = load_environment("templates/input_titanic.xml")
        ind_list = eval_file("titanic_tree_eval.txt", pset_info)

        self.database = database
        self.ind_list = ind_list

    def setup(self):

        random.seed(100)
        np.random.seed(100)
        

    def test_individuals(self):

        print("Evaluating Individuals")

        for i, ind in enumerate(self.ind_list):

            print()
            print(i)
            print(eval_wrapper(ind))
        
    def test_objectives(self):

        print("Testing Objectives")

        for ind in self.ind_list:

            print("Evaluating Objectives for Individual: ", my_str(ind))
            individual, run_time, error, retry = emade.evaluate_individual(ind, 0)
            if len(error) > 0:
                print("Got Error: ", error)
            else:
                print("Successfully Evaluated Objectives for individual")


if __name__ == '__main__':
    unittest.main()
