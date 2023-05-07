from GPFramework.sql_connection_orm_base import IndividualStatus, ConnectionSetup
from GPFramework.sql_connection_orm_master import SQLConnectionMaster
import GPFramework.gp_framework_helper as gpFrameworkHelper
from GPFramework.gp_framework_helper import LearnerType
from GPFramework.general_methods import parse_tree, load_environment
from GPFramework.EMADE import my_str, my_hash
from GPFramework.data import EmadeDataPair
from GPFramework import EMADE as emade
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from lxml import etree
import argparse as ap
import pandas as pd
from deap import gp
from deap import creator
import copy as cp
import pickle
import time
import sys
import ast
import gc
import os


def evaluation_wrapper(individual, dataset=0, database=None):
    """
    Wrap the evaluation process to return the fitness
    """
    # Capture outputs back from the evaluation function
    # The second input in the tuple represents the dataset id,
    # In this case we are forcing the first (0th)
    print('Evaluating Individual', my_str(individual))
    sys.stdout.flush()


    individual.hash_val = my_hash(individual)
    my_hash_id = individual.hash_val
    data = database.select(my_hash_id)
    if not data:
        database.insertInd(hash=individual.hash_val, tree=my_str(individual), individual=cp.deepcopy(individual),
                        age=individual.age, evaluation_gen=0,
                        evaluation_status=IndividualStatus.NOT_EVALUATED)

    individual, elapsed_time, error_string, retry = emade.evaluate_individual(individual, dataset)

    data = database.select(my_hash_id)

    if not data:
        if retry:
            database.insertInd(hash=my_hash_id, tree=my_str(individual), individual=cp.deepcopy(individual),
                            age=individual.age, elapsed_time=elapsed_time, retry_time=individual.retry_time,
                            evaluation_status=IndividualStatus.NOT_EVALUATED,
                            error_string=error_string)
        else:
            database.insertInd(hash=my_hash_id, tree=my_str(individual), individual=cp.deepcopy(individual),
                            age=individual.age, elapsed_time=elapsed_time, retry_time=individual.retry_time,
                            evaluation_status=IndividualStatus.WAITING_FOR_MASTER,
                            error_string=error_string)
    else:
        if retry:
            database.updateInd(row=data, individual=cp.deepcopy(individual),
                            age=individual.age, elapsed_time=elapsed_time,
                            retry_time=individual.retry_time, evaluation_start_time=None,
                            evaluation_status=IndividualStatus.NOT_EVALUATED, error_string=error_string)
        else:
            database.updateInd(row=data, individual=cp.deepcopy(individual),
                            age=individual.age, elapsed_time=elapsed_time, retry_time=individual.retry_time,
                            evaluation_status=IndividualStatus.WAITING_FOR_MASTER,
                            error_string=error_string)
            
    if retry:
        print("Individual Sent Back for Evaluation | " + str(datetime.now()))
        print('Received: ' + my_str(individual))
        print('\tWith Hash', my_hash_id)
        print('\tComputed in: ' + str(elapsed_time) + ' seconds')
        print('\tWith Age: ' + str(individual.age))
    else:
        print("Individual Evaluated | " + str(datetime.now()))
        print('Received: ' + my_str(individual))
        print('\tWith Hash', my_hash_id)
        print('\tComputed in: ' + str(elapsed_time) + ' seconds')
        print('\tWith Fitnesses: ' + str(individual.fitness.values))
        print('\tWith Age: ' + str(individual.age))
        if error_string:
            print('\tWith Error: ', error_string)
    sys.stdout.flush()

    if retry:
        print("Individual Sent Back for Evaluation | " + str(datetime.now()))
        print('Received: ' + my_str(individual))
        print('\tWith Hash', my_hash_id)
        print('\tComputed in: ' + str(elapsed_time) + ' seconds')
        print('\tWith Age: ' + str(individual.age))
    else:
        print("Individual Evaluated | " + str(datetime.now()))
        print('Received: ' + my_str(individual))
        print('\tWith Hash', my_hash_id)
        print('\tComputed in: ' + str(elapsed_time) + ' seconds')
        print('\tWith Fitnesses: ' + str(individual.fitness.values))
        print('\tWith Age: ' + str(individual.age))
        if error_string:
            print('\tWith Error: ', error_string)
    sys.stdout.flush()

    return individual.fitness.values

def eval_string(individual:str, database, pset_info, dataset=0):
    # function to call standalone from another script, written for GPFramework.analyze
    s = individual.rstrip()
    my_func, i = parse_tree(s, pset_info)
    my_tree = gp.PrimitiveTree(my_func)
    my_individual = creator.Individual([my_tree, my_tree, my_tree, my_tree])
    print("Evaluting:\n"+my_str(my_individual))
    return evaluation_wrapper(my_individual, dataset=dataset, database=database)

if __name__ == '__main__':
    """
    Run this script colacted with the datasets directory or a symlink to it
    """
    print('parsing args')
    parser = ap.ArgumentParser('Read in pickled GP individuals to use for analysis.')
    parser.add_argument('xml_file', type=str)
    #parser.add_argument('startup_settings_filename', type=str)
    parser.add_argument('-tr', '--trainFile', dest='train_file_name', type=str, help='Training data to use')
    parser.add_argument('-te', '--testFile', dest='test_file_name', type=str, help='Testing data to use')
    args = parser.parse_args()
    print('args parsed')

    xml_file = args.xml_file
    #startup_file = args.startup_settings_filename
    train_file_name = args.train_file_name
    train_file_name = None
    test_file_name = args.test_file_name
    test_file_name = None

    # Start off by loading up the environment

    print('starting')
    database, pset_info = load_environment(xml_file, train_file=train_file_name, test_file=test_file_name)
    print('env loaded') 
    """
    Sample Learner Strings
    
    "Learner(ARG0, LearnerType('RAND_FOREST', {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0}), EnsembleType('SINGLE', None))"
    "Learner(ARG0, ModifyLearnerInt(learnerType('RAND_FOREST', {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0}), 99, 0), ensembleType('SINGLE', None))"
    
    """
    # treefile = open('SeedingFiles/seeding_test_toxicity','r')
    #strings_to_eval = treefile.readlines()

    # strings_to_eval = ["NNLearner(ARG0, InputLayerTerminal, 6, SGDOptimizer)", "NNLearner(ARG0, InputLayer(), 6, SGDOptimizer)"]
    strings_to_eval = ["NNLearner(ARG0, DenseLayer(DenseLayerUnit32, defaultActivation, Conv2DLayer(Conv2DFilterUnit48, defaultActivation, Conv2DKernelSize3, InputLayerTerminal)), 128, AdamOptimizer)"]
    # strings_to_eval = ["NNLearner(ARG0,OutputLayer(DenseLayer(10, defaultActivation, 10, LSTMLayer(16, defaultActivation, 0, trueBool, trueBool,EmbeddingLayer(100, ARG0, randomUniformWeights, InputLayer())))), 128, AdamOptimizer)"]

    print('setup complete') 
    # Evaluate each string individually
    for s in strings_to_eval:
        s = s.rstrip()
        print(s)
        # Construct a list of nodes from a string
        my_func, i = parse_tree(s, pset_info)

        # Construct a deap primitive tree and individual from the given list
        my_tree = gp.PrimitiveTree(my_func)
        #import pdb; pdb.set_trace()
        import emade_operators
        #emade_operators.modify_layer(my_individual[0])
        # emade_operators.swap_layer(my_individual[0])
        my_individual = creator.Individual([my_tree, my_tree, my_tree, my_tree])

        # Print the seeded individual to terminal
        print(my_str(my_individual))
        
        # Run Evaluation with Dataset 0
        print(evaluation_wrapper(my_individual, dataset=0, database=database))
