from GPFramework.data import EmadeDataPair, EmadeData, load_many_to_one_from_file, load_feature_data_from_file
import GPFramework.sql_connection_orm_master as sql_connection_orm
from GPFramework.sql_connection_orm_base import ConnectionSetup
import GPFramework.gp_framework_helper as gpFrameworkHelper
from GPFramework.gp_framework_helper import LearnerType
from deap import creator
from GPFramework.emade_operators import cx_ephemerals
from GPFramework.general_methods import str2bool
from GPFramework.EMADE import my_str, my_hash
from GPFramework import EMADE as emade

import xml.etree.ElementTree as ET
from deap import creator
from lxml import etree
import argparse as ap
from deap import base
from deap import gp
import pandas as pd
import numpy as np
import random
from deap import gp
from GPFramework.emade_operators import cx_ephemerals
import copy as cp
import itertools
import hashlib
import inspect
import random
import pickle
import time
from GPFramework.sql_connection_orm_master import SQLConnectionMaster
import sys
import os
import re
import gc
import multiprocess as mp
import multiprocess.pool
from multiprocess import Process, Manager, Pool
from multiprocess.queues import Queue
import tensorflow as tf

if sys.platform != 'win32':
    import resource
elif sys.version_info[0] < 3.6:
    try:
        #resolve a bug with python 3.5 on windows
        #that incorrectly writes UNICODE to
        #command prompt (can comment out, but may experience)
        #OS errors while resizing command prompt on Windows 10
        import win_unicode_console
        #pip install win_unicode_console
        win_unicode_console.enable()
    except:
        print("UNICODE support not installed. You may experience crashes\
        related to writing to command prompt.")

def load_trees(individuals_filename):
    """
    Return some trees to evaluate.
    """
    # For now let's work with a saved pareto front
    with open(individuals_filename, 'rb') as input_file:
        pareto_front = pickle.load(input_file)

    individuals = [individual for individual in pareto_front]
    return individuals

def evaluation_wrapper(individual):
    """
    Wrap the evaluation process to return the fitness.
    """
    # Capture outputs back from the evaluation function
    # The second input in the tuple represents the dataset id,
    # In this case we are forcing the first (0th)
    print('Evaluating Individual', emade.my_str(individual))
    # sys.stdout.flush()
    del individual.fitness.values
    print(individual.fitness.values)
    print(tf.test.is_gpu_available())
    pool = emade.MyPool(processes=1)
    fitness_launches = [pool.apply_async(emade._inst.toolbox.evaluate, (individual, int(individual.age)))]
    eval_future = fitness_launches[0]
    individual, run_time, error_string, retry = eval_future.get()
    #individual, run_time, error_string, retry = emade.evaluate_individual(individual, 0)
    print("Individual: ", emade.my_str(individual), " ran for ", run_time, "s\nand returned the error:"  ,error_string)
    print("With the following fitness values: ", individual.fitness.values)

    #individual, run_time, error_string, retry = emade.evaluate_individual(individual, 1)
    #print("Individual: ", emade.my_str(individual), " ran for ", run_time, "s\nand returned the error:"  ,error_string)
    #print("With the following fitness values: ", individual.fitness.values)
    return individual.fitness.values


def evaluate_new_data(individual):
    """
        Reads in the information
    """
    # print(type(individual))
    emade._inst.pset.context['learnerType'] = LearnerType
    func = emade._inst.toolbox.compile(expr=individual)
    data_pair = emade._inst.datasetDict[0]['dataPairArray'][0]
    train = data_pair.get_train_data().get_instances()
    test = data_pair.get_test_data().get_instances()
    # print(train)
    # print(test)
    truth_data = emade._inst.datasetDict[0]['truthDataArray'][0]
    print(type(data_pair), type(truth_data))
    sys.stdout.flush()
    test = func(data_pair)
    print(type(test))
    sys.stdout.flush()
    print(type(test.get_train_data()), type(test.get_test_data()))
    # print(truth_data)
    #test_results = np.array([inst.get_target()[0] for inst in test.get_test_data().get_instances()])
    test_results = test.get_test_data().get_target()
    # results_dataset = np.loadtxt('datasets/new_lidar_final/test_0.csv.gz', delimiter=',')
    # good_instances = test_results != -1
    # new_dataset = results_dataset[good_instances, :]
    # print(new_dataset)
    # print(test_results.shape)
    # print(test_results)
    # print(type(truth_data))
    # print(type(test_results))
    print(emade.my_str(individual))
    #import pdb; pdb.set_trace()
    # pd.DataFrame(test_results).iloc[:, -1].to_csv('results.csv', header=False, index=False)
    # for prediction, truth in zip(test_results, truth_data):
    #     print("Prediciton: " + prediction + " " + "Truth: " + truth)
    # pd.DataFrame(test_results).to_csv('results2.csv', header=False, index=False)

def plotExecTimes(individual, N = 200):
    """
         Method to time the execution of an individual N times, then plot a histogram
         of the result. This can be useful for debugging if there is environmental pressure
         that kills some individuals do to implementation details (e.g. memory read time is
         sometimes delayed)
    """
    times = []
    import matplotlib as plt
    for i in range(N):
        print(i)
        start = time.time()
        emade.evaluate_individual(individual, 0)
        times += [time.time() - start]
    print("TIMES", times)
    plt.hist(times)
    plt.title("Execution Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")

def evaluateAllIndividuals(individuals):
    """
        Run evaluation locally on each individual.
    """
    # Iterate through and evaluate each individual
    for individual in individuals:
        #print str(individual), evaluation_wrapper(individual)
        objectives = evaluation_wrapper(individual)
        print(objectives)
        # sys.stdout.flush()

if __name__ == '__main__':
    """
    Script to evaluate 1 individual from the database.
    Syntax: thisScript.py path/to/XMLSchemaFile hashOfInterestFromDB
    """
    parser = ap.ArgumentParser('Read in pickled GP individuals to use for analysis.')
    parser.add_argument('input_xml', help='Input to EMADE, see inputSample.xml')
    parser.add_argument('hash', help='Hash to parse')
    args = parser.parse_args()
    if not args.hash or not args.input_xml:
        print("User must provide XML and hash to evaluate.")
        print('python src/GPFramework/database_tree_evaluator.py path_to_XML --hash "hashValue"')
        raise Exception("Need more input from user")
    else:
        print("Evaluating individual", args.hash, "with data from this file: ", args.input_xml)


    from GPFramework.general_methods import load_environment

    database, pset_info = load_environment(str(args.input_xml))

    # Selects individual based on hash value
    individual = database.select(args.hash)
    print(emade.my_str(individual.pickle))
    # Selects individual based on hash value
    #print(f"evaluating ind: {individual}")
    
        #outfile = open('myout','a')
    print(evaluation_wrapper(individual.pickle))
    # pop = [cp.deepcopy(individual.pickle) for i in range(100)]
    # count = emade.mutate(pop, emade._inst.toolbox.mutateEphemeral, .25)
    # print(count, 'mutated')
    # for i in pop:
    #     print(emade.my_str(i))
    # print(emade.my_str(individual.pickle))
        #outfile.write(evaluation_wrapper(individual.pickle))
        #outfile.close()
    # plotExecTimes(individual.pickle,5)
    # evaluate_new_data(individual.pickle)

    # get trees
    # individuals = load_trees(individual_file)

    # print(len(individuals))
    #
    # [print(emade.my_str(ind)) for ind in individuals]

    # evaluate_new_data(individuals[1])
    # print(evaluation_wrapper(individuals[1]))
