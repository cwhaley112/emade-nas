from GPFramework.sql_connection_orm_base import IndividualStatus, ConnectionSetup
from GPFramework.sql_connection_orm_master import SQLConnectionMaster
import GPFramework.gp_framework_helper as gpFrameworkHelper
from GPFramework.gp_framework_helper import LearnerType
from GPFramework.general_methods import parse_tree
from GPFramework.EMADE import my_str, my_hash
from GPFramework.data import EmadeDataPair, EmadeDataPairNN
from GPFramework import EMADE as emade
from GPFramework import neural_network_methods as nnm
import GPFramework.emade_operators as opfuncs
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from lxml import etree
import argparse as ap
import pandas as pd
from deap import gp
import copy as cp
import pickle
import time
import sys
import ast
import gc
import os
import unittest

def load_environment(startup_settings_filename, xml_file, train_file=None, test_file=None):
    """
    This method takes in a pickle file used to instantiate a EMADE environent.
    It will replicate the start up process so that evaluations can all be performed correctly
    startup_settings_filename: pickle file from launchEMADE.py
    """
    # Open the startup settings file for reading
    with open(startup_settings_filename, 'rb') as startup_settings_file:
        # Load evolution parameters, objectives, and datasets in order from the file
        evolutionParametersDict = pickle.load(startup_settings_file)
        objectivesDict = pickle.load(startup_settings_file)
        datasetDict = pickle.load(startup_settings_file)
        stats_dict = pickle.load(startup_settings_file)
        misc_dict = pickle.load(startup_settings_file)
        cacheDict = pickle.load(startup_settings_file)

        # Use the items from the settings file to prepare the enviornment
        if 'regression' in evolutionParametersDict:
            regression = evolutionParametersDict['regression']
        else:
            regression = False

        if train_file and test_file:
            datasetDict[0]['trainFilenames'] = [train_file]
            datasetDict[0]['testFilenames'] = [test_file]
            # del datasetDict[1]; gc.collect()

        # print(datasetDict)

        emade.create_representation(mods=3, regression=regression)

        emade.setObjectives(objectivesDict)
        ind = emade.setDatasets(datasetDict)
        emade.setMemoryLimit(misc_dict['memoryLimit'])
        emade.setCacheInfo(cacheDict)
        emade.set_statistics(stats_dict)
        emade.buildClassifier()

        # Initialize database for storing information about each individual
        inputFile = xml_file

        # Valid XML file with inputSchema.xsd using lxml.etree
        schema_doc = etree.parse(os.path.join('templates', 'inputSchema.xsd'))
        schema = etree.XMLSchema(schema_doc)

        doc = etree.parse(inputFile)
        # Raise error if invalid XML
        try:
            schema.assertValid(doc)
        except:
            raise

        # Uses xml.etree.ElementTree to parse the XML
        tree = ET.parse(inputFile)
        root = tree.getroot()

        db_info = root.find('dbConfig')
        server = db_info.findtext('server')
        username = db_info.findtext('username')
        password = db_info.findtext('password')
        database = db_info.findtext('database')
        reuse = int(db_info.findtext('reuse'))

        database_str = 'mysql://' + username + ':' + password + '@' + server + '/' + database
        fitness_names = [objectivesDict[objective]['name'] for objective in objectivesDict]
        dataset_names = [datasetDict[dataset]['name'] for dataset in datasetDict]
        ConnectionSetup()
        database = SQLConnectionMaster(connection_str=database_str, reuse=False, fitness_names=fitness_names,
                                                    dataset_names=dataset_names, statistics_dict=stats_dict, cache_dict=cacheDict,
                                                    is_worker=True)
        
        try:
            database.add_host('central')
        except:
            pass

        return ind, database

def load_trees(func, creator):
    """
    Return a tree to evaluate
    """
    return creator.Individual(func)


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
                            age=individual.age, elapsed_time=elapsed_time, retry_time=individual.retry_time,
                            evaluation_status=IndividualStatus.NOT_EVALUATED,
                            error_string=error_string)
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

    return individual.fitness.values


class Test(unittest.TestCase):
    
    def setUp(self): 

        xml_file = "templates/input_toxicity.xml"
        startup_file = "myPickleFile14928.dat"
        train_file_name = "datasets/wikidetox/train.csv.gz"
        test_file_name = "datasets/wikidetox/test.csv.gz"

        # Start off by loading up the environment
        from GPFramework.general_methods import load_environment as loader
        ind, database = loader(xml_file, train_file=train_file_name, test_file=test_file_name)

        print('env loaded')
        pset = gp.PrimitiveSetTyped("MAIN", [EmadeDataPairNN], EmadeDataPairNN)
        gpFrameworkHelper.addTerminals(pset)
        ephemeral_methods = gpFrameworkHelper.addPrimitives(pset)

        import inspect
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
                    "ephemeral_methods": ephemerals, 
                    "context": pset.context}
        

        strings_to_eval = ['NNLearner(ARG0, OutputLayer(ARG0, DenseLayer(15, defaultActivation, -14, DropoutLayer(4.7443816430113355, GRULayer(2, defaultActivation, 32, trueBool, trueBool, GlobalMaxPoolingLayer1D(Conv1DLayer(256, defaultActivation, 6, 1, randomUniformWeights, 1, EmbeddingLayer(7708, ARG0, randomUniformWeights, InputLayer(ARG0)))))))), 100, AdamOptimizer)']

        #strings_to_eval = ['NNLearner(ARG0,OutputLayer(ARG0,ConcatenateLayer2(ReLULayer(DenseLayer(10,10,EmbeddingLayer(100, ARG0, fasttextWeights, InputLayer(ARG0)))),ReLULayer(DenseLayer(10,10,LSTMLayer(16,0, trueBool, trueBool, EmbeddingLayer(100,ARG0,gloveWeights,InputLayer(ARG0))))))), 100)']
        #strings_to_eval = ['NNLearner(ARG0,OutputLayer(ARG0,ConcatenateLayer2(ConcatenateLayer2(ReLULayer(DenseLayer(10,10,EmbeddingLayer(100, ARG0, gloveFasttextWeights, InputLayer(ARG0)))),ReLULayer(DenseLayer(10,10, EmbeddingLayer(100, ARG0, gloveTwitterWeights, InputLayer(ARG0))))),ReLULayer(DenseLayer(10,10,LSTMLayer(16,0, trueBool, trueBool, EmbeddingLayer(100,ARG0,gloveWeights,InputLayer(ARG0))))))), 100)']
        #strings_to_eval = ['NNLearner(ARG0,OutputLayer(ARG0, []), 100)']
                           
        print('setup complete') 
        # Evaluate each string individually
        for s in strings_to_eval:
            # Construct a list of nodes from a string
            my_func, i = parse_tree(s, pset_info)

            # Construct a deap primitive tree and individual from the given list
            my_tree = gp.PrimitiveTree(my_func)
            from deap import creator
            my_individual = creator.Individual([my_tree, my_tree, my_tree, my_tree])

            # Print the seeded individual to terminal
            print('individual loaded')
            #print(my_str(my_individual))
            self.individual = my_individual[0]
            self.individual2 = gp.PrimitiveTree(self.individual.copy())
            #self.individual = opfuncs.remove_layer(self.individual)[0]
            self.pset = pset
    
#    def test_check_layer_nlp(self):
#        testlist = [self.individual]
#        from GPFramework.EMADE import mutate 
#        #individual = opfuncs.check_learner(self.individual, self.pset)[0]
#        individual = opfuncs.check_learner_nlp(self.individual, self.pset)[0]
#        
#        assert(individual == False),f"{str(self.individual)}"
#    
#    
#    def test_add_layer(self):
#        print(type(self.individual))
#        individual = opfuncs.add_layer(self.individual, self.pset)[0]
#         
#        assert(individual == False),f"{str(self.individual)}"

#        print(type(individual))
##    def test_shuffle_layers(self):
##        print(type(self.individual))
##        individual = opfuncs.shuffle_layers(self.individual)[0]
##         
##        assert(individual == False),f"{str(individual)}"
##        print(type(individual))
#    def test_delete_layer(self):
#        print(type(self.individual))
#        individual = opfuncs.remove_layer(self.individual, self.pset)[0]
#         
#        assert(individual == False),f"{str(self.individual)}"
#        print(type(individual))

#    def test_swap_layer(self):
#        print(type(self.individual))
#        individual = opfuncs.swap_layer(self.individual, self.pset)[0]
#        print(individual)
#        assert(individual == False),f"{str(self.individual)}"
##
#    def test_shuffle_layer(self):
#        print(type(self.individual))
#        individual = opfuncs.shuffle_layers(self.individual)[0]
#        assert(individual == False),f"{str(individual)}"
#        print(type(self.individual))


    def test_one_point_full_crossover(self):
        print(type(self.individual2))
        individual, individual2 = opfuncs.one_point_crossover_full_prim(self.individual, self.individual2)
        assert(individual == False),f"{str(individual) + str(individual2)}"
        assert(individual == False),f"{str(individual)}"
        print(type(self.individual))
    def test_one_point_middle_crossover(self):
        print(type(self.individual2))
        individual, individual2 = opfuncs.one_point_crossover_middle_prim(self.individual, self.individual2)
        assert(individual == False),f"{str(individual) + str(individual2)}"
        assert(individual == False),f"{str(individual)}"
        print(type(self.individual))

    def test_param_switched(self):
        print(type(self.individual))
        individual = opfuncs.mut_terminal_type_based(self.individual, self.pset, nnm.Activation)[0]
        assert(individual == False),f"{str(individual)}"
        print(type(self.individual))



if __name__ == '__main__':
    unittest.main()

