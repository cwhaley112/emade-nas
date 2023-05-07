"""
Programmed by Jason Zutty
Modified by VIP Team
Implements a number of general purpose functions for the GP primitives
"""
import tensorflow as tf
import numpy as np
import copy as cp
import sys
import ast

import argparse as ap
import xml.etree.ElementTree as ET
from lxml import etree

from deap import gp
from deap import base
from deap import creator

import os
import itertools
import inspect

def str2bool(v):
    return v is not None and v.lower() in ("yes", "true", "t", "1")

"""Written by Austin Dunn"""
def parse_tree(line, pset_info, mode=0):
    """
    Parses the string of a seed assuming the seed is a tree

    Args:
        line (string): string of the tree
        mode (int):  0 for a normal primitive, 1 for a learner, 2 for an ensemble
  
    Returns:
        List of deap.gp objects (Primitives, Terminals, and Ephemerals)
    """
    # setup variables
    my_func = []
    node = ""
    wait = False
    parse = True
    i = 0
 
    # parse string until end of ephemeral/primitive is reached
    while parse:
  
        if line[i] == ")" and node == "":
            return my_func, i
        
        if (line[i] == "(" or line[i] == ")" or line[i] == "," or line[i-1] == "}" or line[i-1] == "]") and not wait:
            # remove the space before the "," if it exists
            if node[-1] == " ":
                node = node[:-1]
            elif node[0] == " ":
                node = node[1:]
            # figure out what node is
            if node in pset_info['primitives']:
                # add primitive to my_func since we're done parsing it
                my_func.append(pset_info['primitives'][node])

                # recursive call given rest of un-parsed line
                their_func, x = parse_tree(line[i+1:], pset_info)

                # skip over ")" and "," after returning the recursive call
                # this prevents checking the next string too early
                while (i+x+1 < len(line)) and (line[i+x+1] == ")" or line[i+x+1] == ","):
                    x += 1

                # update current list and index
                my_func += their_func
                i += x
            elif node in pset_info['terminals']:
                # add terminal to my_func since we're done parsing it
                # this is mainly used for ARG0
                my_func.append(pset_info['terminals'][node])
            elif node == "LearnerType":
                # recursive call with mode set for LearnerType
                their_func, x = parse_tree(line[i+1:], pset_info, mode=1)

                # skip over ")" and "," after returning the recursive call
                # this prevents checking the next string too early
                while (i+x+1 < len(line)) and (line[i+x+1] == ")" or line[i+x+1] == ","):
                    x += 1

                # update current list and index
                my_func += their_func
                i += x
            elif node == "EnsembleType":
                # recursive call with mode set for EnsembleType
                their_func, x = parse_tree(line[i+1:], pset_info, mode=2)

                # skip over ")" and "," after returning the recursive call
                # this prevents checking the next string too early
                while (i+x+1 < len(line)) and (line[i+x+1] == ")" or line[i+x+1] == ","):
                    x += 1

                # update current list and index
                my_func += their_func
                i += x
            elif node in pset_info['ephemeral_methods']:
                # handles ephemerals using method references
                ephem = pset_info['ephemerals'][pset_info['ephemeral_methods'][node]]()
                ephem.value = pset_info['context'][node]
                my_func.append(ephem)
            else:
                # assume the node is a literal (float, int, string, list, dict, etc.)
                try:
                    node = ast.literal_eval(node)
                except Exception as e:
                    print("node: {}".format(node))
                    raise e
                if isinstance(node, int):
                    if node <= 15:
                        ephem = pset_info['ephemerals']['myRandInt']()
                        ephem.value = node
                    elif node > 15 and node <= 100:
                        ephem = pset_info['ephemerals']['myMedRandInt']()
                        ephem.value = node
                    elif node > 100:
                        ephem = pset_info['ephemerals']['myBigRandInt']()
                        ephem.value = node
                elif isinstance(node, float):
                    ephem = pset_info['ephemerals']['myGenFloat']()
                    ephem.value = node
                elif isinstance(node, list):
                    ephem = pset_info['ephemerals']['listGen']()
                    ephem.value = node
                elif (isinstance(node, dict) or isinstance(node, str) or node is None) and mode > 0:
                    # in this case we just want the value
                    ephem = node
                else:
                    print("DEBUG (Node that was unsupported):", node)
                    raise ValueError("Unsupported type used as an argument.")

                my_func.append(ephem)

            # reset node to empty string
            # at this point we no longer need the old string because it's already processed
            node = ""

        else:
            # handle dicts and lists
            if line[i] == "{" or line[i] == "[":
                wait = True

            # add the char to node
            node += line[i]
   
        # check if end of ephemeral/primitive has been reached and if end of list or dict has been reached
        if line[i] == ")":
            parse = False
        elif line[i] == "}" or line[i] == "]":
            wait = False
   
        # increment index
        i += 1
    
    if mode == 1:
        # construct learnerType
        if len(my_func) > 2:
            raise ValueError("learnerType has too many arguments in your seed.\n \
                             The format should be: 'learnerType(name, params)'")

        learner = pset_info['ephemerals']['learnerGen']()
        learner.value.name = my_func[0]
        learner.value.params = my_func[1]
        my_func = [learner]
    elif mode == 2:
        # construct ensembleType
        if len(my_func) > 2:
            raise ValueError("ensembleType has too many arguments in your seed.\n \
                             The format should be: 'ensembleType(name, params)'")

        ensemble = pset_info['ephemerals']['ensembleGen']()
        ensemble.value.name = my_func[0]
        ensemble.value.params = my_func[1]
        my_func = [ensemble]
    
    return my_func, i

"""Written by Austin Dunn"""
def map_labels(data_pair, labels, set_):
    """Maps list of predicted labels back to instances ignoring empty instances

    Args:
        data_pair: data object containing data
        labels:    numpy array containing new labels
        set_:       boolean value for whether to apply to train data or test data

    Returns:
        data_pair with updated
    """
    ind = 0
    if set_:
        instances = data_pair.get_test_data().get_instances()
    else:
        instances = data_pair.get_train_data().get_instances()
    for inst in instances:
        data = inst.get_features().get_data()
        if len(data) > 0:
            r = len(data) if len(data.shape) > 1 else 1
            target = np.array([labels[i] for i in range(ind, ind+r)])
            ind += r
            inst.set_target(target)
    return data_pair

"""Written by Austin Dunn"""
def flatten_obj_array(counter, data):
    """
    Recursive function for flattening numpy arrays of type object

    Args:
        counter: int to keep track of which dimension the function is on
        data:    numpy array to flatten

    Returns:
        flattened numpy array
    """
    if counter == 0:
        return data
    
    data = np.concatenate([flatten_obj_array(counter - 1, i).flatten() for i in data])
    
    return data

def if_then_else(expression, if_value, else_value):
    """Return if_value if expression is true

    Args:
        expression: expression to evaluate
        if_value: value if true
        else_value: value if false

    Returns:
        if_value or else_value
    """
    if expression:
        return if_value
    else:
        return else_value

def consume_data(data):
    """
    This is a function to input the data in to a tree.
    This is required so that the data can be copied each time it comes in to the tree, but not in every primitive

    Args:
        data: data to input into the tree

    Returns:
        deep copy of data
    """
    return cp.deepcopy(data)

def to_list(x):
    """This method simply takes an input x and places it in to a list

    Args:
        x: input

    Returns:
        list containing x
    """
    return list([x])

def pingGPU():
    """Tries to execute a single multiply function on a GPU in order to
    determine if Tensorflow is GPU-enabled"""
    try:
        with tf.device('/gpu'):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
            c = tf.matmul(a, b)

        try:
            with tf.Session() as sess:
                sess.run(c)
                print("Tensorflow is running with GPU optimization")
                sess.close()
        except:
            print("Tensorflow session code failed. TF version is:", tf.__version__)
            print("tensorflow.session was depreceated in Tensorflow 2.0.0")

    except:
        print("Tensorflow was unable to run GPU optimization.")

def print_mem():
    if sys.platform == "linux" or sys.platform == "linux2":
        import resource
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(mem, 'MB')
        sys.stdout.flush()

def target_value_check(targets):
    """Checks target value

    Args:
        targets: given targets

    Raises:
        ValueError
    """
    
    unique_targets = np.unique(targets)
    if len(unique_targets) == 1:
       raise ValueError('Only one target: ' + str(unique_targets))

def mod_select(array, dictionary, key):
    """Selects based on key and mod

    Args:
        array: given array
        dictionary: given dictionary
        key: given key

    Returns:
        An item from the array
    """
    if key not in dictionary:
        return array[0]
    else:
        integer = dictionary[key]
        return array[integer % len(array)]

def obsSelect(obsMat, obsVec, obsNum):
    """Creates a tuple of features and their results from a given observation

    Args:
        obsMat: observation matrix
        obsVec: observation vector
        obsNum: observation number

    Returns:
        A tuple containing a set of features from a given observation and the
        result of those features
    """
    return (obsMat[obsNum, :], obsVec[obsNum])

def featSelect(obsMat, obsVec, featNum):
    """Creates a tuple of a set of observations of a given feature and the
    resulting vector of those features

    Args:
        obsMat: observation matrix
        obsVec: observation vector
        featNum: feature number

    Returns:
        A tuple containing a set of observations of a given feature and the
        result vector of those features
    """
    return (obsMat[:, featNum], obsVec)


def load_environment(input_xml, train_file=None, test_file=None, reuse=1):

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

    from GPFramework.EMADE import my_str, my_hash
    from GPFramework.sql_connection_orm_base import IndividualStatus, ConnectionSetup
    from GPFramework.sql_connection_orm_master import SQLConnectionMaster
    from GPFramework.gp_framework_helper import LearnerType
    from GPFramework.data import EmadeDataPair, EmadeDataPairNN, EmadeDataPairNNF
    from GPFramework import EMADE as emade
    from GPFramework import eval_methods
    import GPFramework.data as data
    import GPFramework.gp_framework_helper as gpFrameworkHelper
    
    inputFile = input_xml

    '''
    schema_doc = etree.parse(os.path.join('../../templates', 'inputSchema.xsd'))
    schema = etree.XMLSchema(schema_doc)

    doc = etree.parse(inputFile)
    # Raise error if invalid XML
    try:
        schema.assertValid(doc)
    except:
        raise
    '''

    tree = ET.parse(inputFile)
    root = tree.getroot()

    database_str_info = root.find('dbConfig')
    server_ip = database_str_info.findtext('server')
    server_username = database_str_info.findtext('username')
    server_password = database_str_info.findtext('password')
    server_database = database_str_info.findtext('database')
    database_str = "mysql://" + server_username + ":" + server_password + "@" + server_ip + "/" + server_database

    cacheInfo = root.find('cacheConfig')
    cacheDict =  {'cacheLimit': float(cacheInfo.findtext('cacheLimit')),
                  'central': str2bool(cacheInfo.findtext('central')),
                  'compression': str2bool(cacheInfo.findtext('compression')),
                  'useCache': str2bool(cacheInfo.findtext('useCache')),
                  'timeThreshold': float(cacheInfo.findtext('timeThreshold')),
                  'timeout': cacheInfo.findtext('timeout'),
                  'masterWaitTime': int(cacheInfo.findtext('masterWaitTime')),
                  'database': database_str}

    valid_types = {"int":int, "float":float, "str":str}

    
    # Initializes the dataset dictionary (Only need this to get the sname of the dataset)
    datasetDict = {}
    datasetList = root.iter('dataset')
    for datasetNum, dataset in enumerate(datasetList):
    # Iterate over each dataset and add to dictionary
        monte_carlo = dataset.iter('trial')
        datasetDict[datasetNum] = {'name': dataset.findtext('name'),
                               'type': dataset.findtext('type'),
                               'pickle': False if dataset.findtext('pickle') is None else str2bool(dataset.findtext('pickle')),
                               'multilabel': False if dataset.findtext('multilabel') is None else str2bool(dataset.findtext('multilabel')),
                               'regression': False if dataset.findtext('regression') is None else str2bool(dataset.findtext('regression')),
                               'reduceInstances': 1 if dataset.findtext('reduceInstances') is None else float(dataset.findtext('reduceInstances')),
                               'batchSize': None if dataset.findtext('batchSize') is None else int(dataset.findtext('batchSize')),
                               'trainFilenames':[] if train_file is None else [train_file],
                               'testFilenames':[] if test_file is None else [test_file]}
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
    try:
        evaluationModule = __import__(evaluationInfo.findtext('module'))
    except:
        evaluationModule = __import__("GPFramework."+evaluationInfo.findtext('module'), fromlist=[None])
    #evaluationModule = __import__("eval_methods")
    for objectiveNum, objective in enumerate(objectiveList):
        # Iterate over each objective and add to dictionary

        # convert all the arguments in the xml into their correct types
        o_args = {}
        args_dir = objective.find('args')
        if args_dir is not None:
            for i in args_dir.iterfind('arg'):
                o_args[i.findtext('name')] = valid_types[i.findtext('type')](i.findtext('value'))

        objectiveDict[objectiveNum] = {'name': objective.findtext('name'), 'weight': float(objective.findtext('weight')),
        'achievable': float(objective.findtext('achievable')), 'goal': float(objective.findtext('goal')),
        'evaluationFunction': getattr(evaluationModule, objective.findtext('evaluationFunction')),
        'lower': float(objective.findtext('lower')), 'upper': float(objective.findtext('upper')),
        'args': o_args}

    evolutionParameters = root.find('evolutionParameters')
    evolutionParametersDict = {}
    # Adds evolution parameters to evolution dictionary
    evolutionParametersDict['initialPopulationSize'] = int(evolutionParameters.findtext('initialPopulationSize'))
    evolutionParametersDict['elitePoolSize'] = int(evolutionParameters.findtext('elitePoolSize'))
    evolutionParametersDict['launchSize'] = int(evolutionParameters.findtext('launchSize'))
    evolutionParametersDict['minQueueSize'] = int(evolutionParameters.findtext('minQueueSize'))

    # Adds mating dict parameters to evolution dictionary
    evolutionParametersDict['matingDict'] = {}
    matingList = evolutionParameters.iter('mating')
    for mating in matingList:
        evolutionParametersDict['matingDict'][mating.findtext('name')] = float(mating.findtext('probability'))

    # Adds mutation dict parameters to evolution dictionary
    evolutionParametersDict['mutationDict'] = {}
    mutationList = evolutionParameters.iter('mutation')
    for mutation in mutationList:
        evolutionParametersDict['mutationDict'][mutation.findtext('name')] = float(mutation.findtext('probability'))


    # Iterate over selection algorithms
    selectionList = evolutionParameters.iter('selection')
    for selection in selectionList:
        evolutionParametersDict['selection'] = selection.findtext('name')


    # Add rest of parameters to evolution parameters dictionary

    seedFile = root.find('seedFile').findtext('filename')
    genePoolFitnessOutput = root.find('genePoolFitness').findtext('prefix')
    paretoFitnessOutput = root.find('paretoFitness').findtext('prefix')
    paretoOutput = root.find('paretoOutput').findtext('prefix')
    parentsOutput = root.find('parentsOutput').findtext('prefix')
    #print(seedFile, genePoolFitnessOutput, paretoFitnessOutput)
    evolutionParametersDict['seedFile'] = seedFile
    evolutionParametersDict['genePoolFitness'] = genePoolFitnessOutput
    evolutionParametersDict['paretoFitness'] = paretoFitnessOutput
    evolutionParametersDict['paretoOutput'] = paretoOutput
    evolutionParametersDict['parentsOutput'] = parentsOutput
    evolutionParametersDict['outlierPenalty'] = float(evolutionParameters.findtext('outlierPenalty'))

    evolutionParametersDict['memoryLimit'] = float(evaluationInfo.findtext('memoryLimit'))   


    misc_dict = {}
    misc_dict['seedFile'] = tree.find('seedFile').findtext('filename')
    misc_dict['genePoolFitness'] = tree.find('genePoolFitness').findtext('prefix')
    misc_dict['paretoFitness'] = tree.find('paretoFitness').findtext('prefix')
    misc_dict['paretoOutput'] = tree.find('paretoOutput').findtext('prefix')
    misc_dict['parentsOutput'] = tree.find('parentsOutput').findtext('prefix')
    misc_dict['memoryLimit'] = float(tree.find('evaluation').findtext('memoryLimit'))

    fitness_names = [objectiveDict[objective]['name'] for objective in objectiveDict]
    dataset_names = [datasetDict[dataset]['name'] for dataset in datasetDict]

    # Initialize pset
    pset = gp.PrimitiveSetTyped("MAIN", [EmadeDataPairNN], EmadeDataPairNNF)
   
    datatype = datasetDict[0]['type']
    gpFrameworkHelper.addTerminals(pset, datatype)
    ephemeral_methods = gpFrameworkHelper.addPrimitives(pset, datatype)

    # Determine whether the problem is a regression (True) or classification (False) problem
    regression = 0
    if root.findtext('regression') is None:
        gpFrameworkHelper.set_regression(False)
    elif root.findtext('regression') == 1:
        gpFrameworkHelper.set_regression(True)
        regression = 1
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
    creator.create("Individual", list, pset=pset, fitness=creator.FitnessMin, age=0, hash_val=None, **fitness_attr)

    fitness_names = [objectiveDict[objective]['name'] for objective in objectiveDict]
    dataset_names = [datasetDict[dataset]['name'] for dataset in datasetDict]
    
    emade.create_representation(mods=3, regression=regression, datatype=datatype)
    emade.setObjectives(objectiveDict)
    emade.setDatasets(datasetDict)
    emade.setMemoryLimit(memoryLimit = float(evaluationInfo.findtext('memoryLimit')))
    emade.setCacheInfo(cacheDict)
    emade.set_statistics({})
    emade.buildClassifier()

    ConnectionSetup()

    # Initialize database for storing information about each individual
    database = SQLConnectionMaster(connection_str=database_str, reuse=reuse, fitness_names=fitness_names,
                                                dataset_names=dataset_names, statistics_dict={}, cache_dict=cacheDict, is_worker=True)
 
    database.add_host('central')

    print("connected to database")

    return database, pset_info
