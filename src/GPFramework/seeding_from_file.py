from GPFramework.data import EmadeDataPairNN, EmadeDataPairNNF
from GPFramework.sql_connection_orm_base import IndividualStatus
from GPFramework.general_methods import parse_tree
from GPFramework.EMADE import my_str, my_hash
from GPFramework.general_methods import str2bool
from GPFramework.neural_network_methods import InputLayer, compile_layerlist, LayerListM
import xml.etree.ElementTree as ET
from deap import creator
from deap import base
import argparse as ap
from deap import gp
import numpy as np
import copy as cp
import itertools
import inspect
import dill


"""
INSTRUCTIONS FOR USE:

1) Edit or create the seeding file in SeedingFiles and type in the string representation of your individuals/modules
2) Make sure you set reuse to 1 or 0 depending on your intention in the xml file, also make sure the xml file has the
   correct database information.
3) run python src/GPFramework/seeding_from_file.py templates/[.xml file] [seeding_file]

WHAT IT DOES:

1) Adds individuals (NNLearners) in the seeding file to the database with status=NOT_EVALUATED. Assuming you run EMADE with reuse=1, 
   these individuals will be part of the initial population in gen 0
2) Adds modules to Global_MODS file if it exists, otherwise generates population of modules and dumps those + the seeds into
   Global_MODS file (these will be added to the database by the master algorithm). 
   If a seeded module name (like "mod_5") is in the existing Global_MODS file, it will overwrite the old module.
   If you want to avoid this, try assigning seeded modules a higher number than EMADE would 
   (example: if EMADE is configured to generate 20 modules, name your seeds "mod_21", "mod_22", etc).


EXAMPLE SEEDS (exact primitive/terminal names are subject to change):

NNLearner(ARG0, DenseLayer(LayerUnit32, defaultActivation, Layer_Normalization, FlattenLayer(mod_9(mod_20(mod_7(mod_11(mod_11(Input(ARG1))))))), NoDropout), AdamOptimizer)
NNLearner(ARG0, DenseLayer(LayerUnit128, reluActivation, Batch_Normalization, FlattenLayer(mod_19(mod_8(mod_15(Input(ARG1))))), NoDropout), AdamOptimizer)

mod_7: Module(datatype, data_dims, Conv2DLayer(FilterUnit48, reluActivation, KernelSize3, Stride1, ValidPadding, Batch_Normalization, ARG0), Conv2DConnection(reluActivation, KernelSize3, Stride2, ValidPadding, Layer_Normalization), MaxPool2D(PoolSize2), NoDropout)
mod_8: Module(datatype, data_dims, SeparableConv2DLayer(FilterUnit16, geluActivation, KernelSize1, Stride2, ValidPadding, No_Normalization, DenseLayer4dim(LayerUnit256, defaultActivation, Layer_Normalization, DenseLayer4dim(LayerUnit32, reluActivation, No_Normalization, ARG0))), Conv2DConnection(geluActivation, KernelSize5, Stride2, ValidPadding, No_Normalization), NoPooling(), Dropout20)
mod_9: Module(datatype, data_dims, SeparableConv2DLayer(FilterUnit128, geluActivation, KernelSize3, Stride1, ValidPadding, Batch_Normalization, ARG0), NoSkipConnection4dim(), MaxPool2D(PoolSize2), NoDropout)
mod_11: Module(datatype, data_dims, Conv2DLayer(FilterUnit256, geluActivation, KernelSize3, Stride1, ValidPadding, Batch_Normalization, SeparableConv2DLayer(FilterUnit16, geluActivation, KernelSize3, Stride1, SamePadding, Batch_Normalization, ARG0)), DenseConnection4dim(geluActivation, No_Normalization), NoPooling(), Dropout20)
mod_15: Module(datatype, data_dims, SeparableConv2DLayer(FilterUnit16, defaultActivation, KernelSize1, Stride1, ValidPadding, No_Normalization, SeparableConv2DLayer(FilterUnit16, defaultActivation, KernelSize3, Stride1, SamePadding, Batch_Normalization, Conv2DLayer(FilterUnit256, defaultActivation, KernelSize3, Stride1, SamePadding, Batch_Normalization, ARG0))), DenseConnection4dim(reluActivation, No_Normalization), MaxPool2D(PoolSize2), Dropout20)
mod_19: Module(datatype, data_dims, Conv2DLayer(FilterUnit256, defaultActivation, KernelSize1, Stride1, ValidPadding, Batch_Normalization, ARG0), SkipConnection4dim(reluActivation), MaxPool2D(PoolSize2), Dropout20)
mod_20: Module(datatype, data_dims, SeparableConv2DLayer(FilterUnit256, defaultActivation, KernelSize5, Stride1, SamePadding, No_Normalization, ARG0), SkipConnection4dim(defaultActivation), AvgPool2D(PoolSize2), NoDropout)
"""

# Takes an XML file to get dataset, objective, and evaluation information
parser = ap.ArgumentParser()
parser.add_argument(
    'filename', help='Input to EMADE, see inputSample.xml')
parser.add_argument('seeding_file', help='File to read seeded individuals from. See seeding_test_1.txt in SeedingFiles')
# parser.add_argument('-d', '--database', dest='database', default='sqlite', type=str, help='SQLAlchemy connection string in the form of dialect[+driver]://user:password@host/dbname. See http://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine')

args = parser.parse_args()
inputFile = args.filename
seedingFile = args.seeding_file
# database_str = args.database

import pathlib

#from GPFramework.general_methods import load_environment
def load_environment(input_xml, train_file=None, test_file=None):
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
    from GPFramework.data import EmadeDataPair
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
    evaluationModule = __import__(evaluationInfo.findtext('module'))
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

    # Determine whether the problem is a regression (True) or classification (False) problem
    regression = 0
    if root.findtext('regression') is None:
        gpFrameworkHelper.set_regression(False)
    elif root.findtext('regression') == 1:
        gpFrameworkHelper.set_regression(True)
        regression = 1
    else:
        gpFrameworkHelper.set_regression(False)

    # finished parsing XML file

    # Initialize emade instance and receive psets
    mods=20 # TODO put in xml file
    datatype = datasetDict[0]['type']
    emade.setDicts(objectiveDict, datasetDict, cacheDict, {})
    emade.buildClassifier()
    emade.setObjectives()
    pset, mod_pset, ephemeral_methods, data_dims = emade.create_representation(mods=20, regression=regression, datatype=datatype)
    emade.setDatasets()
    emade.setMemoryLimit(misc_dict['memoryLimit'])

    def get_pset_info(pset):
        terminals = {}
        primitives = {}
        ephemerals = {}
        for item in pset.mapping:
            if isinstance(pset.mapping[item], gp.Terminal):
                terminals[item] = pset.mapping[item]
            elif isinstance(pset.mapping[item], gp.Primitive):
                primitives[item] = pset.mapping[item]

        methods = dir(gp)
        for method in methods:
            pointer = getattr(gp, method)
            if inspect.isclass(pointer) and issubclass(pointer, gp.Ephemeral):
                ephemerals[method] = pointer

        pset_info = {"primitives": primitives,
                    "terminals": terminals, 
                    "ephemerals": ephemerals, 
                    "ephemeral_methods": ephemeral_methods, 
                    "context": pset.context,
                    "pset":pset}
        
        return pset_info
    
    nn_pset_info = get_pset_info(pset)
    mod_pset_info = get_pset_info(mod_pset)

    # Compute arrays of weights and thresholds from objective information
    weights = tuple([objectiveDict[objective]['weight'] for objective in objectiveDict])
    creator.create("FitnessMin", base.Fitness, weights=weights)
    fitness_names = (datasetDict[dataset]['name'] for dataset in datasetDict)
    fitness_attr = dict(zip(fitness_names, itertools.repeat(creator.FitnessMin)))

    creator.create("Individual", list, pset=pset, fitness=creator.FitnessMin, age=0, hash_val=None, **fitness_attr)
    creator.create("ADF", list, fitness=creator.FitnessMin, reshape=[],
        pset=mod_pset, weights=None, age=0, num_occur=0, retry_time=0, novelties = None,
        mod_num=None, **fitness_attr)

    fitness_names = [objectiveDict[objective]['name'] for objective in objectiveDict]
    dataset_names = [datasetDict[dataset]['name'] for dataset in datasetDict]

    ConnectionSetup()

    # Initialize database for storing information about each individual
    database = SQLConnectionMaster(connection_str=database_str, reuse=0, fitness_names=fitness_names,
                                                dataset_names=dataset_names, statistics_dict={}, cache_dict=cacheDict, is_worker=True)

    database.add_host('central')

    print("connected to database")

    return database, nn_pset_info, mod_pset_info, datatype, data_dims

database, nn_pset_info, mod_pset_info, datatype, data_dims = load_environment(inputFile)

running_path = str(pathlib.Path(__file__).parent.absolute())
working_path = str(pathlib.Path().absolute())

individuals = list()
modules = list()
#with open(pre_path + "../../SeedingFiles/" + seedingFile, 'r') as input:
with open(working_path+"/SeedingFiles/" + seedingFile, 'r') as input:
    for line in input:
        if line.startswith("#"):
            continue
        elif line.strip() and line.startswith("mod_"):
            modules.append(line.strip())
        elif line.strip():
            individuals.append(line.strip())

# load Global_MODS file (will always exist because we've run emade.create_representation)
with open("Global_MODS", "rb") as mod_file:
    glob_mods = dill.load(mod_file)

# we add modules first because they are elements of seeded NNLearners
mod_names = set()
for mod in modules:
    mod_name, string = [mod_.strip() for mod_ in mod.split(':')]
    if mod_name in mod_names:
        # if a module name is used multiple times in a seed file, only use the first occurrence
        print(f"skipping {string} because {mod_name} has already been seen in the seed file")
        continue
    else:
        mod_names.add(mod_name)
    expr = parse_tree(string, mod_pset_info)[0]
    tree = gp.PrimitiveTree(expr)

    # set attributes of my_mod
    my_mod = creator.ADF([tree])
    my_mod.mod_num = int(mod_name.split('_')[1])
    my_mod.age = 0
    my_mod.num_occur = 0
    my_mod.name = mod_name

    # get reshape value
    func = gp.compile(tree, mod_pset_info['pset'])
    layerlist = func(InputLayer())
    layers, _ = compile_layerlist(layerlist, [], data_dims, datatype)
    reshape = np.subtract(data_dims, layers.shape[1:])
    my_mod.reshape = reshape

    print(my_str(my_mod, name=mod_name))

    # add my_mod to Global_MODS
    glob_mods[mod_name] = [mod_name, my_mod, list(reshape)]

    # overwrite module in nn pset
    nn_pset_info['primitives'][mod_name] = gp.Primitive(mod_name, [LayerListM], LayerListM)
    

# overwrite module file
with open("Global_MODS", "wb") as mod_file:
    dill.dump(glob_mods, mod_file)

for ind in individuals:
    # recursively build a list of a seed
    # appends primitives, ephemerals, and terminals based on the given tree string
    my_func, i = parse_tree(ind, nn_pset_info)

    # construct a deap primitive tree and individual from the given list
    my_tree = gp.PrimitiveTree(my_func)
    my_individual = creator.Individual([my_tree])

    # print the seeded individual to terminal
    print("Got Individual: ",my_str(my_individual))

    # Set Attributes
    my_hash_id = my_hash(my_individual)
    my_individual.hash_val = my_hash_id
    #my_individual.fitness_attr = fitness_attr

    # insert the individual into the database
    try:
        database.insertInd(hash=my_hash_id, individual=cp.deepcopy(my_individual),
                                            age=0, evaluation_gen=0,
                                            evaluation_status=IndividualStatus.NOT_EVALUATED,
                                            tree=my_str(my_individual))
    except:
        print("This individual is already in the database")