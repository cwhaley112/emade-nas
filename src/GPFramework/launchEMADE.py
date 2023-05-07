#!/usr/bin/env python
import argparse as ap
import xml.etree.ElementTree as ET
from lxml import etree
import os
import subprocess
import multiprocess
import pickle
import sys
import glob
import time
import datetime
import re
from importlib import import_module

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
        
def str2bool(v):
    return v is not None and v.lower() in ("yes", "true", "t", "1")


def cache_params(tree) -> dict:
    # Initializes the cache dictionary
    cacheInfo = root.find('cacheConfig')
    # Get database information
    db_info = root.find('dbConfig')
    server = db_info.findtext('server')
    username = db_info.findtext('username')
    password = db_info.findtext('password')
    database = db_info.findtext('database')
    reuse = int(db_info.findtext('reuse'))
    database_str = 'mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database #+ '?charset=utf8'
    return {'cacheLimit': float(cacheInfo.findtext('cacheLimit')),
            'central': str2bool(cacheInfo.findtext('central')),
            'compression': str2bool(cacheInfo.findtext('compression')),
            'useCache': str2bool(cacheInfo.findtext('useCache')),
            'timeThreshold': float(cacheInfo.findtext('timeThreshold')),
            'timeout': cacheInfo.findtext('timeout'),
            'masterWaitTime': int(cacheInfo.findtext('masterWaitTime')),
            'database': database_str}

def dataset_params(tree) -> dict:
    dataset_dict = {}
    datasetList = tree.iter('dataset')
    for datasetNum, dataset in enumerate(datasetList):
        # Iterate over each dataset and add to dictionary
        monte_carlo = dataset.iter('trial')
        ri = dataset.findtext('reduceInstances')
        ml = dataset.findtext('multilabel')
        rg = dataset.findtext('regression')
        dataset_dict[datasetNum] = {'name': dataset.findtext('name'),
                                   'type': dataset.findtext('type'),
                                   'pickle': False if dataset.findtext('pickle') is None else str2bool(dataset.findtext('pickle')),
                                   'multilabel': False if dataset.findtext('multilabel') is None else str2bool(dataset.findtext('multilabel')),
                                   'reduceInstances': 1 if dataset.findtext('reduceInstances') is None else float(dataset.findtext('reduceInstances')),
                                   'regression': False if dataset.findtext('regression') is None else str2bool(dataset.findtext('regression')),
                                   'batchSize': None if dataset.findtext('batchSize') is None else int(dataset.findtext('batchSize')),
                                   'trainFilenames':[],
                                   'testFilenames':[]}

        # The data is already folded for K-fold cross validation. Add these to the trainFilenames
        # and testFilenames lists
        for trial in monte_carlo:
            dataset_dict[datasetNum]['trainFilenames'].append(
                trial.findtext('trainFilename'))
            dataset_dict[datasetNum]['testFilenames'].append(
                trial.findtext('testFilename'))
    return dataset_dict

valid_types = {"int":int, "float":float, "str":str}

def objective_params(tree) -> dict:
    objectives_dict = {}
    objectiveList = tree.iter('objective')
    evaluationInfo = tree.find('evaluation')
    evaluationModule = import_module(evaluationInfo.findtext('module'))
    for objectiveNum, objective in enumerate(objectiveList):
        # Iterate over each objective and add to dictionary

        # convert all the arguments in the xml into their correct types
        o_args = {}
        args_dir = objective.find('args')
        if args_dir is not None:
            for i in args_dir.iterfind('arg'):
                o_args[i.findtext('name')] = valid_types[i.findtext('type')](i.findtext('value'))

        objectives_dict[objectiveNum] = {'name': objective.findtext('name'), 'weight': float(objective.findtext('weight')),
        'achievable': float(objective.findtext('achievable')), 'goal': float(objective.findtext('goal')),
        'evaluationFunction': getattr(evaluationModule, objective.findtext('evaluationFunction')),
        'lower': float(objective.findtext('lower')), 'upper': float(objective.findtext('upper')),
        'args': o_args}
    return objectives_dict

def selection_params(tree) -> dict:
    selections_dict = {}
    # Iterate over selection algorithms
    for selection in tree.iter('selection'):
        selection_dict = {}
        sel_name = selection.findtext('name')
        # Get given selection method and verify it is in the correct format
        try:
            selection_module = __import__('selection_methods')
            fun = getattr(selection_module, sel_name)
        except AttributeError:
            raise ValueError('selection method {} is not in selection_methods.py'.format(sel_name))
        if not callable(fun):
            raise ValueError('selection method {} is not a function'.format(sel_name))
        func_args = fun.__code__.co_varnames
        for arg in ['individuals', 'k']:
            if arg not in func_args:
                raise ValueError('selection method {} does not use required argument name {}'.format(sel_name, arg))

        selection_dict['fun'] = fun

        reserved_arg_names = ['individuals', 'k', 'dynamic_args']

        dynamic_args_str = selection.findtext('useDynamicArgs')
        dynamic_args = str2bool(dynamic_args_str)
        selection_dict['dynamic_args'] = dynamic_args
        if dynamic_args and 'dynamic_args' not in func_args:
            raise ValueError('selection method {} does not use argument name dynamic_args')

        argsDict = {}
        # Iterate over selection method arguments
        for arg in selection.iter('arg'):
            arg_name = arg.findtext('name')
            # Check whether function uses argument
            if arg_name not in func_args:
                raise ValueError('selection method {} does not use given argument name {}'.format(sel_name, arg))
            # Check whether argument name is reserved
            if arg_name in reserved_arg_names:
                raise ValueError('selection argument name cannot be {}'.format(arg_name))
            # Check if repeated argument names
            if arg_name in argsDict:
                raise ValueError('repeated selection argument name {}'.format(arg_name))
            # Evaluate given value
            arg_val_str = arg.findtext('val')
            try:
                arg_val = eval(arg_val_str)
            except TypeError:
                raise ValueError('could not interpret value {}'.format(arg_val_str))
            argsDict[arg_name] = arg_val
        selection_dict['args'] = argsDict
        selections_dict[sel_name] = selection_dict
    return selections_dict

def statistics_params(tree) -> dict:
    if not tree:
        return {}
    statisticsDict = {}
    for statistic in tree.iter('statistic'):
        statisticDict = {}
        stat_name = statistic.findtext('name')
        statisticDict['type'] = statistic.findtext('type')

        reserved_arg_names = ['individuals', 'dynamic_args']

        dynamic_args = statistic.findtext('useDynamicArgs')
        statisticDict['dynamicArgs'] = str2bool(dynamic_args)

        argsDict = {}
        argList = statistic.iter('arg')

        # Get given statistic method and verify it is in the correct format
        try:
            statistics_module = __import__('statistics')
            fun = getattr(statistics_module, stat_name)
        except AttributeError:
            raise ValueError('statistic method {} is not in statistics.py'.format(stat_name))
        if not callable(fun):
            raise ValueError('statistic method {} is not a function'.format(stat_name))
        func_args = fun.__code__.co_varnames
        for arg in ['individuals']:
            if arg not in func_args:
                raise ValueError('statistic method {} does not use required argument name {}'.format(stat_name, arg))

        # Iterate over statistic method arguments
        for arg in argList:
            arg_name = arg.findtext('name')
            # Check whether argument name is reserved
            if arg_name in reserved_arg_names:
                raise ValueError('statistic argument name cannot be {}'.format(arg_name))
            # Check if repeated argument names
            if arg_name in selectionArgsDict:
                raise ValueError('repeated statistic argument name {}'.format(arg_name))

            # Evaluate given value
            arg_val_str = arg.findtext('val')
            try:
                arg_val = eval(arg_val_str)
            except TypeError:
                raise ValueError('could not evaluate value {}'.format(arg_val_str))
            argsDict[arg_name] = arg_val
        statisticDict['args'] = argsDict
        statisticDict['fun'] = fun
        statisticsDict[stat_name] = statisticDict
    return statisticsDict

def evoluation_params(tree) -> dict:
    # Initializes the evolution dictionary
    evolutionParametersDict = {}
    # Adds evolution parameters to evolution dictionary
    evolutionParametersDict['initialPopulationSize'] = int(tree.findtext('initialPopulationSize'))
    evolutionParametersDict['elitePoolSize'] = int(tree.findtext('elitePoolSize'))
    evolutionParametersDict['launchSize'] = int(tree.findtext('launchSize'))
    evolutionParametersDict['minQueueSize'] = int(tree.findtext('minQueueSize'))
    evolutionParametersDict['outlierPenalty'] = float(tree.findtext('outlierPenalty'))

    # Adds mating dict parameters to evolution dictionary
    evolutionParametersDict['matingDict'] = {}
    matingList = tree.iter('mating')
    for mating in matingList:
        evolutionParametersDict['matingDict'][mating.findtext('name')] = float(mating.findtext('probability'))

    # Adds mutation dict parameters to evolution dictionary
    evolutionParametersDict['mutationDict'] = {}
    mutationList = tree.iter('mutation')
    for mutation in mutationList:
        evolutionParametersDict['mutationDict'][mutation.findtext('name')] = float(mutation.findtext('probability'))

    # Determine whether the problem is regression or classification
    regression = tree.findtext('regression')
    if regression:
        evolutionParametersDict['regression'] = str2bool(regression)

    # Load selection methods
    selections_dict = selection_params(tree)
    evolutionParametersDict['selection_dict'] = selections_dict

    return evolutionParametersDict

def misc_params(tree) -> dict:
    misc_params = {}
    seedFile = tree.find('seedFile').findtext('filename')
    genePoolFitnessOutput = tree.find('genePoolFitness').findtext('prefix')
    paretoFitnessOutput = tree.find('paretoFitness').findtext('prefix')
    paretoOutput = tree.find('paretoOutput').findtext('prefix')
    parentsOutput = tree.find('parentsOutput').findtext('prefix')
    memoryLimit = float(tree.find('evaluation').findtext('memoryLimit'))
    misc_params['seedFile'] = seedFile
    misc_params['genePoolFitness'] = genePoolFitnessOutput
    misc_params['paretoFitness'] = paretoFitnessOutput
    misc_params['paretoOutput'] = paretoOutput
    misc_params['parentsOutput'] = parentsOutput
    misc_params['memoryLimit'] = memoryLimit
    return misc_params

if __name__ == '__main__':
    # Takes an XML file to get dataset, objective, and evaluation information
    parser = ap.ArgumentParser()
    parser.add_argument(
        'filename', help='Input to EMADE, see inputSample.xml')
    parser.add_argument('-w', '--worker', dest='worker', default=False, action='store_true', help='Only run workers')

    args = parser.parse_args()
    inputFile = args.filename


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

    # Get python information
    python_info = root.find('pythonConfig')
    grid_python_command = python_info.findtext('gridPythonPath')
    local_python_command = python_info.findtext('localPythonCommand')
    slurm_worker_python_command = python_info.findtext('slurmPythonPathWorker')
    slurm_master_python_command = python_info.findtext('slurmPythonPathMaster')
    pace_python_command = python_info.findtext('pacePythonPath')

    # Initializes the dataset dictionary
    datasetDict = dataset_params(root)
    print('Dataset dict')
    print(datasetDict)
    print()

    # Initializes the objective dictionary
    objectiveDict = objective_params(root)
    print('Objectives dict')
    print(objectiveDict)
    print()

    # Initializes the statistics dictionary
    statisticsDict = statistics_params(root.find('statistics'))
    print('Stats dict')
    print(statisticsDict)
    print()

    # Initializes the evolution parameters dictionary
    evolution_dict = evoluation_params(root.find('evolutionParameters'))
    print('Evolution dict')
    print(evolution_dict)
    print()

    # Initializes the miscellanious dictionary
    misc_dict = misc_params(root)
    print('Misc dict')
    print(misc_dict)
    print()

    # Initializes the cache parameters dictionary
    cache_dict = cache_params(root)
    wait_time = cache_dict['masterWaitTime']
    print('Cache dict')
    print(cache_dict)
    print()

    # Get database information
    db_info = root.find('dbConfig')
    server = db_info.findtext('server')
    username = db_info.findtext('username')
    password = db_info.findtext('password')
    database = db_info.findtext('database')
    reuse = int(db_info.findtext('reuse'))

    pid_string = str(os.getpid())

    pickle_file_name = 'myPickleFile' + pid_string + '.dat'
    log_file_name = 'log_file' + pid_string + '.txt'
    with open(pickle_file_name, 'wb') as pickleFile:
        pickle.dump(evolution_dict, pickleFile)
        pickle.dump(objectiveDict, pickleFile)
        pickle.dump(datasetDict, pickleFile)
        pickle.dump(statisticsDict, pickleFile)
        pickle.dump(misc_dict, pickleFile)
        pickle.dump(cache_dict, pickleFile)

    didLaunch = False
    #First try to run with slurm:
    try:
        #first let's test for slurm (we don't need the .sh if there's no grid)
        subprocess.call("squeue") #this will print your current processes on the cluster
        print("Slurm cluster detected.")
        if not slurm_master_python_command or not slurm_worker_python_command:
            raise ValueError("You are on a Slurm cluster, but you have not defined the slurmPythonPathMaster or slurmPythonPathWorker in the XML. Exiting Emade")

        # Get hosts information
        engine_info = root.find('slurmEngineParameters')
        num_hosts = int(engine_info.findtext('numHosts'))
        workers_per_host = int(engine_info.findtext('workersPerHost'))
        job_name = engine_info.findtext('jobName')
        runtime = engine_info.findtext('runtime') # Runtime in D-HH:MM
        modules = engine_info.findtext('modules').split(" ")
        numGPUs = engine_info.findtext('numberOfGPUs')
        anacondaEnvironmentMaster = engine_info.findtext('anacondaEnvironmentMaster')
        anacondaEnvironmentWorker = engine_info.findtext('anacondaEnvironmentWorker')
        if numGPUs:
            numGPUs = int(numGPUs)
        else:
            numGPUs = 0
        hardware_request_string = engine_info.findtext('otherHardwareRequestString')
        mem_to_alloc = engine_info.findtext('memoryToAlloc')
        node = engine_info.findtext('specificNode')


        if not args.worker:
            # Create script to run on Son of Grid Engine cluster and call it using subprocess
            qsubFileNameMaster = 'slurmEngineJobSubmit_master' + pid_string + '.sh'
            with open(qsubFileNameMaster,'w') as qsubFile:
                qsubFile.write('#!/bin/bash\n')
                qsubFile.write('#SBATCH --job-name=' + job_name + "Master" + pid_string + "\n")
                if runtime:
                    qsubFile.write('#SBATCH -t ' + runtime + "\n")
                if node:
                    qsubFile.write('#SBATCH -w ' + node + "\n")
                qsubFile.write('#SBATCH -x ice[123-134] -x ice[137-140]' + "\n") # exclude machines which don't support AVX instructions
                qsubFile.write('#SBATCH --mem-per-cpu ' + str(mem_to_alloc) + "GB\n") #in gigs
                modules = engine_info.findtext('modules').split(" ")
                for module in modules:
                    qsubFile.write('module load ' + module + "\n")
                qsubFile.write('source activate ' + anacondaEnvironmentMaster +'\n')
                if not reuse:
                    qsubFile.write(slurm_master_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name + ' -ma -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + ' -nr\n')
                    # qsubFile.write(slurm_master_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name + ' -ma -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + '?charset=utf8' + ' -nr\n')
                else:
                    qsubFile.write(slurm_master_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name + ' -ma -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + ' -r\n')
                    # qsubFile.write(slurm_master_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name + ' -ma -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + '?charset=utf8' + ' -r\n')
                qsubFile.close()
                subprocess.call(['sbatch', qsubFileNameMaster])
                time.sleep(wait_time)
        qsubFileNameWorker = 'slurmEngineJobSubmit_worker' + pid_string + '.sh'
        with open(qsubFileNameWorker,'w') as qsubFile:
            qsubFile.write('#!/bin/bash\n')
            qsubFile.write('#SBATCH --job-name=' + job_name + "Worker" + pid_string + "\n")
            if node:
                qsubFile.write('#SBATCH -w ' + node + "\n")
            if runtime:
                qsubFile.write('#SBATCH -t ' + runtime + "\n")
            if numGPUs > 0:
                qsubFile.write('#SBATCH --gres=gpu:' + str(numGPUs) + "\n")
            if hardware_request_string:
                qsubFile.write('#SBATCH ' + hardware_request_string + "\n")
            # qsubFile.write('#SBATCH  -n ' + str(workers_per_host) + "\n") #number of tasks
            #qsubFile.write('#SBATCH -x ice[123-140]' + "\n") # exclude machines which don't support AVX instructions
            qsubFile.write('#SBATCH -x ice[123-134] -x ice[137-140]' + "\n") # exclude machines which don't support AVX instructions
            qsubFile.write('#SBATCH -c' + str(workers_per_host) + "\n") #cpus-per-task
            qsubFile.write('#SBATCH -N 1' + "\n") # Force to single machine
            qsubFile.write('#SBATCH --mem-per-cpu ' + str(mem_to_alloc) + "GB\n") #in gigs
            for module in modules:
                qsubFile.write('module load ' + module + "\n")
            #even if gpus =0, this is good to assign (as default is empty string:):
            gpus = ""
            for i in range(numGPUs):
                gpus += str(i) + ','
            gpus = gpus[:-1] #delete last comma
            # JPZ - Commenting out below line as slurm should handle this automatically
            #qsubFile.write('export CUDA_VISIBLE_DEVICES=' + str(gpus) + "\n")
            qsubFile.write('source activate ' + anacondaEnvironmentWorker +'\n')
            #on Grid it's sufficient to include the environment in the python path,
            #but without source activating tensorflow cannot find the necessary
            #file to compile CUDA sources
            qsubFile.write(slurm_worker_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name +  ' -n ' + str(workers_per_host) + ' -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + '\n')
            # qsubFile.write(slurm_worker_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name +  ' -n ' + str(workers_per_host) + ' -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + '?charset=utf8' + '\n')
            qsubFile.close()
            for i in range(num_hosts):
                subprocess.call(['sbatch', qsubFileNameWorker])
                time.sleep(15)
            didLaunch = True

    except (FileNotFoundError, PermissionError) as e:
        ##If this is reached, we're not on the head of a grid cluster
        pass

    try:
        #next let's test with pace 
        # raise FileNotFoundError # uncomment when testing
        subprocess.call("pace-check-queue")
        print("PACE detected.")
        if not pace_python_command:
            raise ValueError("You are on PACE, but you have not defined the pacePythonPath in the XML.")

        # Get host's information
        engine_info = root.find('paceEngineParameters')
        engine_queue = engine_info.findtext('queue')
        wall_time = engine_info.findtext('walltime')
        modules = engine_info.findtext('modules').split()
        num_hosts = int(engine_info.findtext('numHosts'))
        workers_per_host = int(engine_info.findtext('workersPerHost'))
        nodes_per_worker = 1
        processors_per_worker = workers_per_host
        ram_per_worker = engine_info.findtext('ramPerHostCPU')
        num_nodes_master = 1
        num_processors_master = 1
        ram_master = engine_info.findtext('ramPerMasterCPU')
        
        gpu = engine_queue == "pace-ice-gpu"
        if gpu and workers_per_host>1:
            print("GPU use was specified in input file, but workersPerHost was set to {}.\nChanging this value to 1 to prevent hardware competition amongst workers.".format(workers_per_host))
            workers_per_host = 1

        if not args.worker:
            # Create script to run on PACE cluster and call it using subprocess
            qsubFileNameMaster = 'paceEngineJobSubmit_master' + pid_string + '.sh'
            with open(qsubFileNameMaster,'w') as qsubFile:
                qsubFile.write('#PBS -q ' + engine_queue + '\n')
                qsubFile.write('#PBS -N EMADE_master\n')
                qsubFile.write('#PBS -l pmem={}\n'.format(ram_master))
                qsubFile.write('#PBS -l nodes={}:ppn={}\n'.format(num_nodes_master, num_processors_master))
                qsubFile.write('#PBS -l walltime={}\n'.format(wall_time))
                qsubFile.write('cd ' + os.getcwd() + '\n')
                for module in modules:
                    qsubFile.write("module load " + module + "\n")
                qsubFile.write("export CC=gcc\n")

                if not reuse:
                    qsubFile.write(pace_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name + ' -ma -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database +' -nr > master{}.out 2> master{}.err\n'.format(pid_string, pid_string))
                else:
                    qsubFile.write(pace_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name + ' -ma -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database +' -r > master{}.out 2> master{}.err\n'.format(pid_string, pid_string))
                qsubFile.close()
                subprocess.call(['qsub', qsubFileNameMaster])
                time.sleep(wait_time)

        qsubFileNameWorker = 'paceEngineJobSubmit_worker' + pid_string + '.sh'
        with open(qsubFileNameWorker,'w') as qsubFile:
            qsubFile.write('#PBS -q ' + engine_queue + '\n')
            qsubFile.write('#PBS -N EMADE_worker\n')
            qsubFile.write('#PBS -l pmem={}\n'.format(ram_per_worker))
            if gpu:
                qsubFile.write('#PBS -l nodes={}:ppn={}:gpus=1:exclusive_process\n'.format(nodes_per_worker, processors_per_worker))
            else:
                qsubFile.write('#PBS -l nodes={}:ppn={}\n'.format(nodes_per_worker, processors_per_worker))
            qsubFile.write('#PBS -l walltime={}\n'.format(wall_time))
            if gpu:
                qsubFile.write('#PBS -l naccesspolicy=UNIQUEUSER\n') # Empirically, I've found that this does what we expect the exclusive_process flag to do (leaving them both in though)
            qsubFile.write("cd " + os.getcwd() + "\n")
            for module in modules:
                qsubFile.write("module load " + module + "\n")
            qsubFile.write("export CC=gcc\n")
            
            qsubFile.write(pace_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name +  ' -n ' + str(workers_per_host) + ' -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + '\n')
            qsubFile.close()
            for i in range(num_hosts):
                subprocess.call(['qsub', qsubFileNameWorker])
                # Sleep 5 seconds between submits. This should help to avoid database collision at startup
                time.sleep(5)
            didLaunch = True

    except (FileNotFoundError, PermissionError) as e:
        ##If this is reached, we're not on pace
        pass

    try:
        # next let's test for grid (we don't need the .sh if there's no grid)
        subprocess.call("qhost")
        #note that this prints current jobs to stdout
        print("Grid cluster detected.")
        if not grid_python_command:
            raise ValueError("You are on a Grid cluster, but you have not defined the gridPythonPath in the XML.")

        # Get hosts information
        engine_info = root.find('gridEngineParameters')
        num_hosts = int(engine_info.findtext('numHosts'))
        workers_per_host = int(engine_info.findtext('workersPerHost'))
        engine_project = engine_info.findtext('project')
        engine_queue = engine_info.findtext('queue')
        engine = engine_info.findtext('parallelEnv')
        if not args.worker:
            # Create script to run on Son of Grid Engine cluster and call it using subprocess
            qsubFileNameMaster = 'gridEngineJobSubmit_master' + pid_string + '.sh'
            with open(qsubFileNameMaster,'w') as qsubFile:
                #qsubFile.write('#$ -l h_rt=300\n')
                #qsubFile.write('#$ -ar 2\n')
                qsubFile.write('#$ -R y\n')
                # Currently forced to 3 due to allocation rule, look in to this
                qsubFile.write('#$ -pe ' + engine  + ' ' + str(3) + '\n')
                qsubFile.write('#$ -q ' + engine_queue + '\n')
                qsubFile.write('#$ -P ' + engine_project + '\n')
                qsubFile.write('#$ -cwd\n')
                qsubFile.write('#$ -notify\n')
                qsubFile.write('#$ -l h_rt=480:00:00\n')
                #qsubFile.write('echo `which python`')
                #qsubFile.write(grid_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name + ' -ma\n')
                if not reuse:
                    qsubFile.write(grid_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name + ' -ma -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database +  ' -nr\n')
                    # qsubFile.write(grid_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name + ' -ma -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + '?charset=utf8' + ' -nr\n')
    
                else:
                    qsubFile.write(grid_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name + ' -ma -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + ' -r\n')
                    # qsubFile.write(grid_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name + ' -ma -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + '?charset=utf8' + ' -r\n')
 
                qsubFile.close()
                subprocess.call(['qsub', qsubFileNameMaster])
                time.sleep(wait_time)
        qsubFileNameWorker = 'gridEngineJobSubmit_worker' + pid_string + '.sh'
        with open(qsubFileNameWorker,'w') as qsubFile:
            #qsubFile.write('#$ -l h_rt=300\n')
            #qsubFile.write('#$ -ar 2\n')
            qsubFile.write('#$ -R y\n')
            # Currently forced to 3 due to allocation rule, look in to this
            qsubFile.write('#$ -pe ' + engine  + ' ' + str(workers_per_host) + '\n')
            qsubFile.write('#$ -q ' + engine_queue + '\n')
            qsubFile.write('#$ -P ' + engine_project + '\n')
            qsubFile.write('#$ -cwd\n')
            qsubFile.write('#$ -notify\n')
            qsubFile.write('#$ -l h_rt=480:00:00\n')
            #qsubFile.write('echo `which python`')
            #python src/GPFramework/didLaunch.py myPickleFile9452.dat -n 4 -d sqlite:///EMADE_07-21-2017_10-18-41.db
            #qsubFile.write(grid_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name +  ' -n ' + str(workers_per_host) + ' -d  sqlite:///' + emade_path + '/' + db_file + '\n')
            qsubFile.write(grid_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name +  ' -n ' + str(workers_per_host) + ' -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database  + '\n')
            # qsubFile.write(grid_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name +  ' -n ' + str(workers_per_host) + ' -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + '?charset=utf8' + '\n')
            
            #subprocess.call(['chmod', '+x', qsubFileNameWorker])
            qsubFile.close()
            #subprocess.call(['chmod', '+x', qsubFileName])
            for i in range(num_hosts):
                subprocess.call(['qsub', qsubFileNameWorker])
                # Sleep 5 seconds between submits. This should help to avoid database collision at startup
                time.sleep(5)
            didLaunch = True

    except (FileNotFoundError, PermissionError) as e:
        #If this is reached, we're not on the head of a grid cluster
        pass



    if not didLaunch:
        #we default to a local run if grid and slurm are not detected:
        # Get hosts information
        engine_info = root.find('localRunParameters')
        workers_per_host = int(engine_info.findtext('workersPerHost'))

        if not args.worker:
            def runMaster(masterCMD, id):
                import subprocess
                sout = open("master" + id + ".out", "w")
                serr = open("master" + id + ".err", "w")
                masterProcessHandle = subprocess.Popen(masterCMD.split(), stdout= sout, stderr =serr )
            if not reuse:
                masterCMD = local_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name + ' -ma -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + '?charset=utf8' + ' -nr\n'
            else:
                masterCMD = local_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name + ' -ma -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + '?charset=utf8' + ' -r\n'
            m = multiprocess.Process(target=runMaster, args=(masterCMD, str(os.getpid())))
            m.start()
            print("Master process # " + str(os.getpid()) + " began: ", masterCMD)
            # Now let's wait for the database to appear
            #note this sleep could be eliminated by removing the SQL preparations to
            # a new subprocess
            print("Waiting for database connection.")
            time.sleep(wait_time)

        workerCMD =  local_python_command + ' src/GPFramework/didLaunch.py ' + pickle_file_name +  ' -n ' + str(workers_per_host) + ' -d mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database + '?charset=utf8' + '\n'


        def runWorker(workerCMD, id):
            import subprocess
            sout = open("worker" + id + ".out", "w")
            serr = open("worker" + id + ".err", "w")
            workerProcessHandle = subprocess.Popen(workerCMD.split(), stdout = sout, stderr = serr)



        w = multiprocess.Process(target=runWorker, args=(workerCMD, str(os.getpid())))

        w.start()

        #multiple workers for testing
        print("Worker process # " + str(os.getpid()) + " began: ", workerCMD)

        while True:
            #we could just run the processes as daemons, but
            # this allows the user to keep the parent processes
            #alive and therefore exit emade with a keyboard
            #interrupt (e.g. can avoid orphaned
            #processes on headless systems)
            time.sleep(60)
            print("Emade parent process is running. Check logs on disk for updates on master and worker processes.")
