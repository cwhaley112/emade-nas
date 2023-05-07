"""
Programmed by Jason Zutty
Modified by VIP Team
This module includes a number of custom genetic operators for EMADE
"""
from collections import defaultdict
import hashlib
from inspect import isclass
import random
import re
import time
from deap import gp
import dill
import numpy as np
__type__ = object
import GPFramework
import GPFramework.neural_network_methods as nnm
import sys
import copy as cp

##################################################################
##################################################################
################ General Mutation and Mating Methods #############
##################################################################
##################################################################

def simulateMateMut(offspring, gen_dict, CXPB, OUTP, MUTPB, MUTLPB, MUTEPHPB, MUTNRPB, MUTUPB, MUTSPB):
    print('Mating offspring')
    sys.stdout.flush()

    # Call mate on standard mating method.   Operates in place
    count = mate(offspring, gen_dict["toolbox"].mate, gen_dict, CXPB, OUTP)
    print('Mated ' + str(count) + ' individuals')
    sys.stdout.flush()

    #print('Mating offspring Ephemerals')
    #sys.stdout.flush()
    ## Call mate on the mating ephemeral method, operates in place
    #count = mate(offspring,  gen_dict["toolbox"].mateEphemeral, CXPB_ephemeral, OUTP)
    #print('Mated ' + str(count) + ' individuals ephermerals')
    #sys.stdout.flush()

    ## Headless chicken mating is a crossover with a randomly generated individual
    #print('Mating offspring headless chicken')
    #sys.stdout.flush()
    #count = mate_headless(offspring,  gen_dict["toolbox"].mate, CXPB_headless_chicken,  gen_dict["toolbox"].individual())
    #print('Mated ' + str(count) + ' individuals')
    #sys.stdout.flush()

    #print('Mating offspring Ephemerals headless chicken')
    #sys.stdout.flush()
    #count = mate_headless(offspring,  gen_dict["toolbox"].mateEphemeral, CXPB_headless_chicken_ephemeral,  gen_dict["toolbox"].individual())
    #print('Mated ' + str(count) + ' individuals ephermerals')
    #sys.stdout.flush()


    print('Mutating offspring (Insertion)')
    sys.stdout.flush()
    count = mutate(offspring,  gen_dict["toolbox"].mutate, gen_dict, MUTPB, needs_pset=True)
    print('Mutated ' + str(count) + ' individuals')
    sys.stdout.flush()

    print('Mutating offspring (Insert ModLearner)')
    sys.stdout.flush()
    count = mutate(offspring,  gen_dict["toolbox"].mutateLearner, gen_dict, MUTLPB, needs_pset=True)
    print('Mutated ' + str(count) + ' individuals')
    sys.stdout.flush()

    print('Mutating offspring (Ephemeral)')
    sys.stdout.flush()
    count = mutate(offspring,  gen_dict["toolbox"].mutateEphemeral, gen_dict, MUTEPHPB)
    print('Mutated ' + str(count) + ' individuals')
    sys.stdout.flush()

    print('Mutating offspring (Node Replace)')
    sys.stdout.flush()
    count = mutate(offspring,  gen_dict["toolbox"].mutateNodeReplace, gen_dict, MUTNRPB, needs_pset=True)
    print('Mutated ' + str(count) + ' individuals')
    sys.stdout.flush()

    print('Mutating offspring (Uniform)')
    sys.stdout.flush()
    count = mutate(offspring,  gen_dict["toolbox"].mutateUniform, gen_dict, MUTUPB, needs_pset=True, needs_expr=True)
    print('Mutated ' + str(count) + ' individuals')
    sys.stdout.flush()

    print('Mutating offspring (Shrink)')
    sys.stdout.flush()
    count = mutate(offspring,  gen_dict["toolbox"].mutateShrink, gen_dict, MUTSPB)
    print('Mutated ' + str(count) + ' individuals')
    sys.stdout.flush()

    return offspring

def simulateMateMutNN(offspring, gen_dict, CXPB, OUTP, MUTPB, MUTLPB, MUTEPHPB, MUTNRPB, MUTUPB, MUTSPB, isModule=False, isNNLearner=False):
    print('Mating offspring')
    sys.stdout.flush()

    gen_dict["isNNLearner"] = isNNLearner
    gen_dict["isModule"] = isModule

    # # These mutation methods seem kinda useless in terms of actual effect on NNlearners.
    # # It just swaps their nnlayerlist or their entire object.
    if (isModule):
        print('Mating offspring NNMiddle')
        sys.stdout.flush()
        # Call mate on the mating ephemeral method, operates in place
        count = mate(offspring,  gen_dict["toolbox"].mateNNMiddle, gen_dict, CXPB, OUTP)
        print('Mated ' + str(count) + ' individuals ephermerals')
        sys.stdout.flush()

        print('Mating offspring NNFull')
        sys.stdout.flush()
        # Call mate on the mating ephemeral method, operates in place
        count = mate(offspring,  gen_dict["toolbox"].mateNNFull, gen_dict, CXPB, OUTP)
        print('Mated ' + str(count) + ' individuals ephermerals')
        sys.stdout.flush()
    
    
    print('Mutating offspring (Add Layer)')
    sys.stdout.flush()
    count = mutate(offspring,  gen_dict["toolbox"].mutateAddLayer, gen_dict, MUTPB, needs_pset=True, isNNLearner=True)
    print('Mutated ' + str(count) + ' individuals')
    sys.stdout.flush()

    print('Mutating offspring (Remove Layer)')
    sys.stdout.flush()
    count = mutate(offspring,  gen_dict["toolbox"].mutateRemoveLayer, gen_dict, MUTPB, needs_pset=True, isNNLearner=True)
    print('Mutated ' + str(count) + ' individuals')
    sys.stdout.flush()

    print('Mutating offspring (Modify Layer)')
    sys.stdout.flush()
    count = mutate(offspring,  gen_dict["toolbox"].mutateModifyLayer, gen_dict, MUTPB, needs_pset=True, isNNLearner=True)
    print('Mutated ' + str(count) + ' individuals')
    sys.stdout.flush()

    #print('Mutating offspring (Shuffle Layer)')
    #sys.stdout.flush()
    #count = mutate(offspring,  gen_dict["toolbox"].mutateShuffleLayer, MUTPB)
    #print('Mutated ' + str(count) + ' individuals')
    #sys.stdout.flush()

    print('Mutating offspring (Swap Layer)')
    sys.stdout.flush()
    count = mutate(offspring,  gen_dict["toolbox"].mutateSwapLayer, gen_dict, MUTPB, needs_pset=True, isNNLearner=True)
    print('Mutated ' + str(count) + ' individuals')
    sys.stdout.flush()
    
    print('Mutating offspring (Modify Activations)')
    sys.stdout.flush()
    count = mutate(offspring,  gen_dict["toolbox"].mutateActivationParam, gen_dict, MUTPB, needs_pset=True, isNNLearner=True)
    print('Mutated ' + str(count) + ' individuals')
    sys.stdout.flush()
    

    print('Mutating offspring (Modify Optimizers)')
    sys.stdout.flush()
    count = mutate(offspring,  gen_dict["toolbox"].mutateOptimizerParam, gen_dict, MUTPB, needs_pset=True, isNNLearner=True)
    print('Mutated ' + str(count) + ' individuals')
    sys.stdout.flush()
    
    print('Mutating offspring (Modify Weight Initializers)')
    sys.stdout.flush()
    count = mutate(offspring,  gen_dict["toolbox"].mutateWeightInitializerParam, gen_dict, MUTPB, needs_pset=True, isNNLearner=True)
    print('Mutated ' + str(count) + ' individuals')
    sys.stdout.flush()

    return offspring

def mate(offspring, mating_function, mut_dict, CXPB, OUTP):
    """Iterates through a list of offspring and applies the given mating type
    With the given crossover probability

    Args:
        offspring: List of individuals
        mating_function: Function to mate
        CXPB: Probability to mate
    Returns:
        Number of indisviduals modified
    """
    count = 0
    if (mut_dict["isModule"]):
        temp_pset = [mut_dict["mod_pset"]]
    else:
        temp_pset = [mut_dict["pset"]]
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        child1_hash = my_hash(child1)
        child2_hash = my_hash(child2)
        mated = False
        for tree1, tree2, pset in zip(child1, child2, temp_pset):
            if random.random() < CXPB:
                mating_function(tree1, tree2, pset, mut_dict)
                count += 2
                mut_dict["tries"] = 0
                mut_dict["prev_nodes"] = []
                clean_individual(child1, mut_dict)
                clean_individual(child2, mut_dict)
                if not hasattr(child1, 'parents'):
                    child1.parents = []
                if not hasattr(child2, 'parents'):
                    child2.parents = []

                child1.parents, child2.parents = child1.parents + child2.parents, child2.parents + child1.parents

                if not hasattr(child1, 'intermediate_parent') or not child1.intermediate_parent:
                    child1.parents.append(child1_hash)
                    child2.parents.append(child1_hash)
                if not hasattr(child2, 'intermediate_parent') or not child2.intermediate_parent:
                    child1.parents.append(child2_hash)
                    child2.parents.append(child2_hash)
                child1.parents = list(set(child1.parents))
                child2.parents = list(set(child2.parents))

                child1.intermediate_parent = True
                child2.intermediate_parent = True
    return count


# def mate_headless(offspring, mating_function, mut_dict, CXPB_headless_chicken, rand_ind):
#     """Iterates through a list of offspring and applies the given mating type
#     With the given crossover probability
#     Generates random individual for crossover
#     Args:
#         offspring: List of individuals
#         mating_function: Function to mate
#         CXPB: Probability to mate
#     Returns:
#         Number of individuals modified, modified offspring list
#     """
#     count = 0
#     new_individuals = []
#     for child1 in offspring:
#         child2 = rand_ind # randomly generated individual
#         mated = False
#         for tree1, tree2 in zip(child1, child2):
#             if random.random() < 2:
#                 mating_function(tree1, tree2)
#                 mated = True
#         if mated:
#             mut_dict["tries"] = 0
#             mut_dict["prev_nodes"] = []
#             clean_individual(child1, mut_dict)
#             clean_individual(child2, mut_dict)
#             count += 2
#             new_individuals.append(child2)
#     offspring += new_individuals
#     return count


def mutate(offspring, mutate_function, mut_dict, MUTPB, needs_pset=False, needs_expr=False, isNNLearner=False, isModule=False):
    """Iterates through a list of offspring and applies the given mutation
    With the given probability

    Args:
        offspring: List of individuals
        mating_function: Function to mate
        MUTPB: mutation probability

    Returns:
        Number of individuals modified
    """
    count = 0
    if (isModule):
        temp_pset = [mut_dict["mod_pset"]]
    else:
        temp_pset = [mut_dict["pset"]]
    for mutant in offspring:
        for tree, pset, expr in zip(mutant, temp_pset, mut_dict["exprs"]):
            if random.random() < MUTPB:
                mutant_hash = my_hash(mutant)
                if needs_pset and needs_expr and isNNLearner:
                    mutate_function(tree, getattr(mut_dict["toolbox"], expr), pset, mut_dict)
                elif needs_pset and needs_expr:
                    mutate_function(tree, getattr(mut_dict["toolbox"], expr), pset)
                elif needs_pset and isNNLearner:
                    mutate_function(tree, pset, mut_dict)
                elif needs_pset:
                    mutate_function(tree, pset)
                elif isNNLearner:
                    mutate_function(tree, mut_dict)
                else:
                    mutate_function(tree)
                mut_dict["tries"] = 0
                mut_dict["prev_nodes"] = []
                clean_individual(mutant, mut_dict)
                if not hasattr(mutant, 'parents'):
                    mutant.parents = []
                if not hasattr(mutant, 'intermediate_parent') or not mutant.intermediate_parent:
                    mutant.parents.append(mutant_hash)
                mutant.parents = list(set(mutant.parents))
                mutant.intermediate_parent = True  # this mutant is now an intermediate parent, will be set to False when added to NNStatistics table
                count += 1
                break
    return count

##################################################################
##################################################################
############### Specific Mutation and Mating Methods #############
##################################################################
##################################################################

def my_str(individual, name=None):
    """Return the string representation of an individual

    Args:
        individual: The individual in question

    Returns:
        the string representation of an individual
    """
    # First we get the string of the main tree
    my_string = str(individual[0])

    if (name):
        my_string = name + ": " + my_string

    if (not name):
        # Next check to see if that tree uses any ADF's
        mod_matches = re.findall(r"mod_(\d+)", my_string)

        # Get the int representing the index from the back of the array e.g. [main, 2, 1, 0]
        mod_matches = np.unique([int(mod_match) for mod_match in mod_matches])

        loaded=False
        while not loaded:
            try:
                with open("Global_MODS", "rb") as mod_file:
                    glob_mods = dill.load(mod_file)
                loaded=True
            except EOFError:
                time.sleep(15)

        while len(mod_matches) > 0:
            for mod_match_num in mod_matches:
                my_string += '\nmod_' + str(mod_match_num) + ': ' + str(glob_mods["mod_" + str(mod_match_num)][1][0])
            mod_matches = re.findall(r"mod_(\d+)", str(glob_mods["mod_" + str(mod_match_num)][1][0]))
            mod_matches = np.unique([int(mod_match) for mod_match in mod_matches if 'mod_' + mod_match + ': ' not in my_string])

    return my_string

def my_hash(individual):
    """Return the SHA256 hash of an individual

    Returns:
        the SHA256 hash of individual
    """
    return hashlib.sha256(my_str(individual).encode()).hexdigest()

def clean_individual(individual, clean_dict):
    """
    This method restores all the tracked progress on an individual
    that it may have inheritted from its parent.
    This method works inplace and does not return the individual, but this can be changed
    in the future if we ever need to decorate or something.
    """
    del individual.fitness.values
    for dataset in clean_dict["datasetDict"]:
        del getattr(individual, clean_dict["datasetDict"][dataset]['name']).values
    for objective in clean_dict["objectivesDict"]:
        setattr(individual, clean_dict["objectivesDict"][objective]['name'], np.ndarray(0))

    individual.retry_time = 0
    individual.elapsed_time = 0
    individual.age = 0
    individual.novelties = []

def insert_adf(individual, pset):
    """Taken from deap mutInsert

    Args:
        individual: individual to mutate
        pset: primitive set

    Returns:
        tuple containing new individual
    """
    # We only care about nodes that have return type GPFramework.data.EmadeDataPair
    # Because an ADF has been set up for this type
    # data_pair_indices = [i for i in np.arange(len(individual)) if isinstance(individual[i].ret, GPFramework.data.EmadeDataPair)]

    index = random.randrange(len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)
    choice = random.choice

    # As we want to keep the current node as children of the new one,
    # it must accept the return value of the current node
    primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args]

    if len(primitives) == 0:
        return individual,

    new_node = choice(primitives)
    new_subtree = [None] * len(new_node.args)
    position = choice([i for i, a in enumerate(new_node.args) if a == node.ret])

    for i, arg_type in enumerate(new_node.args):
        if i != position:
            term = choice(pset.terminals[arg_type])
            if isclass(term):
                term = term()
            new_subtree[i] = term

    new_subtree[position:position + 1] = individual[slice_]
    new_subtree.insert(0, new_node)
    individual[slice_] = new_subtree
    return individual,
    pass

def insert_modifyLearner(individual, pset):
    """Taken from deap mutInsert

    Args:
        individual: individual to mutate
        pset: primitive set

    Returns:
        tuple containing new individual
    """
    index = random.randrange(len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)
    choice = random.choice
    plist = ['ModifyLearnerInt', 'ModifyLearnerBool',
                'ModifyLearnerFloat', 'ModifyLearnerList']

    # As we want to keep the current node as children of the new one,
    # it must accept the return value of the current node
    primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args and p.name in plist]

    if len(primitives) == 0:
        return individual,

    new_node = choice(primitives)
    new_subtree = [None] * len(new_node.args)
    position = choice([i for i, a in enumerate(new_node.args) if a == node.ret])

    for i, arg_type in enumerate(new_node.args):
        if i != position:
            term = choice(pset.terminals[arg_type])
            if isclass(term):
                term = term()
            new_subtree[i] = term

    new_subtree[position:position + 1] = individual[slice_]
    new_subtree.insert(0, new_node)
    individual[slice_] = new_subtree
    return individual,

def mut_terminal(individual, pset, mode):
    """This operators works on the constants (terminals) of the tree individual.
       It will change the value of one of the terminals in the individual
       by randomly choosing another terminal of the same type if one exists

    Args:
        individual: individual to mutate
        pset: primitive set
        mode: a bool to indicate to change "one" or "all"
              ephemeral constants. 1 == "one" and 0 == "all"

    Returns:
        tuple containing new individual
    """
    terminals_idx = [index
                      for index, node in enumerate(individual)
                      if isinstance(node, gp.Terminal)]
    
    if len(terminals_idx) > 0:
        if mode:
            terminals_idx = (random.choice(terminals_idx),)
            
        for i in terminals_idx:
            # find every terminal of the same type excluding the terminal already in that position
            filtered_terminals = [x for x in pset.terminals[type(individual[i])] if x != individual[i]]
            
            if len(filtered_terminals) > 0:
                # randomly choose new terminal of the same type
                term = random.choice(filtered_terminals)
                if isclass(term):
                    term = term()
                individual[i] = term
    
    return individual,

def cx_ephemerals(ind1, ind2):
    """Randomly select in each individual and exchange each subtree with the
    point as root between each individual.

    Args:
        ind1: first tree participating in the crossover
        ind2: second tree participating in the crossover

    Returns:
        a tuple of two trees
    """

    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        #JZ: I need to fix this part I think, but not important  for STGP
        # Not STGP optimization
        types1[__type__] = range(1, len(ind1))
        types2[__type__] = range(1, len(ind2))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            if isinstance(node, gp.Ephemeral):
                types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            if isinstance(node, gp.Ephemeral):
                types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))
    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2

def adjust_for_arity(population):
    """
    Given population with calculated fitness scores,
    Remove bias towards solutions with fewer variations

    Args:
        population: population to adjust

    Returns:
        adjusted population
    """
    for individual in population:
        count = 0
        for node in individual:
            if isinstance(node, gp.Ephemeral):
                count += 1
            elif isinstance(node, gp.Terminal):
                count += 1
        # An individual with a lot of terminals will result in
        # many variations on the Pareto front, and thus a lower crowding distance.
        # This will counteract that.
        individual.fitness.crowding_dist *= count

    return population


def mut_terminal_type_based(individual, pset, mut_dict, sel_type=None, prob_choose=0.5):
    """This operators works on the constants (terminals) of the tree individual.
       It will change the value of one of the terminals in the individual
       by randomly choosing another terminal of the same type if one exists

    Args:
        individual: individual to mutate
        pset: primitive set
        type: a bool to indicate to change "one" or "all"
              ephemeral constants. 1 == "one" and 0 == "all"

    Returns:
        tuple containing new individual
    """
    terminals_idx = [index
                      for index, node in enumerate(individual)
                      if (isinstance(node, gp.Terminal) and node.ret == sel_type) ]
    if len(terminals_idx) > 0:
        for i in terminals_idx:
            # find every terminal of the same type excluding the terminal already in that position
            filtered_terminals = [x for x in pset.terminals[sel_type] if x != individual[i]]
            print(filtered_terminals)
            if len(filtered_terminals) > 0 and random.random() >= prob_choose:
                # randomly choose new terminal of the same type
                term = random.choice(filtered_terminals)
                if isclass(term):
                    term = term()
                individual[i] = term
    
    return individual,



skip_layers = ["InputLayer", "OutputLayer","EmbeddingLayer", "PretrainedEmbeddingLayer", "BERTInputLayer", "InitLayerList", "VGGInputLayer", "MobileNetInputLayer", "InceptionInputLayer"]
skip_layers_add_function = ["InputLayer", "EmbeddingLayer", "PretrainedEmbeddingLayer", "BERTInputLayer", "InitLayerList", "VGGInputLayer", "MobileNetInputLayer", "InceptionInputLayer"]

def add_layer(individual, orig_pset, mut_dict):
    """
    Add a layer to the individual at a random  position

    Args:
        individual: tree to add layer
        pset: primitive set

    Returns:
        new individual with added layer
    """
    print(individual)
    print(len(individual))
    if len(individual) < 2:
        return individual,

    pset = cp.deepcopy(orig_pset)
    # look for nodes where a layer can be inserted
    # layer_bool = [node.ret == LayerList and isinstance(node, gp.Primitive) and node.name not in skip_layers_add_function for node in individual]
    layer_bool = [issubclass(node.ret, nnm.LayerList) and isinstance(node, gp.Primitive) and node not in mut_dict["prev_nodes"] for node in individual]
    layer_inds = np.where(layer_bool)[0].tolist()
    if layer_inds != []:
        index = random.choice(layer_inds) # pick random node where insert can happen
        node = individual[index]
    
        # As we want to keep the current node as children of the new one,
        # it must accept the return value of the current node
        #primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args and p.name == "ConcatenateLayer3"]
        primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args]
    
        if len(primitives) == 0:
            return individual,

        if len(mut_dict["prev_nodes"]) == len(primitives):
            return individual,
    
        new_node = random.choice(primitives)

        print("add node")
        print(new_node.name)
        new_subtree = [new_node]
        arg_list = [i for i, type_ in enumerate(node.args) if type_ == node.ret]
        if len(arg_list)>0:
            arg_idx = random.choice(arg_list)
        else:
            return individual,
        rindex = index + 1
        for _ in range(arg_idx + 1):
            rslice = individual.searchSubtree(rindex)
            subtree = individual[rslice]
            rindex += len(subtree)

        for i, arg_type in enumerate(new_node.args[:-1]):
           # if arg_type==GPFramework.neural_network_methods.LayerList:
           #     list_choices = [node for node in pset.primitives[GPFramework.neural_network_methods.LayerList] if node.name == "EmbeddingLayer"]
           #     enode = list_choices[0]
           #     new_subtree.append(enode)
           #     for earg in enode.args[:-1]:
           #         list_choices = [node for node in pset.terminals[earg] if type(node) == gp.Terminal]
           #         new_subtree.append(random.choice(list_choices)) # need to ensure list_choices is not empty
           #     list_choices = [node for node in pset.primitives[GPFramework.neural_network_methods.LayerList] if node.name == "InputLayer"] 
           #     new_subtree.append(list_choices[0])
           # else:
            print(i, arg_type) # have seen an error here before
            term = random.choice(pset.terminals[arg_type])
            if isclass(term):
                term = term()
            new_subtree.append(term)
    
        #new_subtree[position:position + 1] = individual[slice_]
        #new_ind_list = individual[:slice_.start] + new_subtree + individual[slice_.stop:]
        
        # Checks for validity of nnlearners or modules
        try:
            temp_ind = gp.PrimitiveTree(individual[2:rslice.start] + new_subtree + subtree)
            func = gp.compile(temp_ind, pset)
            for dataset in mut_dict["datasetDict"]:
                func = func(nnm.InputLayer())
                layers, _ = nnm.compile_layerlist(func, [], mut_dict["data_dim"], mut_dict["datasetDict"][dataset]['type'], isNNLearner=mut_dict['isNNLearner'])
        except Exception as e:
            print(e)
            mut_dict["tries"] += 1
            mut_dict["prev_nodes"].append(new_node)
            return add_layer(individual, orig_pset, mut_dict)
        
        new_ind = gp.PrimitiveTree(individual[:rslice.start] + new_subtree + subtree + individual[rslice.stop:])
        oldbranch = subtree 
        individual.clear()
        individual.__init__(new_ind)
        print("after make")
        print(individual)
    return individual,

def remove_layer(individual, orig_pset, mut_dict):
    """
    Randomly removes a non-IO layer from an NNLearner 
    :param individual: The tree to be shrinked.
    :returns: A tuple of one tree.
    """
    iprims = []
    orig_individual = cp.deepcopy(individual)
    pset = cp.deepcopy(orig_pset)
    print(individual)
    if len(individual) <= 2:
        return individual,
    for i, node in enumerate(individual[1:], 1):
        if not isinstance(node, gp.Primitive):
            continue
        if issubclass(node.ret, nnm.LayerList) and not issubclass(node.ret, nnm.LayerList2dim) and issubclass(node.args[-1], nnm.LayerList) and i not in mut_dict["prev_nodes"]: # omit layer terminals (inputs) and layers that can function as output layers (LL2dim)
            iprims.append((i, node))
    if len(iprims) != 0:
        
        index, prim = random.choice(iprims)
        arg_list = [i for i, type_ in enumerate(prim.args) if type_ == prim.ret]
        if len(arg_list)>0:
            arg_idx = random.choice(arg_list)
        else:
            return individual,
        rindex = index + 1
        for _ in range(arg_idx + 1):
            rslice = individual.searchSubtree(rindex)
            subtree = individual[rslice]
            rindex += len(subtree)

        slice_ = individual.searchSubtree(index)
        individual[slice_] = subtree

        try:
            temp_ind = gp.PrimitiveTree(individual[2:rslice.start-1] + subtree)
            func = gp.compile(temp_ind, pset)
            for dataset in mut_dict["datasetDict"]:
                func = func(nnm.InputLayer())
                layers, _ = nnm.compile_layerlist(func, [], mut_dict["data_dim"], mut_dict["datasetDict"][dataset]['type'], isNNLearner=mut_dict['isNNLearner'])
        except Exception as e:
            print(e)
            mut_dict["tries"] += 1
            mut_dict["prev_nodes"].append(index)
            return remove_layer(orig_individual, pset, mut_dict)

    return individual,

def swap_layer(individual, orig_pset, mut_dict):
    """
    Randomly replaces a layer from an NNLearner with another layer  
    :param individual: The tree to be shrinked.
    :returns: A tuple of one tree.
    """
    iprims = []
    print(individual)
    if len(individual) < 2:
        return individual,

    pset = cp.deepcopy(orig_pset)
    for i, node in enumerate(individual[1:], 1):
        if not isinstance(node, gp.Primitive):
            continue
        if issubclass(node.ret, nnm.LayerList):
            iprims.append((i, node))
    if len(iprims) != 0:        
        index, prim = random.choice(iprims) 

        #pick random primitive with the same return type as prim
        nnode_list = [node for node in pset.primitives[prim.ret] if issubclass(node.args[-1], nnm.LayerList)] # Should skip layers that don't take layerlists (Embedding Layer)
        if len(nnode_list) > 0:
            new_node = random.choice(nnode_list)
        else:
            return individual,
        new_subtree = [new_node] 
       
        print("new node")
        print(new_node.name)
        for arg in new_node.args[:-1]:
            
            list_choices = [node for node in pset.terminals[arg] if type(node) == gp.Terminal]
        
            term = random.choice(pset.terminals[arg])
            if isclass(term):
                term = term()
            new_subtree.append(term)

        arg_list = [i for i, type_ in enumerate(prim.args) if type_ == prim.ret]
        if len(arg_list)>0:
            arg_idx = random.choice(arg_list)
        else:
            return individual,
        rindex = index + 1
        for _ in range(arg_idx + 1):
            rslice = individual.searchSubtree(rindex)
            subtree = individual[rslice]
            rindex += len(subtree)
        
        
        slice_ = individual.searchSubtree(index)

        # Checks for validity of nnlearners or modules
        try:
            temp_ind = gp.PrimitiveTree(individual[2:slice_.start] + new_subtree + subtree)
            func = gp.compile(temp_ind, pset)
            for dataset in mut_dict["datasetDict"]:
                func = func(nnm.InputLayer())
                layers, _ = nnm.compile_layerlist(func, [], mut_dict["data_dim"], mut_dict["datasetDict"][dataset]['type'], isNNLearner=mut_dict['isNNLearner'])
        except Exception as e:
            print(e)
            mut_dict["tries"] += 1
            mut_dict["prev_nodes"].append(new_node)
            return swap_layer(individual, orig_pset, mut_dict)
        
        newind = gp.PrimitiveTree(individual[:slice_.start] + new_subtree + subtree + individual[slice_.stop:])
        individual.clear()
        individual.__init__(newind)
        # concat_healer(individual, pset, subtree) 
    print(individual)
    return individual,

def modify_layer(individual, orig_pset, mut_dict):
    """
    Shrinks the *individual* by choosing randomly a branch and
    replacing it with one of the branch's arguments (also randomly chosen).
    :param individual: The tree to be shrinked.
    :returns: A tuple of one tree.
    """
    iprims = []

    pset = cp.deepcopy(orig_pset)
    for i, node in enumerate(individual[1:], 1):
        if not isinstance(node, gp.Primitive):
            continue
        if issubclass(node.ret, nnm.LayerList) and issubclass(node.args[-1], nnm.LayerList) and i not in mut_dict["prev_nodes"]:
            iprims.append((i, node))
    
    new_individual = cp.deepcopy(individual)
    if len(iprims) != 0:
        index, prim = random.choice(iprims)
        arg_list = [i for i, type_ in enumerate(prim.args) if type_ == prim.ret]
        if len(arg_list)>0:
            arg_idx = random.choice(arg_list)
        else:
            return individual,
        rindex = index + 1
        for _ in range(arg_idx + 1):
            rslice = new_individual.searchSubtree(rindex)
            subtree = new_individual[rslice]
            rindex += len(subtree)

        slice_ = new_individual.searchSubtree(index)
        new_individual[slice_] = subtree

        try:
            temp_ind = gp.PrimitiveTree(new_individual[2:rslice.start-1] + subtree)
            func = gp.compile(temp_ind, pset)
            for dataset in mut_dict["datasetDict"]:
                func = func(nnm.InputLayer())
                layers, _ = nnm.compile_layerlist(func, [], mut_dict["data_dim"], mut_dict["datasetDict"][dataset]['type'], isNNLearner=mut_dict['isNNLearner'])
        except Exception as e:
            print(e)
            mut_dict["tries"] += 1
            mut_dict["prev_nodes"].append(index)
            return modify_layer(individual, pset, mut_dict)

    return individual, # added comma here. Don't know if it was omitted on purpose

def one_point_crossover_full_prim(ind1, ind2, orig_pset, mut_dict):

    """Randomly select crossover primitive in each individual and exchange each subtree with the point as root between each individual.
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossovers on single node tree
        return ind1, ind2

    pset_1 = cp.deepcopy(orig_pset)
    pset_2 = cp.deepcopy(orig_pset)
    orig_ind1 = cp.deepcopy(ind1)
    orig_ind2 = cp.deepcopy(ind2)

    if (mut_dict["prev_nodes"] == []):
        mut_dict["prev_nodes"] = [[],[]]

    # List all available primitive types in each individual
    types1 = defaultdict(list) 
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        types1[__type__] = xrange(1, len(ind1))
        types2[__type__] = xrange(1, len(ind2))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):

            if isinstance(node, gp.Primitive) and issubclass(node.ret, nnm.LayerList) and idx not in mut_dict["prev_nodes"][0]:
                types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            if isinstance(node, gp.Primitive) and issubclass(node.ret, nnm.LayerList) and idx not in mut_dict["prev_nodes"][1]:
                types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1) 
        slice2 = ind2.searchSubtree(index2)

        try:
            temp_ind_1 = gp.PrimitiveTree(ind1[:slice1.start] + ind2[slice2] + ind2[slice2.stop:])
            temp_ind_2 = gp.PrimitiveTree(ind2[:slice2.start] + ind1[slice1] + ind1[slice1.stop:])
            func_1 = gp.compile(temp_ind_1, pset_1)
            func_2 = gp.compile(temp_ind_2, pset_2)
            for dataset in mut_dict["datasetDict"]:
                func_1 = func_1(nnm.InputLayer())
                func_2 = func_2(nnm.InputLayer())
                layers, _ = nnm.compile_layerlist(func_1, [], mut_dict["data_dim"], mut_dict["datasetDict"][dataset]['type'], isNNLearner=mut_dict['isNNLearner'])
                layers, _ = nnm.compile_layerlist(func_2, [], mut_dict["data_dim"], mut_dict["datasetDict"][dataset]['type'], isNNLearner=mut_dict['isNNLearner'])
        except Exception as e:
            print(e)
            mut_dict["tries"] += 1
            mut_dict["prev_nodes"][0].append(index1)
            mut_dict["prev_nodes"][1].append(index2)
            return one_point_crossover_full_prim(orig_ind1, orig_ind2, orig_pset, mut_dict)

        new_ind1 = gp.PrimitiveTree(ind1[:slice1.start] + ind2[slice2] + ind2[slice2.stop:])
        new_ind2 = gp.PrimitiveTree(ind2[:slice2.start] + ind1[slice1] + ind1[slice1.stop:])

        print(new_ind1)
        print(new_ind2)
        return new_ind1, new_ind2
    return ind1, ind2


def one_point_crossover_middle_prim(ind1, ind2, orig_pset, mut_dict):
    """Randomly select crossover primitive in each individual and exchange each subtree with the point as root between each individual.
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossovers on single node tree
        return ind1, ind2
    
    pset_1 = cp.deepcopy(orig_pset)
    pset_2 = cp.deepcopy(orig_pset)
    orig_ind1 = cp.deepcopy(ind1)
    orig_ind2 = cp.deepcopy(ind2)

    if (mut_dict["prev_nodes"] == []):
        mut_dict["prev_nodes"] = [[],[]]

    # List all available primitive types in each individual
    types1 = defaultdict(list) 
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        types1[__type__] = xrange(1, len(ind1))
        types2[__type__] = xrange(1, len(ind2))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):

            if isinstance(node, gp.Primitive) and issubclass(node.ret, nnm.LayerList) and idx not in mut_dict["prev_nodes"][0]:
                types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            if isinstance(node, gp.Primitive) and issubclass(node.ret, nnm.LayerList) and idx not in mut_dict["prev_nodes"][1]:
                types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1) 
        slice2 = ind2.searchSubtree(index2)

        try:
            temp_ind_1 = gp.PrimitiveTree(ind1[:slice1.start] + ind2[slice2] + ind2[slice2.stop:])
            temp_ind_2 = gp.PrimitiveTree(ind2[:slice2.start] + ind1[slice1] + ind1[slice1.stop:])
            func_1 = gp.compile(temp_ind_1, pset_1)
            func_2 = gp.compile(temp_ind_2, pset_2)
            for dataset in mut_dict["datasetDict"]:
                func_1 = func_1(nnm.InputLayer())
                func_2 = func_2(nnm.InputLayer())
                layers, _ = nnm.compile_layerlist(func_1, [], mut_dict["data_dim"], mut_dict["datasetDict"][dataset]['type'], isNNLearner=mut_dict['isNNLearner'])
                layers, _ = nnm.compile_layerlist(func_2, [], mut_dict["data_dim"], mut_dict["datasetDict"][dataset]['type'], isNNLearner=mut_dict['isNNLearner'])
        except Exception as e:
            print(e)
            mut_dict["tries"] += 1
            mut_dict["prev_nodes"][0].append(index1)
            mut_dict["prev_nodes"][1].append(index2)
            return one_point_crossover_middle_prim(orig_ind1, orig_ind2, orig_pset, mut_dict)

        new_ind1 = gp.PrimitiveTree(ind1[:slice1.start] + ind2[slice2] + ind1[slice1.stop:])
        new_ind2 = gp.PrimitiveTree(ind2[:slice2.start] + ind1[slice1] + ind2[slice2.stop:])
        print(new_ind1)
        print(new_ind2)
        return new_ind1, new_ind2
    return ind1, ind2


#def two_point_crossover_prim(ind1, ind2):
#    """Randomly select crossover primitive in each individual and exchange each subtree with the point as root between each individual.
#    :param ind1: First tree participating in the crossover.
#    :param ind2: Second tree participating in the crossover.
#    :returns: A tuple of two trees.
#    """
#    if len(ind1) < 2 or len(ind2) < 2:
#        # No crossovers on single node tree
#        return ind1, ind2
#
#    # List all available primitive types in each individual
#    types1 = defaultdict(list) 
#    types2 = defaultdict(list)
#    if ind1.root.ret == __type__:
#        types1[__type__] = xrange(1, len(ind1))
#        types2[__type__] = xrange(1, len(ind2))
#        common_types = [__type__]
#    else:
#        for idx, node in enumerate(ind1[1:], 1):
#            if isinstance(node, gp.Primitive) and node.ret == GPFramework.neural_network_methods.LayerList:
#                types1[node.ret].append(idx)
#        for idx, node in enumerate(ind2[1:], 1):
#            if isinstance(node, gp.Primitive) and node.ret == GPFramework.neural_network_methods.LayerList:
#                types2[node.ret].append(idx)
#        common_types = set(types1.keys()).intersection(set(types2.keys()))
#
#    if len(common_types) > 0:
#        type_ = random.choice(list(common_types))
#
#        index1 = random.choice(types1[type_])
#        index2 = random.choice(types2[type_])
#
#        slice1 = ind1.searchSubtree(index1) 
#        slice2 = ind2.searchSubtree(index2)
#
#        new_ind1 = gp.PrimitiveTree(ind1[:slice1.start] + ind2[slice2] + ind2[slice2.stop:])
#        new_ind2 = gp.PrimitiveTree(ind2[:slice2.start] + ind1[slice1] + ind1[slice1.stop:])
#    
#    types3 = defaultdict(list) 
#    types4 = defaultdict(list)
#    if new_ind1.root.ret == __type__:
#        types1[__type__] = xrange(1, len(new_ind1))
#        types2[__type__] = xrange(1, len(new_ind2))
#        common_types = [__type__]
#    else:
#        for idx, node in enumerate(new_ind1[1:], 1):
#            if isinstance(node, gp.Primitive) and node.ret == GPFramework.neural_network_methods.LayerList:
#                types3[node.ret].append(idx)
#        for idx, node in enumerate(new_ind2[1:], 1):
#            if isinstance(node, gp.Primitive) and node.ret == GPFramework.neural_network_methods.LayerList:
#                types4[node.ret].append(idx)
#        common_types = set(types3.keys()).intersection(set(types4.keys()))
#
#    if len(common_types) > 0:
#        type_ = random.choice(list(common_types))
#
#        index3 = random.choice(types3[type_])
#        index4 = random.choice(types4[type_])
#
#        slice3 = new_ind1.searchSubtree(index3) 
#        slice4 = new_ind2.searchSubtree(index4)
#
#        new_new_ind1 = gp.PrimitiveTree(new_ind1[:slice3.start] + new_ind2[slice4] + new_ind2[slice4.stop:])
#        new_new_ind2 = gp.PrimitiveTree(new_ind2[:slice4.start] + new_ind1[slice3] + new_ind1[slice3.stop:])
#
#    return new_new_ind1, new_new_ind2

# def shuffle_layers(individual):
#     print(individual)

#     from GPFramework.neural_network_methods import LayerList
#     layer_bool = [node.ret == LayerList for node in individual]
#     layer_inds = np.where(layer_bool)[0].tolist()
#     new_layer_inds = layer_inds[:]
#     random.shuffle(new_layer_inds)
#     import copy as cp
#     new_individual = cp.deepcopy(individual)
    
#     if new_layer_inds != []:
#         tracker = 0
#         for index in range(len(layer_inds)):
#             arity1 = individual[layer_inds[index]].arity
#             orig_inds = []
#             for i in range(arity1):
#                 orig_inds.append(i+layer_inds[index]+arity1)

#             arity2 = individual[new_layer_inds[index]].arity
#             swap_inds=[]
#             for i in range(arity2):
#                 swap_inds.append(i+layer_inds[index]+arity2)

#             for i in range(len(swap_inds)):
#                 new_individual[orig_inds[i]] = individual[swap_inds[i]]
            
#     # if new_layer_inds != []:
#     #     tracker = 0
#     #     for index in range(len(layer_inds)):
#     #         #get indices of layer being replaced
#     #         slice1 = individual.searchSubtree(layer_inds[index])

#     #         subtree = individual[slice1]
            
#     #         height1 = len(subtree)
#     #         orig_inds = []
#     #         for i in range(height1):
#     #             orig_inds.append(i+layer_inds[index]+tracker)

#     #         #get indices of layer that is replacing
#     #         slice2 = individual.searchSubtree(new_layer_inds[index])
#     #         subtree = individual[slice2]
#     #         height2 = len(subtree)

#     #         swap_inds = []
#     #         for i in range(height2):
#     #             swap_inds.append(i+new_layer_inds[index])

#     #         import pdb; pdb.set_trace()
#     #         #swap layers
#     #         for i in range(len(swap_inds)):
#     #             new_individual[orig_inds[i]] = individual[swap_inds[i]]

#     #         tracker += len(new_inds)-len(orig_inds)
    

#     return new_individual
