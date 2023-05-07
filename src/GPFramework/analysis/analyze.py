"""
Methods to analyze out files from EMADE

Meant to make debugging, analysis, and visualizations of EMADE results easier

TODO:
  - Add code to emade algo --> dump to EmadeOutput object via DB @ end of run & write to json ??? Ask Dr. Zutty or do nothing
"""

import json
import numpy as np
import os
import xml.etree.ElementTree as ET
import gzip
from GPFramework.tree_vis import get_img, write_png, process_string
from sqlalchemy import func
from deap.tools._hypervolume import hv
import random






# Functions to init an EmadeOutput object:

def read_out(out_file, xml_file, db=False, emade_path=""):
  return EmadeOutput(xml_path=xml_file, out_path=out_file, emade_path=emade_path, db=db)

def read_db(xml_file, emade_path=""):
  return EmadeOutput(xml_path=xml_file, db=True, emade_path=emade_path)

def read_json(filename, emade_path="", db=False):
  return EmadeOutput(out_json=filename, emade_path=emade_path, db=db)



  

class EmadeOutput():
  def __init__(self, out_json:str="", xml_path:str="", out_path:str="", emade_path="", db=False):
    """ TODO: update this description

    out_json: path to json file created by EmadeOutput.save()

    xml_path: Path to XML input file. 
      - If out_path == out_json == "", run info will be retrived from database
    
    out_path: Path to .out file of EMADE master algorithm
      - If supplied, run results will be retrieved from this file (unless json is provided)
      - Run parameters (e.g., objectives, DB info, datasets) will be read from XML file and stored
      - Pros:
        - Can be used for older runs whose DB info wasn't saved
        - Doesn't have to manually calculate AUCs
      - Cons:
        - .out file usually doesn't have access to error strings
        - .out file doesn't have individuals from an incomplete generation (these still don't show up on the pareto front when loading from DB)

    emade_path: Path to your emade directory
      - if not supplied, tries to find emade in xml_path's parent directory 
    
    db: True if loading from out/json but you want to connect to db anyway


    Attributes after init:
      - gens: (sorted) list of generations (ints)
      - gens_complete: might be 1 lower than max(self.gens) if we have individuals that were evaluated in an incomplete generation
      - pareto: Population of Individuals on final pareto front
      - pareto_history: dict in format {generation: Population} 
      - auc
      - auc_history
      - inds: Population of all evaluated individuals
      - objectives: tuple of objectives (strings)
      - obj_ranges: tuple of (lower, upper) tuples for each objective
    """

    self.xml_path = xml_path

    ### Checking for valid inputs: ###
    if not (out_json or xml_path and db or xml_path and out_path):
      raise ValueError("Cannot load EMADE data with given arguments. EMADE data can be loaded from an EmadeOutput json, a master.out file + input.xml file, or an input.xml file + database connection")
    if db and not xml_path:
      raise ValueError("Must supply an XML file to connect to the database")

    ### try to find path to emade directory ###
    self.emade_path=None
    if emade_path: # checking if provided
      self.emade_path=emade_path
    elif self.xml_path: # if not, checks xml_path grandparent. If "src" is in directory, naively assumes we've found emade
      emade_path = os.path.dirname(os.path.dirname(self.xml_path))
      if "src" in os.listdir(emade_path):
        self.emade_path=emade_path
    if self.emade_path:
      print("set current working directory to {}".format(self.emade_path))
      os.chdir(self.emade_path)
    else:
      print("Warning: emade directory path couldn't be found manually. Processes requiring datasets (e.g., standalone tree evaluator) might not work properly")
      print("If this isn't intentional, consider supplying an argument for `emade_path` on init")

    ### Determines method to load EMADE data ###    
    if out_json:
      self.load(out_json)
      if self.xml_path and db:
        self.db_init()
    elif not out_path:
      # init from DB
      self.parse_xml(db)
      self.process_db()
    else:
      # init from master algorithm's out file
      self.parse_xml(db)
      self.process_data(read_out(out_path, self.objectives))

  def parse_xml(self,db):
    tree = ET.parse(self.xml_path)

    objectives = [objective.findtext('name') for objective in tree.iter('objective')]
    datasets = [dataset.findtext("name") for dataset in tree.getroot().iter('dataset')]
    self.objectives = tuple([dataset+" "+obj for dataset in datasets for obj in objectives]) # Puts objectives in format emade uses. e.g., `FullDataSet False Positives`
    self.obj_ranges = tuple([(float(obj.findtext("lower")), float(obj.findtext("upper"))) for obj in tree.iter("objective")])

    try:
      assert db==True
      self.db_init()
    except Exception as e:
      self.db=None
      self.pset=None
      print("Did not connect to database. Things like tree evaluation/error strings won't work")
      if db:
        print("\nDatabase error: {}".format(e))

  def db_init(self):
    from GPFramework.general_methods import load_environment
    self.db, self.pset= load_environment(self.xml_path) # for standalone tree evaluator mostly
    self.sql=self.db.sessions[os.getpid()]

  def process_db(self):
    # main function to process data from DB
    self.gens = list(range(self.sql.query(func.max(self.db.Individual.evaluation_gen)).all()[0][0]+1))
    self.pareto_history = {}
    self.auc_history = {}
    self.inds = Population()

    # build population of individuals
    for ind in self.sql.query(self.db.Individual).filter_by(evaluation_status="EVALUATED").all(): 
      self.inds.add(Individual(self.ind_from_hash(ind.hash)))

    # build pareto front, pareto history, auc, and auc_history here
    if len(self.inds.all) > 0:
      for gen in self.gens:
        # add to pareto_history and auc_history
        pareto_inds = [Individual(self.ind_from_hash(ind[0])) for ind in self.sql.query(self.db.ParetoFront.hash).filter_by(generation=gen).all()]
        if gen==max(self.gens) and len(pareto_inds)==0: # check if the last generation is incomplete (so pareto table isn't updated)
          self.pareto=self.pareto_history[gen-1]
          self.auc=self.auc_history[gen-1]
          self.gens_complete = gen-1
        else:
          self.pareto_history[gen]=Population(pareto_inds, pareto=True)
          auc=self.get_auc(Population(pareto_inds))
          self.auc_history[gen]=auc
          if gen == max(self.gens):
            # create self.auc and self.pareto
            self.pareto=Population(pareto_inds, pareto=True)
            self.auc=auc
            self.gens_complete = gen
    else:
      self.pareto = self.auc = None
      self.gens_complete = 0

  def ind_from_hash(self,_hash):
#     takes hash, queries individual, & objectives and returns dictionary used by Individual.__init__
    ind = self.sql.query(self.db.Individual).filter_by(hash=_hash).first()
    cast_inf = lambda x: x if x != None else np.inf # inf fitnesses get written as NULL/None to DB
    fitness = [cast_inf(getattr(ind, obj)) for obj in self.objectives]
    result = {}
    result["individual"]=ind.tree
    result["error"]=ind.error_string
    result["gen"]=ind.evaluation_gen
    result["fitness"]=fitness
    return result
  
  def nn_ind_from_hash(self,_hash):
    inds = self.sql.query(self.db.NNStatistics).filter_by(hash=_hash)
    inds = sorted(inds, key=lambda ind: ind.age)
    results = []
    for ind in inds:
      result  = {}

      result["age"] = int(ind.age)
      result["current_tree"] = ind.current_tree
      result["error_string"] = ind.error_string

      results.append(result)
    return results

  def view_nn_statistics(self, _hash, filepath="./nn_statistics.csv"):
    inds = self.nn_ind_from_hash(_hash=_hash)
    import csv
    with open(filepath, 'w') as out:
        writer = csv.DictWriter(out, inds[0].keys() if inds else [])
        writer.writeheader()
        writer.writerows(inds)
  
  def process_data(self, data):
    # main function to process data from .out file
    self.gens=sorted(data.keys())
    self.gens_complete = max(self.gens)
    self.pareto_history={}
    self.auc_history={}
    self.inds = Population()

    for gen in self.gens:
      if gen==max(self.gens):
        self.pareto = Population([Individual(ind) for ind in data[gen]["pareto_individuals"]], pareto=True)
        self.auc = data[gen]["hypervolume"]
      self.pareto_history[gen] =  Population([Individual(ind) for ind in data[gen]["pareto_individuals"]], pareto=True)
      self.auc_history[gen] = data[gen]["hypervolume"]

      for ind in data[gen]["individuals"]:
        self.inds.add(Individual(ind))


  #################################
  ### Useful methods start here ###
  #################################
  def standalone(self, ind):
    from GPFramework.standalone_tree_evaluator import eval_string
    if isinstance(ind, Individual):
      return eval_string(ind.tree, self.db, self.pset)
    elif isinstance(ind, str):
      return eval_string(ind, self.db, self.pset)
    else:
      raise ValueError("Please pass an `Individual` object or a tree's string representation")

  def get_auc(self,pop):
    # a more convenient way to call get_auc for an EmadeOutput object
    return get_auc(pop, self.obj_ranges)

  def save(self,filename, _zip=True):
    # saves attributes to name.json
    output = dict()
    output["individuals"]=[{"individual":ind.tree, "gen":ind.gen, "fitness":ind.fitness,"root":ind.root, "error":ind.error} for ind in self.inds.all]
    output["xml_path"]=self.xml_path
    output["max_gen"]=max(self.gens)
    output["gens_complete"]=self.gens_complete
    output["auc_history"]=self.auc_history
    output["pareto_history"]={gen:[{"individual":ind.tree, "gen":ind.gen, "fitness":ind.fitness,"root":ind.root, "error":ind.error} for ind in self.pareto_history[gen]] for gen in range(self.gens_complete+1)}
    output["objectives"]=self.objectives
    output["objective_ranges"]=self.obj_ranges

    json_object = json.dumps(output, indent = 4)
    
    filename = filename.split('.')[0]
    if _zip:
      filename += ".json.gz"
      with gzip.open(filename, "w") as outfile: 
        outfile.write(json_object.encode('utf-8'))
    else:
      filename += ".json"
      with open(filename, "w") as outfile:
        outfile.write(json_object)
    print("EmadeOutput object saved to {}".format(filename))

  def load(self, filename):
    if ".json" not in filename:
      raise ValueError("{} is not a valid (potentially zipped) json file}".format(filename))
    
    if ".gz" in filename:
      d = load_data(filename, _zip=True)
    else:
      d = load_data(filename)
    self.xml_path = d["xml_path"]
    self.gens_complete = d["gens_complete"]
    self.gens = list(range(d["max_gen"]+1))
    self.objectives = tuple(d["objectives"])
    self.obj_ranges = tuple(tuple(pair) for pair in d["objective_ranges"])
    self.inds = Population([Individual(ind) for ind in d["individuals"]])
    self.auc_history = {int(gen):auc for gen, auc in d["auc_history"].items()}
    self.auc = self.auc_history[self.gens_complete]
    self.pareto_history = {gen: Population([Individual(ind) for ind in d["pareto_history"][str(gen)]], pareto=True) for gen in range(self.gens_complete+1)}
    self.pareto = self.pareto_history[self.gens_complete]



class Population(): # filter() --> filter_prim & add other filter methods?
  def __init__(self, pop=None, pareto=False):
    if pop==None:
      self.all = []
    else:
      self.all=pop
    self.pareto=pareto

  def __getitem__(self,index):
    return self.all[index]

  def __eq__(self, o: object) -> bool:
      assert isinstance(o,Population)
      return self.all == o.all and self.pareto == o.pareto

  def add(self,item):
    self.all.append(item)

  def valid(self):
    return Population([ind for ind in self.all if np.inf not in ind.fitness], pareto=self.pareto)

  def invalid(self):
    return Population([ind for ind in self.all if np.inf in ind.fitness])

  def filter(self, prim, root=False):
    '''
    returns individuals containing given primitive
    if root==True, only returns individuals with primitive at root

    args:
      - prim (string): primitive to search for
    '''
    if root:
      result= [ind for ind in self.all if ind.root==prim.lower()]
    else:
      result= [ind for ind in self.all if prim.lower() in ind.primitives()]
    return Population(result)

  def sort(self, axis=0, reverse=False):
    return Population(sorted(self.all, key=lambda x:x.fitness[axis], reverse=reverse), pareto=self.pareto)

  def random(self):
    return self.all[random.randint(0, len(self.all)-1)]

  def __repr__(self):
    return str(self.all)





class Individual():
  def __init__(self, ind): 
    self.tree = ind["individual"]
    self.gen = ind["gen"]
    self.fitness = ind["fitness"]
    if "root" in ind.keys():
      self.root=ind["root"]
    else:
      self.root = self.tree[0:self.tree.find('(')].lower()
    self.error=ind["error"] # storing error strings might add a lot of bloat

  def __eq__(self, o: object) -> bool:
      assert isinstance(o,Individual)
      return self.tree == o.tree and self.gen==o.gen and self.fitness==o.fitness

  def show(self, prim_col="black", arg_col="black", prim_shape="rect", arg_shape="oval"):
    # outputs image of tree when in a notebook
    try:
      get_img(self.tree, prim_col, arg_col, prim_shape, arg_shape)
    except Exception as e:
      print("Something went wrong. Please make sure you have GraphViz installed and added to your environment's path")
      print("\nError: {}".format(e))
  
  def png(self, filename, prim_col="black", arg_col="black", prim_shape="rect", arg_shape="oval"):
    # writes image of tree to filename.png
    try:
      write_png(filename, self.tree, prim_col, arg_col, prim_shape, arg_shape)
    except Exception as e:
      print("Something went wrong. Please make sure you have GraphViz installed and added to your environment's path")
      print("\nError: {}".format(e))

  def primitives(self):
    # returns list of primitives in individual (not unique)
    nested_prims = process_string(self.tree.replace(' ',''))
    def find_prim(_list,result):
      for i in range(len(_list)):
        if len(_list)>i+1 and type(_list[i+1])==list:
          result.append(_list[i].lower())
          find_prim(_list[i+1], result)
      return result
    return find_prim(nested_prims, [])

  def __repr__(self):
    return self.tree


def read_out(filename, objectives, write=False):
  # I have no idea if this works with >2 objectives
  # writes to a .json if write=True
  # returns dictionary regardless

  try:
      inputfile = open(filename)
  except:
      print(filename)
      raise ValueError("There was a problem loading out file {}".format(filename))
  output = {}
  generation = 0
  individuals = False
  pareto = False
  individuals = []
  pareto_individuals = []
  hypervolume = 0

  for line in inputfile:
      if line == 'Pareto Front Updated\n':
          pareto = True
          pareto_individuals = []
      if line == 'Updated Elite Pool\n':
          pareto = False
          output[generation] = {"individuals":individuals, 
                                  "hypervolume": hypervolume, 
                                  "pareto_individuals":pareto_individuals}
          generation = generation + 1
      if pareto:
          if line[:17] == 'Pareto Individual':
              string = line[line.index('is') + 3:len(line)-1]
              ind = {"individual": string[:string.index(')(')+1]}
              ind["gen"]=generation
              ind["error"]=None # can't get errors from master.out file
              fitnesses = [float(fitness) for fitness in string.split(')(')[1].split(')')[0].split(', ')]
              ind["fitness"]=fitnesses
              pareto_individuals.append(ind)
          if line.startswith("Hypervolume"):
              hypervolume = float(line[14:])

      if line == 'Updating population\n':
          individual = True
          individuals = []
      if line == 'Gene pool assembled':
          individual = False
      if line[:8] == 'Received'  and individual:
          ind = {"individual": line[10:len(line)-1]}
          ind["gen"]=generation
          ind["error"]=None # can't get individual error strings from master.out file
          individuals.append(ind)
      if line.startswith('\tWith Fitnesses:'):
          fitnesses = [float(fitness) for fitness in line.split('(')[1].split(')')[0].split(',')]
          individuals[len(individuals) - 1]["fitness"] = fitnesses
  json_object = json.dumps(output, indent = 4) 

  if write:
    with open(filename[:-4]+".json", "w") as outfile: 
        outfile.write(json_object) 
  inputfile.close()
  return output

def load_data(json_filename, _zip=False):
  # reads a .json file and returns the dictionary
  if _zip:
    with gzip.open(json_filename, 'r') as fin:
      data = json.loads(fin.read().decode('utf-8'))
  else:
    file = open(json_filename,"r")
    data = json.load(file)
    file.close()
  return data

def get_pareto(pop):
  """
  Returns indices and fitnesses of pareto optimal individuals in pop (assumes minimization)
  Inputs:
    - pop: Population object
  Outputs: tuple with (Population object, numpy array):
    - Subset of population containing only pareto optimal individuals
    - fitnesses of said individuals
  """
  fitnesses = np.array([ind.fitness for ind in pop.all])

  pareto = np.arange(fitnesses.shape[0]) # eventually becomes indices of pareto fitnesses
  next_index = 0
  while next_index < len(fitnesses):
    pareto_masks = np.any(fitnesses < fitnesses[next_index], axis=1) # boolean array, false for dominated fitnesses
    pareto_masks[next_index]=True 
    pareto = pareto[pareto_masks] # remove dominated indices
    fitnesses = fitnesses[pareto_masks] # remove dominated fitnesses
    next_index = np.sum(pareto_masks[:next_index])+1 # recalculate index of next fitness to check
  pop = Population([pop.all[i] for i in pareto], pareto=True)
  return pop, fitnesses

def get_auc(pop, obj_ranges):
    """
    Calculates auc of a pareto front. 
    Assumes the following: 
      - minimization objectives
      - all fitnesses are within specified ranges

    Inputs:
      - pop: Population object
      - obj_ranges: nested sequence of ranges for objectives
    Returns:
      - auc (float)
    """
    if not pop.pareto:
      front = get_pareto(pop)[1]
    objLimits = [r[1] for r in obj_ranges]
    objGoals = [r[0] for r in obj_ranges]
    hv_input = 1
    for obj in range(len(objGoals)):
        hv_input *= (objLimits[obj] - objGoals[obj])
    if len(front)==0:
      return hv_input
    return hv_input - hv.hypervolume(np.array(front), tuple(objLimits))


# seeded = [
#           "NNLearner(ARG0, OutputLayer(DenseLayer(10, defaultActivation, 10, LSTMLayer(16, defaultActivation, 0, trueBool, trueBool, EmbeddingLayer(100, ARG0, randomUniformWeights, InputLayer())))), 100, AdamOptimizer)",
#           "NNLearner(ARG0, OutputLayer(DenseLayer(10, defaultActivation, 10, LSTMLayer(16, defaultActivation, 0, trueBool, trueBool, EmbeddingLayer(99, ARG0, randomUniformWeights, InputLayer())))), 90, AdamOptimizer)",
#           "NNLearner(ARG0, OutputLayer(DenseLayer(10, defaultActivation, 10, LSTMLayer(16, defaultActivation, 0, trueBool, trueBool, EmbeddingLayer(98, ARG0, randomUniformWeights, InputLayer())))), 98, AdamOptimizer)",
#           "NNLearner(ARG0, OutputLayer(DenseLayer(10, defaultActivation, 10, LSTMLayer(16, defaultActivation, 0, trueBool, trueBool, EmbeddingLayer(97, ARG0, randomUniformWeights, InputLayer())))), 99, AdamOptimizer)",
#           "NNLearner(ARG0, OutputLayer(DenseLayer(10, defaultActivation, 10, LSTMLayer(16, defaultActivation, 0, trueBool, trueBool, EmbeddingLayer(96, ARG0, randomUniformWeights, InputLayer())))), 94, AdamOptimizer)",
#           "NNLearner(ARG0, OutputLayer(DenseLayer(10, defaultActivation, 10, LSTMLayer(16, defaultActivation, 0, trueBool, trueBool, EmbeddingLayer(94, ARG0, randomUniformWeights, InputLayer())))), 93, AdamOptimizer)",
#           "NNLearner(ARG0, OutputLayer(DenseLayer(10, defaultActivation, 10, LSTMLayer(16, defaultActivation, 0, trueBool, trueBool, EmbeddingLayer(92, ARG0, randomUniformWeights, InputLayer())))), 92, AdamOptimizer)",
#           "NNLearner(ARG0, OutputLayer(DenseLayer(10, defaultActivation, 10, LSTMLayer(16, defaultActivation, 0, trueBool, trueBool, EmbeddingLayer(93, ARG0, randomUniformWeights, InputLayer())))), 91, AdamOptimizer)",
#           "NNLearner(ARG0, OutputLayer(DenseLayer(10, defaultActivation, 10, LSTMLayer(16, defaultActivation, 0, trueBool, trueBool, EmbeddingLayer(90, ARG0, randomUniformWeights, InputLayer())))), 89, AdamOptimizer)"
# ]