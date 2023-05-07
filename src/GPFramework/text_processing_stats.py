import numpy as np
import copy as cp
import re
import os
import scipy
import matplotlib.pyplot as plt
import csv



def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def get_objectives(l):
    return np.array([np.array([t.obj[0], t.obj[1]]) for t in l])


def pareto_filter(l):
    obj = get_objectives(l)
    print(f"obj shape: {obj.shape}")
    bitmask = is_pareto_efficient_simple(obj)
    return (np.array(l)[bitmask == 1]).tolist()

TYPE = "nlp_app"


class ParetoFront:
    """
    Class to hold info for a pareto front in a single generation.
    """
    def __init__(self, generation, individuals):

        self.gen = int(generation) #int
        self.inds = set(individuals) #list[Individual]
        self.list_inds = individuals
        self.auc = 0 #float

        if TYPE != "nlp_app":
            self.inds.add(Individual(0, "hash1", "auc_bound_1", [0, 1]))
            self.inds.add(Individual(0, "hash2", "auc_bound_2", [1, 0]))
        else:
            self.inds.add(Individual(0, "hash1", "auc_bound_1", [0, 8000]))
            self.inds.add(Individual(0, "hash2", "auc_bound_2", [8000, 0]))

    def compute_auc(self):
        self.auc = 0
        temp = cp.deepcopy(self.inds)
        self.inds = list(self.inds)
        self.inds.sort()

        if type(self.inds[0].obj[0]) == str:
            fitness_1 = [float(ind.obj[0][:-1]) for i, ind in enumerate(self.inds)]
            fitness_2 = [float(ind.obj[1][:-1]) for i, ind in enumerate(self.inds)]
        else:
            fitness_1 = [ind.obj[0] for ind in self.inds]
            fitness_2 = [ind.obj[1] for ind in self.inds]
        f1 = np.array(fitness_1)
        f2 = np.array(fitness_2)
        #print(f"f1: {f1}")
        #print(f"f2: {f2}")
        self.auc = np.sum(np.abs(np.diff(f1))*f2[:-1])
        if self.auc == 0:

            # print("lonely pareto ind detected.")
            # print(f"BEFORE: {(f1, f2)}")
            f1 = np.unique(f1)
            f2 = np.unique(f2)
            # print(f"AFTER: {(f1, f2)}")
            print()

        # self.inds = temp
        # if len(f1) == 1:
        #     self.inds.add(Individual(self.gen, "hash1", "auc_bound_1", [0, 1]))
        #     self.inds.add(Individual(self.gen, "hash2", "auc_bound_2", [1, 0]))
        #     print("lonely pareto --> RECOMPUTE")
        #     self.compute_auc()
        #     print(f"recompute done. {self.auc} > simple product {f1[0] * f2[0]}")
    def __lt__(self, other):
        return self.gen < other.gen
    def __str__(self):
        s = f"Gen: {self.gen}\n AUC: {self.auc}\n"
        for i in self.inds:
            s += str(i)
        return s
        


class Individual:
    """
    Class to hold info on an individual
    """
    def __init__(self, gen, hash_id, tree, objectives):
        #TODO add members if necessary

        self.gen = int(gen)
        self.hash = hash_id
        self.tree = tree
        self.obj = objectives #n-dim list of objectives. n >= 2. n = # objectives
    def __lt__(self, other):
        return self.obj[0] < other.obj[0]
    def __hash__(self):
        h = hash(self.tree)
        # print(f"current hash: {self.hash}")
        # print(f"hash tree: {h}")
        return h
    def __eq__(self, other):
        return self.hash == other.hash
    def __str__(self):
        return f"TREE: {self.tree}\nHash: {self.hash}\nGen: {self.gen}\nObjectives: {self.obj}\n"

def paretos(iterable, debug=False):

    """
    ITEM #1: Load/organize in individual information from SQL schema. (current option: convert to csv file, load csv file)

    Args:
        csv_path: path to csv file containing information from a single EMADE run.
        assume each line is a pareto individual with hash_id, tree, objectives, and generation number.

    Returns:
        pareto_fronts: 1D numpy array of ParetoFront objects
        where pareto_fronts[i] = pareto front for generation i
    """
    paretos = [] #list of ParetoFront objects
    d = {} #d: gen number -> list of individuals

    unique = set()
    for index in range(len(iterable)):
        # if index == 0: continue
        gen = int(iterable["evaluation_gen"][index]) #get gen number.
        hash_id = iterable["hash"][index]
        tree = iterable["tree"][index]
        obj1 = iterable["FullDataSet False Positives"][index]
        obj2 = iterable["FullDataSet False Negatives"][index]
        obj = [ int(obj1), int(obj2) ]
        ind = Individual(gen, hash_id, tree, obj)

        unique.add(ind)

    #for some reasone pareto_front_table is cumulative
    # so we'll keep a cumulative pareto fronts and filter for dominance.
    #
    #d[i] = list of inds with (gen <= i)
    unique_inds = sorted(list(unique), key=lambda x: x.gen)
    for i in unique_inds:
        print("+========+")
        print(i)
        print("+========+")
    gens = [i.gen for i in unique_inds]
    for gi, g in enumerate(gens):
        d[g] = pareto_filter(unique_inds[:gi+1])

    for gen, inds in d.items():
        paretos.append(ParetoFront(gen, inds))
        paretos[-1].compute_auc()
        print(f"added ParetoFront with auc: {paretos[-1].auc}")
    pareto_fronts = np.array(paretos)
    return pareto_fronts

"""
    paretos = [] #list of ParetoFront objects
    d = {} #d: gen number -> list of individuals


    #print(iterable["evaluation_gen"])
    for index in range(len(iterable)):
        if index == 0: continue
        gen = int(iterable["evaluation_gen"][index]) #get gen number.
        hash_id = iterable["hash"][index]
        tree = "tree"
        obj1 = iterable["FullDataSet False Positives"][index]
        obj2 = iterable["FullDataSet False Negatives"][index]
        #print(f"gen {gen}\n tree{tree}\n obj{obj}")
        if gen not in d.keys():
            d[gen] = []
        d[gen].append(Individual(gen, hash_id, tree, (obj1, obj2)))
                
    for gen, inds in d.items():
        paretos.append(ParetoFront(gen, inds))
        paretos[-1].compute_auc()

        if debug:print(f"added ParetoFront with auc: {paretos[-1].auc}")
    if debug: print(f"all paretos: {paretos}")
    pareto_fronts = np.array(paretos)
    return pareto_fronts
    """


# def load_pareto_fronts(csv_paths):
#     """
#     ITEM #1: Load/organize in individual information from SQL schema. (current option: convert to csv file, load csv file)

#     Args:
#         csv_path: path to csv file containing information from a single EMADE run.
#         assume each line is a pareto individual with hash_id, tree, objectives, and generation number.

#     Returns:
#         pareto_fronts: 1D numpy array of ParetoFront objects
#         where pareto_fronts[i] = pareto front for generation i
#     """
#     paretos = [] #list of ParetoFront objects
#     #TODO: put data into ParetoFront objects
#     for csv_path in csv_paths:
#         d = {} #d: gen number -> list of individuals
#         with open(csv_path, newline='') as csvfile:
#             spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#             gen = -1
#             for index, row in enumerate(spamreader):
#                 if index == 0: continue
#                 #print(row)
#                 gen = int(row[5][:-1]) #get gen number.
#                 hash_id = row[0] 
#                 tree = "".join(row[8:-6])
#                 obj = row[-4:-2] 
#                 #print(f"gen{gen}\n tree{tree}\n obj{obj}")
#                 if gen not in d.keys():
#                     d[gen] = []
#                 d[gen].append(Individual(gen, hash_id, tree, obj))
#                 #print(f"{gen}")
#     for gen, inds in d.items():
#         paretos.append(ParetoFront(gen, inds))
#         paretos[-1].compute_auc()
#         print(f"added ParetoFront with auc: {paretos[-1].auc}")
#     pareto_fronts = np.array(paretos)
#     return pareto_fronts


def compute_mean_variance(aucs):
    """
    ITEM #2: Compute Mean and Medians across generations.

    Args:
        aucs: 2D numpy array of AUC values
        where:
        N = number of emade runs 
        M = number of emade generations
        0 <= i <= N - 1
        0 <= j <= M - 1
        Arr[i] = list of AUCs for M generations for the ith EMADE run.
        Arr[i][j] = AUC for the jth generation of the ith EMADE run.

    Returns:
        2 1D numpy arrays. 1 for mean, 1 for variance.
        mean: s.t. mean[i] = mean of aucs of pareto front for generation i
        variance: s.t. variance[i] = variance of aucs of pareto front for generation i
    """
    #TODO: fill in
    mean_x1 = np.mean(aucs, axis=0)
    var_x1 = np.var(aucs, axis=0)
    return mean_x1, var_x1
    

def plot_boxplot(arr, show_mean=False, title=""):
    """
    ITEM #3: Plot Mean on top of Boxplot. (Bek)

    Args:
        arr: Given 2D numpy array of some values(likely be aucs, but we could use for other metrics).
            arr[i] = 
        show_mean: if True, then we show mean overlayed on boxplot.

    Returns:
        nothing, but shows matplotlib plot
    """
    shape = list(arr.shape)
    shape.reverse()
    arr.reshape(tuple(shape))

    print(shape)

    plt.xlabel(f'Generations {len(arr[0])}')
    plt.ylabel('Average AUC')

    if show_mean:  

        plt.suptitle(title+ ' Mean and Medians across generations', fontsize=12, fontweight="bold")

        plt.boxplot(arr, showmeans=True)
        mean_arr = np.insert(np.mean(arr, axis=0), 0, 0)
        print(f"mean arr shape: {mean_arr.shape}")
        plt.plot(mean_arr)


        start, end = plt.axes.Axes.get_xlim()
        plt.xaxis.set_ticks(np.arange(start, end, stepsize))

        plt.show()
    else:
        plt.suptitle(f'AVG PF AUC of all {title} runs across generations', fontsize=12, fontweight="bold")
        plt.boxplot(arr, showmeans=True)

        stepsize = 10
        start, end = plt.xlim()
        print(f"start: {start}")
        print(f"end: {end}")
        tickmarks = np.arange(start, end, stepsize, dtype='int')
        plt.xticks(tickmarks, tickmarks)
        # plt.xaxis.set_ticks()

        plt.show()

from scipy import stats

def compute_pvalues(mean1, var1, n1, mean2, var2, n2):
    """
    ITEM #4: Compute p-values from Hypothesis Test. Do Welch-T test.

    Args:
        meanWithPrimitive : 1d numpy array
        varianceWithPrimitive : 1d numpy array
        meanWithoutPrimitive : 1d numpy array
        varianceWithoutPrimitive : 1d numpy array

    Returns:
        numpy array of pValues across generations.
    """
    pValues = []
    for i in range(len(mean1)):

        pValues += [list(stats.ttest_ind_from_stats(mean1=mean1[i], std1=np.sqrt(var1[i]), nobs1=n1,

                                                     mean2=mean2[i], std2=np.sqrt(var2[i]), nobs2=n2,
                                                     equal_var=False))[1]]
    #TODO: fill in
    return np.array(pValues)



def plot_pvalues(pValues, title, critical_value=0.05):


    """
    ITEM #5: plot pvalues vs generations and also show critical value

    Args:
        pValues: Given 1D numpy array

    Returns:
        nothing, but shows matplotlib plot
    """
    plt.axhline(y=critical_value, xmin=0.0, color='r')

    plt.suptitle(title, fontsize=12, fontweight="bold")

    plt.plot(pValues)
    plt.xlabel('Generations')
    plt.ylabel('P-Values')
    plt.show()

import pandas as pd
def join(start, files):
    a, b = files
    a = start + a
    b = start + b
    print(a)
    print(b)
    DF1 = pd.read_csv(a)
    DF2 = pd.read_csv(b)
    DF1 = DF1[['hash','evaluation_gen', 'tree', "FullDataSet False Positives", "FullDataSet False Negatives"]]
    DF2 = DF2[['hash']]
    output = pd.merge(DF1, DF2, how='inner', left_on='hash', right_on='hash')
    return output

def to_numpy(l):
    for index, elem in enumerate(l):
        l[index] = np.array(elem)
    return np.array(l)


"""
Data as of April 17, 2020

BASELINEs
movie_reviews_nonnull_baseline_nlp-app_3.csv
movie_reviews_paretofront_baseline_nlp-app_3.csv

PRIMITIVEs       
movie_reviews_nlp-app_1.csv                      
movie_reviews_nlp-app_paretofront_1.csv 

movie_reviews_all_non-null_nlp-app_3.csv    
movie_reviews_paretoindividuals_nlp-app_3.csv

movie_reviews_all_non-null_nlp_app_4.csv
movie_reviews_nlp-app_paretofront_4.csv

April 11th/movie_reviews_all_non-null_nlp_app_5.csv
April 11th/movie_reviews_nlp-app_paretofront_5.csv
"""

def nlp_app_test():
    start = "VIP_AAD_NeuralNet/"
    baseline_f = [
            ("movie_reviews_nonnull_baseline_nlp-app_3.csv", "movie_reviews_paretofront_baseline_nlp-app_3.csv"),
            ("April 17th/movie_reviews_all_non-null_baseline_6.csv", "April 17th/movie_reviews_all_pareto_baseline_6.csv"),
            ("April 17th/movie_reviews_all_non-null_baseline_5.csv", "April 17th/movie_reviews_all_pareto_baseline_5.csv")
            ]

    primitive_f = [
            ("movie_reviews_nlp-app_1.csv", "movie_reviews_nlp-app_paretofront_1.csv"),
            ("movie_reviews_all_non-null_nlp-app_3.csv", "movie_reviews_paretoindividuals_nlp-app_3.csv"),
            ("movie_reviews_all_non-null_nlp_app_4.csv", "movie_reviews_nlp-app_paretofront_4.csv"),
            ("April 11th/movie_reviews_all_non-null_nlp_app_5.csv", "April 11th/movie_reviews_nlp-app_paretofront_5.csv"),
            ("April 17th/movie_reviews_all_non-null_nlp_app_6.csv",  "April 17th/movie_reviews_all_pareto_nlp_app_6.csv")
            ]
    
    def get_samples(filenames, debug=False):
        """gets a list of samples of Emade runs.
        Args:
        filenames: an iterable
        each element is a tuple of csv paths. (individuals, paretofront)

        Returns
        samples: 2D np.ndarray of ParetoFront objects.
        axis=0 is the sample number 
        axis=1 is the generation number
        """
        samples = []
        gen_cut = np.inf
        max_gen = -1
        for f in filenames:
            if type(f) == tuple: #requires joining of 2 files.
                p = paretos(join(start, f), debug=debug) #list of pareto objects
            elif type(f) == str: #csv already joined.
                p = load_pareto_fronts([start+f])
            gen_cut = min(gen_cut, len(p))
            max_gen = max(max_gen, len(p))
            samples.append(np.array([pp.auc for pp in p]))
            ## did I ever call compute_auc???
        if debug:
            print(f"gen_cut: {gen_cut}")
            print(f"max_gen: {max_gen}")
            print(f"samples: {samples}")
        # print(f"uncut samples: {samples}")
        # for i, v in enumerate(samples):
        #     #samples[i] = v[:gen_cut]
        #     #instead of min_cut, fill up to max
        #     d = max_gen - len(v)
        #     if d != 0:
        #         samples[i] = np.concatenate((v, np.zeros((d))))
        #         print(f"extended samples: {len(samples[i])}")

        # print(f"CUT samples: {samples}")
        return np.array(samples), max_gen

    def extend_mean(samples, max_gen):
        samples = samples.tolist()
        for i, v in enumerate(samples):
            d = max_gen - len(v)
            if d != 0:
                avg = np.mean(v)

                extension = np.ones((d)) * avg
                extended = np.concatenate((v, extension))
                # print(f"v: {v}")
                # print(f"extended: {extended}")
                # print(f"extension: {extension}")
                # print(f"extended: {extended}")
                samples[i] = extended
                print(f"extended samples: {len(samples[i])}")
        return np.array(samples)

    baselines, base_max_gen = get_samples(baseline_f, True)
    primitives, prim_max_gen = get_samples(primitive_f)
    print(f"Raw AUC")
    print(f"baseline: {baselines}")
    print()
    # assert(False), "mean is 0. find out why."

    #extend with mean
    max_gen = max(base_max_gen, prim_max_gen)
    baselines = extend_mean(baselines, max_gen)
    primitives = extend_mean(primitives, max_gen)


    #variables[0] = [1.25 * f for f in variables[1]]
    print(f"baseline: {baselines}")
    # assert(False), "mean is 0. find out why."
    print(f"primitives: {primitives}")
    base_muvar = compute_mean_variance(baselines)
    prim_muvar = compute_mean_variance(primitives)
    print(f"Mean and Variance")
    print(f"baseline: {base_muvar}")
    print(f"primitives: {prim_muvar}")
    print()
    
    #ParetoAUC boxplot over time
    plot_boxplot(to_numpy(baselines), title="Baseline")
    plot_boxplot(to_numpy(primitives), title="Stemmatizer/Sentiment")

    #compute pvalues
    mean1, var1 = prim_muvar
    mean2, var2 = base_muvar
    pvalues = compute_pvalues(mean1, var1, len(primitives), mean2, var2, len(baselines))
    print(f"P_Values")
    print(f"P_Values: {pvalues}")
    print()

    #TODO plot pvalues over time
    plot_pvalues(pvalues, "Primitive vs Baseline, Average Pareto AUC")


def load_pareto_fronts(csv_paths):
    """
    Args:
        csv_path: path to csv file containing information from a single EMADE run.
        assume each line is a pareto individual with hash_id, tree, objectives, and generation number.

    Returns:
        pareto_fronts: 1D numpy array of ParetoFront objects
        where pareto_fronts[i] = pareto front for generation i
    """
    paretos = [] #list of ParetoFront objects
    for csv_path in csv_paths:
        d = {} #d: gen number -> list of individuals
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)


            #get unique individuals.
            # for some reason pareto_front_table has duplicates.
            unique = set()
            for index, row in enumerate(reader):
                # if index == 0: continue
                gen = row["evaluation_gen"] #get gen number.
                hash_id = row["hash"] 
                tree = row["tree"]
                obj = [float(row["FullDataSet RMS Error"]), float(row["FullDataSet Accuracy Score"])]
                ind = Individual(gen, hash_id, tree, obj)

                unique.add(ind)

            #for some reasone pareto_front_table is cumulative
            # so we'll keep a cumulative pareto fronts and filter for dominance.
            #
            #d[i] = list of inds with (gen <= i)
            unique_inds = sorted(list(unique), key=lambda x: x.gen)
            for i in unique_inds:
                print("+========+")
                print(i)
                print("+========+")
            assert(False)
            gens = [i.gen for i in unique_inds]
            for gi, g in enumerate(gens):
                d[g] = pareto_filter(unique_inds[:gi+1])


    for gen, inds in d.items():
        paretos.append(ParetoFront(gen, inds))
        paretos[-1].compute_auc()
        print(f"added ParetoFront with auc: {paretos[-1].auc}")
    pareto_fronts = np.array(paretos)
    return pareto_fronts


def plot_pareto_front_single_gen(paretoFront):
    """ plots 2D pareto front for a single gen
    gen (int) : generation number
    paretoIndObjects (list) : Individual objects.
    """
    auc = paretoFront.auc
    gen = paretoFront.gen
    paretoIndObjects = list(paretoFront.inds) #so we can sort by 1st obj

    paretoIndObjects.sort() #sort by 1st obj
    obj = [np.array([ind.obj[0], ind.obj[1]]) for ind in paretoIndObjects]
    obj = np.unique(np.array(obj), axis=0)

    print(f"obj: {obj}")

    xs = [o[0] for o in obj]
    ys = [o[1] for o in obj]

    print(f"xs: {xs}")
    print(f"ys: {ys}")


    fig, ax = plt.subplots(figsize=(10,5))
    ax.step(xs, ys, where="post")
    ax.scatter(xs, ys)


    def format(num, dec=5):
        if type(num) == tuple or type(num) == list:
            for i, n in enumerate(num):
                num[i] = round(n, dec)
        else:
            num = round(num, dec)
        return num

    #sort by gen for tree
    paretoIndObjects.sort(key=lambda x: x.gen)
    base_title = f'Pareto Front | Gen: {gen} | AUC: {format(auc)}\n'
    obj_title = "\n".join([f"G ({ind.gen}) | RMSE, ACC ({format(ind.obj)})" for ind in paretoIndObjects])
    title = base_title + obj_title
    ax.set(xlabel=f'Root Mean Squared Error', ylabel=f'Accuracy',
           title=title)

    # Used to return the plot as an image array
    plt.tight_layout()
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    plot_image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # #####################################################
    # plt.clf()
    # fig, ax = plt.subplots(figsize=(10,5))
    # ax.step(xs, ys, where="post")
    # ax.scatter(xs, ys)

    # base_title = f'Pareto Front\nGen: {gen}\nAUC: {auc}\n'
    # obj_title = "\n".join([f"{ind.tree}" for ind in paretoIndObjects[0:-1]])
    # title = base_title + obj_title
    # ax.scatter([], [])
    # ax.set_title(title)

    # plt.tight_layout()
    # fig.canvas.draw()       # draw the canvas, cache the renderer
    # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    # info_image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return plot_image, None


import imageio
def plot_pareto_front_all_gen(filename, paretoFrontObjects):
    #https://ndres.me/post/matplotlib-animated-gifs-easily/
    """ saves pareto front plots across time into gif file.
    filename (str) : name of gif file.
    paretoFrontObjects (list) : ParetoFront objects.
    """
    paretoFrontObjects.sort()
    for i in paretoFrontObjects:
        print(f"gen: {i.gen}")
    
    plot_arr = []
    info_arr = []
    for pf in paretoFrontObjects:
        plot, info = plot_pareto_front_single_gen(pf)
        plot_arr.append(plot)
        info_arr.append(info)

    imageio.mimsave(f'./{filename}', plot_arr, fps=0.75)
    # front = filename[:-4]
    # back = filename[-4:]
    # imageio.mimsave(f'./{front}_plot{back}', plot_arr, fps=0.75)
    # imageio.mimsave(f'./{front}_info{back}', plot_arr, fps=0.75)   


def freq():
    start = "VIP_AAD_NeuralNet/"
    baseline_f = [
            ("movie_reviews_nonnull_baseline_nlp-app_3.csv", "movie_reviews_paretofront_baseline_nlp-app_3.csv"),
            ("April 17th/movie_reviews_all_non-null_baseline_6.csv", "April 17th/movie_reviews_all_pareto_baseline_6.csv"),
            ("April 17th/movie_reviews_all_non-null_baseline_5.csv", "April 17th/movie_reviews_all_pareto_baseline_5.csv")
            ]

    primitive_f = [
            ("movie_reviews_nlp-app_1.csv", "movie_reviews_nlp-app_paretofront_1.csv"),
            ("movie_reviews_all_non-null_nlp-app_3.csv", "movie_reviews_paretoindividuals_nlp-app_3.csv"),
            ("movie_reviews_all_non-null_nlp_app_4.csv", "movie_reviews_nlp-app_paretofront_4.csv"),
            ("April 11th/movie_reviews_all_non-null_nlp_app_5.csv", "April 11th/movie_reviews_nlp-app_paretofront_5.csv"),
            ("April 17th/movie_reviews_all_non-null_nlp_app_6.csv",  "April 17th/movie_reviews_all_pareto_nlp_app_6.csv")
            ]
    
    def get_samples(filenames, debug=False):
        samples = []
        gen_cut = np.inf
        max_gen = -1
        for f in filenames:
            if type(f) == tuple: #requires joining of 2 files.
                p = paretos(join(start, f), debug=debug) #list of pareto objects
            elif type(f) == str: #csv already joined.
                p = load_pareto_fronts([start+f])
            samples.append(p)

            gen_cut = min(gen_cut, len(p))

        if debug:
            print(f"gen_cut: {gen_cut}")
            print(f"max_gen: {max_gen}")
            print(f"samples: {samples}")
        return samples, gen_cut

    baselines, base_max_gen = get_samples(baseline_f, True)
    primitives, prim_max_gen = get_samples(primitive_f)


    # reduce to min gens
    # for i, b in enumerate(baselines):
    #     baselines[i] = b[:base_max_gen]

    # for i, b in enumerate(baselines):
    #     baselines[i] = b[:base_max_gen]

    print(f"baselines: {baselines}")


    #find max gen sample
    longest_base = 0
    base_index = 0
    for bi, b in enumerate(baselines):
        if len(b) > longest_base:
            longest_base = len(b)
            base_index = bi
    print(f"max gen bas: {longest_base}")
    # print(f"max gen bas: {baselines[base_index]}")


    longest_prim = 0
    prim_index = 0
    for bi, b in enumerate(primitives):
        if len(b) > longest_prim:
            longest_prim = len(b)
            prim_index = bi
    print(f"max gen prim: {longest_prim}")
    # print(f"max gen prim: {primitives[prim_index]}")
    # assert(False)

    #aggregate into a single pf object
    new_base = []
    new_prim = []
    # for i in range(len(baselines)):
    #     b = ParetoFront(i, [])
    #     for bb in baselines[i]:
    #         b.list_inds += bb.list_inds
    # for i in range(len(primitives)):
    #     p = ParetoFront(i, [])
    #     for pp in primitives[i]:
    #         p.list_inds += pp.list_inds
    #     new_base.append(b)
    #     new_prim.append(p)



    # arr[g] = list of trees
    def get_trees(fronts):
        trees = []
        for pf in fronts:
            gen_trees = []
            for ind in pf.inds:
                gen_trees.append(ind.tree)
            trees.append(gen_trees)
        return trees

    

    # arr[g] = float rep proportion of linear classifiers
    def get_freq(trees):
        f = []
        for g in trees:
            lin_count = 0.0
            for t in g:
                if "LINSVC" in t:
                    lin_count += 1.0
            f.append(lin_count / len(g))


        default = {"ARGMAX":0, "ARGMIN":0,
                 "KNN":0, "SVM":0, "KMEANS":0,
                 "RAND_FOREST":0, 
                 "BOOSTING":0, 
                 "DECISION_TREE":0,
                 "LOGR":0, "LINSVC":0, "SGD":0,
                 "PASSIVE":0,
                 "EXTRATREES":0,
                 "XGBOOST":0,
                 "LIGHTGBM":0,
                 "BOOSTING_REGRESSION":0,
                 "ADABOOST_REGRESSION":0,
                 "RANDFOREST_REGRESSION":0,
                 "SVM_REGRESSION":0,
                 "KNN_REGRESSION":0}
        d_list = []
        for g in trees:
            d = default
            

            for t in g:
                for k in default.keys():
                    if k in t:
                        d[k] += 1.0
                        break

            max_f = 0
            max_k = None
            for k, v in d.items():
                if d[k] > max_f:
                    max_f = d[k]
                    max_k = k

            d_list.append(max_k)
        return f, d_list


    # first_base = baselines[0].inds
    # agg_base = ParetoFront(first_base)
    # for i in range(len(baselines)):
    #     if i == 0: continue
    #     agg_base.inds += baselines[i].inds

    # first_prim = primitives[0].inds
    # agg_prim = ParetoFront(first_prim)
    # for i in range(len(primitives)):
    #     if i == 0: continue
    #     agg_prim.inds += primitives[i].inds

    new_base = baselines[base_index]
    new_prim = primitives[prim_index]
    
    base_t = get_trees(new_base)
    prim_t = get_trees(new_prim)

    # print(f"new_base: {new_base}")
    # print(f"base_t: {base_t}")
    # assert(False)


    """

    self.names = [#"DEPTH_ESTIMATE",
                 "ARGMAX", "ARGMIN",
                 #"GMM",
                 "KNN", "SVM", "KMEANS",
                 "RAND_FOREST", 
                 "BOOSTING", 
                 "DECISION_TREE",
                 "LOGR", "LINSVC", "SGD",
                 "PASSIVE",
                 "EXTRATREES",
                 "XGBOOST",
                 "LIGHTGBM",
                 "BOOSTING_REGRESSION",
                 "ADABOOST_REGRESSION",
                 "RANDFOREST_REGRESSION",
                 "SVM_REGRESSION",
                 "KNN_REGRESSION"]

    """


    base_f, base_max_k = get_freq(base_t)
    prim_f, prim_max_k = get_freq(prim_t)

    print(f"base: max key: {base_max_k}")
    print(f"prim: max key: {prim_max_k}")

    plt.suptitle("Baseline frequency of LINEARSVC")
    xs = range(len(base_f))
    ys = base_f
    plt.plot(xs, ys)
    plt.show()

    plt.suptitle("Primitive frequency of LINEARSVC")
    xs = range(len(prim_f))
    ys = prim_f
    plt.plot(xs, ys)
    plt.show()

import argparse as ap

if __name__ == "__main__":
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.size'] = 12

    parser = ap.ArgumentParser(description='visualization of pareto front from EMADE runs.')
    parser.add_argument('mode', choices=["compare", "single"], help='')
    parser.add_argument('--file', default="ParetoFront.csv", help='')
    parser.add_argument('--title', nargs="*", help='')
    
    args = parser.parse_args()

    #TODO: add checks here.

    if args.mode == "compare":
        #freq()
        nlp_app_test()
    else:
        TYPE = "else"
        pf_objects = load_pareto_fronts([args.file])

        for pf in pf_objects:
            pf.compute_auc()
            print(pf)
        # assert(False)
        plot_pareto_front_all_gen("newStudentNLP_NN.gif", pf_objects)



    

