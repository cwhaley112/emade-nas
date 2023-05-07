# from GPFramework.analysis.analyze import Individual
from GPFramework.sql_connection_orm_base import SQLConnection, IndividualStatus
from sqlalchemy import func
from numpy import isinf, isnan
from random import random
from datetime import datetime
import time
from heapq import heapify, heappop
import pandas as pd
import os
import sys
import os.path
from shutil import rmtree
import numpy as np
from collections import Counter

class Node:

    def __init__(self, value, weight, key):
        self.value = value
        self.weight = weight
        self.key = key

    def __lt__(self, other): # For x < y
        if not isinstance(other, Node):
            return False
        return self.value < other.value
    def __le__(self, other): # For x <= y
        if not isinstance(other, Node):
            return False
        return self.value <= other.value
    def __eq__(self, other): # For x == y
        if not isinstance(other, Node):
            return False
        return self.value == other.value
    def __ne__(self, other): # For x != y
        if not isinstance(other, Node):
            return False
        return self.value != other.value
    def __gt__(self, other): # For x > y
        if not isinstance(other, Node):
            return False
        return self.value > other.value
    def __ge__(self, other): # For x >= y
        if not isinstance(other, Node):
            return False
        return self.value >= other.value

class SQLConnectionMaster(SQLConnection):
    """
    This class is a subclass of SQLConnection for master_algorithm and worker_algorithm transactions
    """

    def __init__(self, connection_str=None, reuse=None, fitness_names=None, dataset_names=None, statistics_dict=None, cache_dict=None, is_worker=False):
        """
        Initializes the SQLConnection object. Assigns return values from get_session as instance variables
        """
        super().__init__(connection_str, reuse, fitness_names, dataset_names, statistics_dict, cache_dict, is_worker=is_worker)

    def select(self, hash, isModule=False):
        """
        Retrieves an individual from the database using its hash and specified table which defaults to individuals table.
        Args:
            hash: hash of the individual or other object with hash
        Returns:
            Queried individual or other object type
        Note: Modules use mod_num in place of hash value
        """
        queried = False
        while not queried:
            try:
                # print("Before Individual Select Count | " + str(datetime.now()))
                # sys.stdout.flush()
                if (isModule):
                    count = self.sessions[os.getpid()].query(self.Modules).filter_by(mod_num=hash).count()
                else:
                    count = self.sessions[os.getpid()].query(self.Individual).filter_by(hash=hash).count()
                # print("After Individual Select Count | " + str(datetime.now()))
                # sys.stdout.flush()
                queried = True
            except Exception as e:
                print('Got error: ', e, '|| While counting all individuals with the same hash, waiting and retrying')
                sys.stdout.flush()
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())

        if count > 1:
            print('Got more than one individual with the same hash!')
            sys.stdout.flush()
        
        queried = False
        while not queried:
            try:
                # print("Before Individual Select | " + str(datetime.now()))
                # sys.stdout.flush()
                if (isModule):
                    query = self.sessions[os.getpid()].query(self.Modules).filter_by(mod_num=hash).first()
                else:
                    query = self.sessions[os.getpid()].query(self.Individual).filter_by(hash=hash).first()
                # print("After Individual Select | " + str(datetime.now()))
                # sys.stdout.flush()
                queried = True
            except Exception as e:
                print('Got error: ', e, '|| While selecting a individuals, waiting and retrying')
                sys.stdout.flush()
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())
        return query

    def selectNN(self, hash, age):
        """
        Retrieves an NNLearner from the database using its hash.
        Args:
            hash: hash of the individual
            age: age of the individual
        Returns:
            Queried individual
        """
        queried = False
        while not queried:
            try:
                # print("Before Individual Select Count | " + str(datetime.now()))
                # sys.stdout.flush()
                count = self.sessions[os.getpid()].query(self.NNStatistics).filter_by(hash=hash, age=age).count()
                # print("After Individual Select Count | " + str(datetime.now()))
                # sys.stdout.flush()
                queried = True
            except Exception as e:
                print('Got error: ', e, '|| While counting all individuals with the same hash, waiting and retrying')
                sys.stdout.flush()
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())

        if count > 1:
            print('Got more than one individual with the same hash!')
            sys.stdout.flush()

        queried = False
        while not queried:
            try:
                # print("Before Individual Select | " + str(datetime.now()))
                # sys.stdout.flush()
                query = self.sessions[os.getpid()].query(self.NNStatistics).filter_by(hash=hash, age=age).first()
                # print("After Individual Select | " + str(datetime.now()))
                # sys.stdout.flush()
                queried = True
            except Exception as e:
                print('Got error: ', e, '|| While selecting a individuals, waiting and retrying')
                sys.stdout.flush()
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())
        return query

    def insertInd(self, hash, tree, individual, age, evaluation_status, evaluation_gen=-1,
               elapsed_time=0, retry_time=0, evaluation_start_time=None, error_string=None):
        """
        Inserts an individual into the database
        Args:
            hash: hash of the individual
            tree: string representation of the individual
            individual: the individual itself
            age: age of the individual
            evaluation_status: evaluation status of the individual
            evaluation_gen: evaluation year of the individual
            elapsed_time: elapsed time of the individual
            retry_time: time spent retrying individual
            evaluation_start_time: evaluation start time of the individual
            error_string: error produced during evaluation
        """
        # print("Starting Individual Insertion | " + str(datetime.now()))
        # sys.stdout.flush()
        committed = False
        while not committed:
            try:
                # print("Attempting Individual Insertion | " + str(datetime.now()))
                # sys.stdout.flush()
                ind = self.Individual(hash=hash, tree=tree, elapsed_time=elapsed_time, retry_time=retry_time, evaluation_gen=evaluation_gen,
                        age=age.item() if isinstance(age, float) else 0., evaluation_status=evaluation_status,
                        evaluation_start_time=evaluation_start_time, pickle=individual)
                if elapsed_time:
                    ind.elapsed_time = elapsed_time
                if retry_time:
                    ind.retry_time = retry_time
                if evaluation_gen != -1:
                    ind.evaluation_gen = evaluation_gen
                if error_string:
                    ind.error_string = error_string
                for dataset in self.dataset_names:
                    for fitness_name, fitness in zip(self.fitness_names, getattr(individual, dataset).values):
                        setattr(ind, dataset + ' ' + fitness_name, fitness.item() if not isinf(fitness) or isnan(fitness) else None)
                
                self.sessions[os.getpid()].add(ind)
                self.sessions[os.getpid()].commit()
                committed = True
                # print("Successful Individual Insertion | " + str(datetime.now()))
                # sys.stdout.flush()
            except Exception as e:
                print('Got error', e, 'while committing waiting and retrying')
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())
    
    def updateInd(self, row, individual, age, evaluation_status, evaluation_gen=-1,
               elapsed_time=0, retry_time=0, evaluation_start_time=-1, error_string=None):
        """
        Updates an individual in the database
        Args:
            row: SQLAlchemy individual retrieved from database
            individual: the individual itself
            age: age of the individual
            evaluation_status: evaluation status of the individual
            evaluation_gen: evaluation year of the individual
            elapsed_time: elapsed time of the individual
            retry_time: time spent retrying individual
            evaluation_start_time: evaluation start time of the individual
            error_string: error produced during evaluation
        """
        # print("Starting Individual Update | " + str(datetime.now()))
        committed = False
        while not committed:
            try:
                # print("Attempting Individual Update | " + str(datetime.now()))
                row.age = age.item() if isinstance(age, float) else 0.
                row.pickle = individual
                row.evaluation_status = evaluation_status
                if elapsed_time:
                    row.elapsed_time = elapsed_time
                if retry_time:
                    row.retry_time = retry_time
                if evaluation_start_time != -1:
                    row.evaluation_start_time = evaluation_start_time
                if evaluation_gen != -1:
                    row.evaluation_gen = evaluation_gen
                if error_string:
                    row.error_string = error_string
                for dataset in self.dataset_names:
                    for fitness_name, fitness in zip(self.fitness_names, getattr(individual, dataset).values):
                        setattr(row, dataset + ' ' + fitness_name, fitness.item() if not isinf(fitness) or isnan(fitness) else None)
                self.sessions[os.getpid()].commit()
                committed = True
                # print("Successful Individual Update | " + str(datetime.now()))
                # sys.stdout.flush()
            except Exception as e:
                print('Got error', e, 'while committing waiting and retrying')
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())

    def insertMod(self, mod_num, age, tree, pickle, num_occur=0, error_string=None, weights=None):
        """
        Adds a new module to the modules table 
        Leaving a weights argument in case we want to load weights while seeding in the future.
        """

        committed = False
        unique_num = True
        failed=False
        while not committed:
            try:
                if not unique_num:
                    mod_num += randint(1,100) # got primary key error for some reason
                mod = self.Modules(mod_num=mod_num, age=age, num_occur=num_occur, tree=tree, pickle=pickle)
                if error_string:
                    mod.error_string
                if weights:
                    if not failed:
                        mod.weights = weights
                    else:
                        mod.weights = None
                if failed:
                    pickle.weights = None
                for dataset in self.dataset_names:
                    for fitness_name, fitness in zip(self.fitness_names, getattr(pickle, dataset).values):
                        setattr(mod, dataset + ' ' + fitness_name, fitness.item() if not isinf(fitness) or isnan(fitness) else None)
                
                self.sessions[os.getpid()].add(mod)
                self.sessions[os.getpid()].commit()
                committed = True
            except Exception as e:
                print('insertModGot error', e, 'while committing waiting and retrying')
                if failed:
                    unique_num=False
                failed=True
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())
                from random import randint

    def updateMod(self, row, mod_num, age, tree, pickle, num_occur=0, error_string=None, weights=None):
        """
        Updates an modules after mating/mutation and resets the saved weights
        """
        committed = False
        failed=False
        while not committed:
            try:
                row.mod_num=mod_num 
                row.age=age
                row.num_occur=num_occur
                row.tree=tree
                if error_string:
                    row.error_string
                if weights:
                    if not failed:
                        row.weights = weights
                    else:
                        row.weights = None # very common to get an error from query being too big (due to these weights)
                if pickle:
                    if failed:
                        pickle.weights = None
                    row.pickle = pickle
                for dataset in self.dataset_names:
                    for fitness_name, fitness in zip(self.fitness_names, getattr(pickle, dataset).values):
                        setattr(row, dataset + ' ' + fitness_name, fitness.item() if not isinf(fitness) or isnan(fitness) else None)
                self.sessions[os.getpid()].commit()
                committed = True
            except Exception as e:
                print('UpdateMod Got error', e, 'while committing waiting and retrying')
                failed=True
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())

    def insertNN(self, hash, age, parents, curr_tree, individual, error_string=None):
        """
        Inserts an individual into the statistics table if NNLearner
        """
        # print("Starting Individual Insertion | " + str(datetime.now()))
        # sys.stdout.flush()
        committed = False
        while not committed:
            try:
                # print("Attempting Individual Insertion | " + str(datetime.now()))
                # sys.stdout.flush()
                
                ind = self.NNStatistics(
                    hash=hash, 
                    age=age.item() if isinstance(age, float) else 0.,
                    parents=parents,
                    current_tree=curr_tree,
                    error_string=error_string
                )
                for dataset in self.dataset_names:
                    for fitness_name, fitness in zip(self.fitness_names, getattr(individual, dataset).values):
                        setattr(ind, dataset + ' ' + fitness_name, fitness.item() if not isinf(fitness) or isnan(fitness) else None)
                
                self.sessions[os.getpid()].add(ind)
                self.sessions[os.getpid()].commit()
                committed = True
                # print("Successful Individual Insertion | " + str(datetime.now()))
                # sys.stdout.flush()
            except Exception as e:
                print('Got error', e, 'while committing waiting and retrying')
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())

    def updateNN(self, row, individual, age, parents, error_string=None):
        """
        Updates an individual in the database
        Args:
            row: SQLAlchemy individual retrieved from database
            individual: the individual itself
            age: age of the individual
            error_string: error produced during evaluation
        """
        # print("Starting Individual Update | " + str(datetime.now()))
        committed = False
        while not committed:
            try:
                # print("Attempting Individual Update | " + str(datetime.now()))

                row.age = age.item() if isinstance(age, float) else 0.
                row.pickle = individual
                if parents:
                    row.parents = parents
                if error_string:
                    row.error_string = error_string
                for dataset in self.dataset_names:
                    for fitness_name, fitness in zip(self.fitness_names, getattr(individual, dataset).values):
                        setattr(row, dataset + ' ' + fitness_name, fitness.item() if not isinf(fitness) or isnan(fitness) else None)

                self.sessions[os.getpid()].commit()
                committed = True
                # print("Successful Individual Update | " + str(datetime.now()))
                # sys.stdout.flush()
            except Exception as e:
                print('Got error', e, 'while committing waiting and retrying')
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())

    def add_history(self, gens_elapsed, hashes):
        """
        Adds hashes of individuals that were just evaluated to the History table.
        Args:
            gens_elapsed: number of gens elapsed on master worker
            hashes: hashes of all the individuals considered
        """
        queried = False
        while not queried:
            try:
                print("Before History Addition | " + str(datetime.now()))
                sys.stdout.flush()
                for hash in hashes:
                    self.sessions[os.getpid()].add(self.History(optimization=self.optimization, generation=gens_elapsed, hash=hash))
                    self.sessions[os.getpid()].commit()
                print("After History Addition | " + str(datetime.now()))
                sys.stdout.flush()
                queried = True
            except Exception as e:
                print('Got error: ', e, '|| While adding to History Table, waiting and retrying')
                sys.stdout.flush()
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())

    def update_layer_frequencies(self, generation: int, individual_layer_frequencies: Counter) -> None:
        """Collate the layer frequencies within individuals across a generation."""
        statistics = self.sessions[os.getpid()].query(self.Statistics).filter(generation == generation).first()
        if not statistics:
            statistics = self.Statistics(generation=generation, layer_frequencies=Counter(), layer_frequencies_str='')
        statistics.layer_frequencies = statistics.layer_frequencies + individual_layer_frequencies
        statistics.layer_frequencies_str = str(statistics.layer_frequencies)
        self.sessions[os.getpid()].add(statistics)
        self.sessions[os.getpid()].commit()

    def get_layer_frequencies(self, generation: int) -> Counter:
        """Get the collated layer frequencies for the given generation."""
        statistics = self.sessions[os.getpid()].query(self.Statistics).filter(self.Statistics.generation == generation).first()
        return statistics.layer_frequencies if statistics else Counter()

    def add_host(self, id):
        """
        Adds a new host to the Host table
        Must be done before a Worker machine begins evaluating
        Args:
            id: unique string id of the host
        """
        if self.sessions[os.getpid()].query(self.Host).filter_by(id=id).count() < 1:
            self.sessions[os.getpid()].add(self.Host(id=id))
            self.sessions[os.getpid()].commit()

    def add_pareto_front(self, gens_elapsed, hashes):
        """
        Adds hashes of individuals on pareto front to the ParetoFront table.
        Args:
            gens_elapsed: number of gens elapsed on master worker
            hashes: hashes of all the individuals considered
        """
        queried = False
        while not queried:
            try:
                print("Before Pareto Front Addition | " + str(datetime.now()))
                sys.stdout.flush()
                for hash in hashes:
                    self.sessions[os.getpid()].add(self.ParetoFront(optimization=self.optimization, generation=gens_elapsed, hash=hash))
                    self.sessions[os.getpid()].commit()
                print("After Pareto Front Addition | " + str(datetime.now()))
                sys.stdout.flush()
                queried = True
            except Exception as e:
                print('Got error: ', e, '|| While adding to Pareto Front Table, waiting and retrying')
                sys.stdout.flush()
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())

    def get_seeded_pareto(self):
        """
        Retrieves non-dominated individuals to seed upon starting a new run of
        EMADE from the latest generation of the latest optimization
        from the paretofront table. Note that this assumes that
        this method is called before a new optimization is written to the db.
        Corresponding SQL query:
            SELECT * FROM `db`.`paretofront` WHERE generation =
                (SELECT MAX(`paretofront`.`generation`) FROM `db`.`paretofront`
                WHERE optimization = (SELECT MAX(`paretofront`.`optimization`)
                FROM `db`.`paretofront`))
                AND (optimization = (SELECT MAX(`paretofront`.`optimization`)
                FROM `db`.`paretofront`));
        Returns:
            All individuals on latest non-dominated front
        """
        queried = False
        while not queried:
            try:
                print("Before Pareto Front Retrieval | " + str(datetime.now()))
                sys.stdout.flush()
                lastOpt = self.optimization - 1
                lastGen = self.sessions[os.getpid()].query(func.max(self.ParetoFront.generation))\
                                                    .filter(self.ParetoFront.optimization == lastOpt).first()[0]

                #use a subquery instead of passing around data locally:
                ParetoFrontLatestGen = self.sessions[os.getpid()].query(self.ParetoFront.hash)\
                                                                .filter(self.ParetoFront.optimization == lastOpt)\
                                                                .filter(self.ParetoFront.generation == lastGen).subquery()

                individuals = [ind.pickle for ind in self.sessions[os.getpid()].query(self.Individual)\
                                                                            .filter(self.Individual.hash.in_(ParetoFrontLatestGen))]
                print("After Pareto Front Retrieval | " + str(datetime.now()))
                sys.stdout.flush()
                queried = True
            except Exception as e:
                print('Got error: ', e, '|| While querying Pareto Front Table, waiting and retrying')
                sys.stdout.flush()
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())
        return individuals

    def get_num_evaluated(self):
        """
        Retrieves count of evaluated individuals
        Returns:
            Count of evaluated individuals
        """
        queried = False
        while not queried:
            try:
                print("Before Uneval Count Query | " + str(datetime.now()))
                sys.stdout.flush()
                query = self.sessions[os.getpid()].query(func.count('*')).filter(self.Individual.evaluation_status == IndividualStatus.EVALUATED).scalar()
                print("After Uneval Count Query | " + str(datetime.now()))
                sys.stdout.flush()
                queried = True
            except Exception as e:
                print('Got error: ', e, '|| While counting unevaluated individuals, waiting and retrying')
                sys.stdout.flush()
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())
        return query

    def get_num_waiting(self):
        """
        Retrieves count of evaluated individuals
        Returns:
            Count of evaluated individuals
        """
        queried = False
        while not queried:
            try:
                print("Before Uneval Count Query | " + str(datetime.now()))
                sys.stdout.flush()
                query = self.sessions[os.getpid()].query(func.count('*')).filter(self.Individual.evaluation_status == IndividualStatus.WAITING_FOR_MASTER).scalar()
                print("After Uneval Count Query | " + str(datetime.now()))
                sys.stdout.flush()
                queried = True
            except Exception as e:
                print('Got error: ', e, '|| While counting unevaluated individuals, waiting and retrying')
                sys.stdout.flush()
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())
        return query

    def get_unevaluated(self):
        """
        Retrieves unevaluated individuals
        Returns:
            Unevaluated individuals
        """
        queried = False
        while not queried:
            try:
                print("Before Uneval Count Query | " + str(datetime.now()))
                sys.stdout.flush()
                query = self.sessions[os.getpid()].query(func.count('*')) \
                                                  .filter((self.Individual.evaluation_status == IndividualStatus.NOT_EVALUATED) | (self.Individual.evaluation_status == IndividualStatus.IN_PROGRESS)).scalar()
                print("After Uneval Count Query | " + str(datetime.now()))
                sys.stdout.flush()
                queried = True
            except Exception as e:
                print('Got error: ', e, '|| While counting unevaluated individuals, waiting and retrying')
                sys.stdout.flush()
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())
        return query

    def get_evaluated_individuals(self):
        """
        Retrieves recently evaluated individuals
        Returns:
            Recently evaluated individuals
        """
        queried = False
        while not queried:
            try:
                print("Before Recent Evaluated Query | " + str(datetime.now()))
                sys.stdout.flush()
                inds = self.sessions[os.getpid()].query(self.Individual) \
                                                 .filter(self.Individual.evaluation_status == IndividualStatus.WAITING_FOR_MASTER).all()
                print("After Recent Evaluated Query | " + str(datetime.now()))
                sys.stdout.flush()
                queried = True
            except Exception as e:
                print('Got error: ', e, '|| While querying recently evaluated individuals, waiting and retrying')
                sys.stdout.flush()
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())
        return [ind.pickle for ind in inds]

    def get_current_ascs(self):
        """
        Retrieves recently evaluated individuals
        Returns:
            Recently evaluated individuals
        """
        queried = False
        while not queried:
            try:
                print("Before Recent Evaluated Query | " + str(datetime.now()))
                sys.stdout.flush()
                inds = self.sessions[os.getpid()].query(self.ADFS).all()
                print("After Recent Evaluated Query | " + str(datetime.now()))
                sys.stdout.flush()
                queried = True
            except Exception as e:
                print('Got error: ', e, '|| While querying recently evaluated individuals, waiting and retrying')
                sys.stdout.flush()
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())
        return [ind.pickle for ind in inds]

    def get_random_uneval_individuals(self, num_ind):
        """
        Retrieves random unevaluated individuals
        Args:
            num_ind: number of random individuals to get
        Returns:
            Recently evaluated individuals
        """
        queried = False
        while not queried:
            try:
                print("Before Uneval Query | " + str(datetime.now()))
                sys.stdout.flush()
                query = self.sessions[os.getpid()].query(self.Individual) \
                    .filter(self.Individual.evaluation_status == IndividualStatus.NOT_EVALUATED) \
                    .order_by(func.rand()) \
                    .limit(int(num_ind)) \
                    .with_for_update().all()
                print("After Uneval Query | " + str(datetime.now()))
                sys.stdout.flush()
                for ind in query:
                    ind.evaluation_status = IndividualStatus.IN_PROGRESS
                    ind.evaluation_start_time = datetime.now()
                print("Status Updated | " + str(datetime.now()))
                self.sessions[os.getpid()].commit()
                print("Lock Released | " + str(datetime.now()))
                sys.stdout.flush()
                queried = True
            except Exception as e:
                print('Got error: ', e, '|| While gathering individuals, waiting and retrying')
                sys.stdout.flush()
                # Sleep 10 seconds and change and then loop
                self.sessions[os.getpid()].rollback()
                time.sleep(10+random())

        return [ind.pickle for ind in query]

    def get_pareto(self, gen, optimization=0):
        stmt = self.sessions[os.getpid()].query(self.Individual) \
            .join(self.ParetoFront) \
            .filter(self.ParetoFront.generation == gen)\
            .filter(self.ParetoFront.optimization == optimization)
        return stmt.all()

    def num_gens(self):
        return self.sessions[os.getpid()].query(func.max(self.Statistics.generation)).first()[0]
        # return self.sessions[os.getpid()].query(func.max(self.ParetoFront.generation)).first()[0] + 1
    
    def reset_cache_rows(self, host):
        query = self.sessions[os.getpid()].query(self.Cache).filter(self.Cache.host_id == host, self.Cache.dirty != 0)
        for row in query:
            row.dirty = 0
            row.gens_since_last_hit = 0
            row.hits_last_gen = 0
        self.commit()
    
    def heap_invalidation(self, nodes, capacity, host, current_size):
        """
        Uses a heap to remove low fitness nodes from the cache
        First builds a min heap from the list of nodes
        Then removes nodes from the root of the heap until capacity is met
        
        O(n log n) where n is the length of nodes 
        Args:
            nodes: list of nodes containing information about each cache key
            capacity: the maximum 'weight' or cost we can store
            host: unique id of host machine
            current_size: current size of the cache
            
        Returns:
            None
        """
        heapify(nodes)
        # Cut the cache size down to 80% and let it build back up next gen
        while current_size >= capacity * 0.8:
            # Pop root of heap O(log n)
            smallest_node = heappop(nodes)
            # Query row from db and lock it
            not_queried = True
            while (not_queried):
                try:
                    smallest_row = self.sessions[os.getpid()].query(self.Cache).with_for_update().filter(self.Cache.id == smallest_node.key, self.Cache.host_id == host).first()
                    not_queried = False
                except:
                    print("Error when querying cache row for removal. (Most likely a lock timeout)")
                    time.sleep(10+random())
            # Update row to show that data was removed
            smallest_row.dirty = 1
            # Update current size
            current_size -= smallest_row.size
            # Delete cached data from local disk
            try:
                rmtree(smallest_row.train_directory + smallest_row.id)
            except Exception as e:
                print("Failed to delete training directory when removing cache entry")
                print("Cache id:", smallest_node.key)
                print("Exception:", e)
            try:
                rmtree(smallest_row.test_directory + smallest_row.id)
            except Exception as e:
                print("Failed to delete testing directory when removing cache entry")
                print("Cache id:", smallest_node.key)
                print("Exception:", e)
            # Release Lock
            self.sessions[os.getpid()].commit()
            
    def sort_and_greed(self, nodes, capacity, host):
        """
        Sorts the cache entry using Timsort
        Then iterates through the entire sorted list
        When a node is found it's weight is added to the current running sum
        If the running sum is less than the capacity after adding the weight
        Then that node is not pruned from the cache
        Else prune the node from the cache
        
        O(n log n) where n is the length of nodes 
        Args:
            nodes: list of nodes containing information about each cache key
            capacity: the maximum 'weight' or cost we can store
            host: unique id of host machine
            
        Returns:
            None
        """
        # Sort nodes
        nodes = sorted(nodes)
        
        # Iterate from highest value to lowest value
        running_sum = 0
        total_value = 0
        for i in range(len(nodes) - 1, -1, -1):
            if running_sum + nodes[i].weight <= capacity:
                # Keep this entry in the cache
                running_sum += nodes[i].weight
                # Update total value of cache
                total_value += nodes[i].value
            else:
                # Query row from db and lock it
                not_queried = True
                while (not_queried):
                    try:
                        row_to_del = self.sessions[os.getpid()].query(self.Cache).with_for_update().filter(self.Cache.id == nodes[i].key, self.Cache.host_id == host).first()
                        not_queried = False
                    except:
                        print("Error when querying cache row for removal. (Most likely a lock timeout). Cache id: {}".format(nodes[i].key))
                        time.sleep(10+random())
                # Update row to show that data was removed
                row_to_del.dirty = 1
                # Delete cached data from local disk
                try:
                    rmtree(row_to_del.train_directory + row_to_del.id)
                except Exception as e:
                    print("Failed to delete training directory when removing cache entry")
                    print("Cache id:", nodes[i].key)
                    print("Exception:", e)
                try:
                    rmtree(row_to_del.test_directory + row_to_del.id)
                except Exception as e:
                    print("Failed to delete testing directory when removing cache entry")
                    print("Cache id:", nodes[i].key)
                    print("Exception:", e)
                # Release Lock
                self.sessions[os.getpid()].commit()
        
        return total_value
    
    def knapsack(self, nodes, capacity, host, percentage_max=None, NUM_BINS=None):
        """
        Finds the best combination of values given their weights (using a dyanmic
        programming solution to the 0-1 knapsack problem)
        Used for optimizing cache invalidation so we find a better solution than the previous
        take-elements-off-a-heap-until-we're-under-max_size solution
        
        O(nW) where n is the length of nodes and W is capacity
        Then O(n) node removal
        Args:
            nodes: list of nodes containing information about each cache key
            capacity: the maximum 'weight' or cost we can store
            host: unique id of host machine
            percentage_max: if not None, a float of the percentage of max_weight we want to use (usually about .01)
            NUM_BINS: if not None, an int defining a fixed number of 'buckets' or array indices to use
            
        Returns:
            None
        """
        # Do nothing if there are no nodes in the cache
        if len(nodes) == 0:
            return
        
        # if len(nodes) < capacity * 0.000001:
        #     # If number of objects is significantly smaller than weight size
        #     # use square matrix
        #     capacity = len(nodes)
        # elif type(percentage_max) is float:
        #     # If defined percentage_max to a value, create number of bins based on percentage
        #     # of weight size
        #     capacity = int(capacity * percentage_max)
        # elif type(NUM_BINS) is int and int(capacity) > NUM_BINS:
        #     # If percentage max is not defined, and we don't have a very small number of objects,
        #     # and number of bins is defined and smaller than max weight, use set bin size
        #     capacity = NUM_BINS
            
        # Convert capacity to an int. apply ceiling to capacity first
        capacity = int(np.ceil(capacity))
        
        # Create 2d array of zeroes to construct solution
        K = np.zeros((len(nodes) + 1, capacity + 1), dtype=int)
    
        for i in range(len(nodes) + 1): # Iterate vertically (i changes the object we're looking at)
            for w in range(capacity + 1): # Iterate horizontally (w changes the total cost case we're looking at)
                if i == 0 or w == 0: # Pad top and left side with zero
                    K[i][w] = 0
                elif nodes[i - 1].weight <= w: # if weight of above object is less than the total weight case we're looking at
                    K[i][w] = max(nodes[i - 1].value + K[i - 1][w - nodes[i - 1].weight], K[i - 1][w]) # Store in array max of potential value
                else: # Weight is too high in this case
                    K[i][w] = K[i - 1][w] # Set equal to what we found with the previous object

        # Finds and removes all nodes not in the final knapsack solution
        w = capacity
        # Store temp of best result
        temp = K[len(nodes)][w]
        # Search for the valuable cache entries
        for i in range(len(nodes), 0, -1):
            # Check if a node should be removed
            if temp <= 0 or temp == K[i - 1][w]:
                # Query row from db and lock it
                not_queried = True
                while (not_queried):
                    try:
                        row_to_del = self.sessions[os.getpid()].query(self.Cache).with_for_update().filter(self.Cache.id == nodes[i - 1].key, self.Cache.host_id == host).first()
                        not_queried = False
                    except:
                        print("Error when querying cache row for removal. (Most likely a lock timeout)")
                        time.sleep(10+random())
                # Update row to show that data was removed
                row_to_del.dirty = 1
                # Delete cached data from local disk
                try:
                    rmtree(row_to_del.train_directory + row_to_del.id)
                except Exception as e:
                    print("Failed to delete training directory when removing cache entry")
                    print("Cache id:", nodes[i - 1].key)
                    print("Exception:", e)
                try:
                    rmtree(row_to_del.test_directory + row_to_del.id)
                except Exception as e:
                    print("Failed to delete testing directory when removing cache entry")
                    print("Cache id:", nodes[i - 1].key)
                    print("Exception:", e)
                # Release Lock
                self.sessions[os.getpid()].commit()
            else:
                # Found a node included in optimal solution
                # Update values to search for next optimal node
                temp = temp - nodes[i - 1].value
                w = w - nodes[i - 1].weight

    def create_db_statistics(self, host):
        """
        Creates statistics using database queries
        Then stores the statistics in a csv format
        Args:
            host: id of host machine ('central' if centralized)
        """
        start_time = time.time()
        # Note: all non-removal queries read-only and do not lock rows
        # This means db information may change during statistic calculation

        non_init = True
        filename = "db_info{}.csv".format(os.getpid())

        # Create csv file and pandas dataframe
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=['cacheSize', 'totalKeys', 'validKeys', 
                                       'keysAdded', 'cacheHits', 'totalEvals', 
                                       'totalWrites', 'totalReads', 'writeTime', 
                                       'readTime', 'writeOverheadTime', 'readOverheadTime', 
                                       'methodOverheadTime', 'timeWithCache', 'timeWithoutCache', 
                                       'totalMethods', 'createStatsTime', 'evalInds', 'elapsedTime'])
            non_init = False
            
        self.sessions[os.getpid()].execute("START TRANSACTION;")
        # Calculate stats for all valid cache rows
        cache_size = self.sessions[os.getpid()].execute("SELECT SUM(size) from cache WHERE host_id='{}' and dirty=0;".format(host)).scalar()
        valid_keys = self.sessions[os.getpid()].execute("SELECT COUNT(*) from cache WHERE host_id='{}' and dirty=0;".format(host)).scalar()
        # Calculate stats for all cache rows
        total_keys = self.sessions[os.getpid()].execute("SELECT COUNT(*) from cache WHERE host_id='{}';".format(host)).scalar()
        write_total = self.sessions[os.getpid()].execute("SELECT SUM(num_writes) from cache WHERE host_id='{}';".format(host)).scalar()
        read_total = self.sessions[os.getpid()].execute("SELECT SUM(num_reads) from cache WHERE host_id='{}';".format(host)).scalar()
        write_time_total = self.sessions[os.getpid()].execute("SELECT SUM(write_time) from cache WHERE host_id='{}';".format(host)).scalar()
        read_time_total = self.sessions[os.getpid()].execute("SELECT SUM(read_time) from cache WHERE host_id='{}';".format(host)).scalar()
        write_oh_time_total = self.sessions[os.getpid()].execute("SELECT SUM(write_overhead_time) from cache WHERE host_id='{}';".format(host)).scalar()
        read_oh_time_total = self.sessions[os.getpid()].execute("SELECT SUM(read_overhead_time) from cache WHERE host_id='{}';".format(host)).scalar()
        # Calculate stats for all method rows
        total_methods = self.sessions[os.getpid()].execute("SELECT COUNT(*) from method WHERE host_id='{}';".format(host)).scalar()
        eval_time_total = self.sessions[os.getpid()].execute("SELECT SUM(eval_time) from method WHERE host_id='{}';".format(host)).scalar()
        expected_eval_time = self.sessions[os.getpid()].execute("SELECT SUM((eval_time / num_eval) * ref) from method WHERE host_id='{}';".format(host)).scalar()
        method_oh_time_total = self.sessions[os.getpid()].execute("SELECT SUM(overhead_time) from method WHERE host_id='{}';".format(host)).scalar()
        eval_total = self.sessions[os.getpid()].execute("SELECT SUM(num_eval) from method WHERE host_id='{}';".format(host)).scalar()
        # Calculate stats for individual rows
        elapsed_time = self.sessions[os.getpid()].execute("SELECT SUM(elapsed_time) from individuals;").scalar()
        eval_inds = self.sessions[os.getpid()].execute("SELECT COUNT(*) from individuals WHERE evaluation_status = 'EVALUATED';").scalar()
        
        reuse_total = self.get_total_cache_hits(host)
        
        # End Transaction
        self.sessions[os.getpid()].execute("ROLLBACK;")
        
        # Convert None variables to 0
        cache_size = cache_size if cache_size is not None else 0
        valid_keys = valid_keys if valid_keys is not None else 0
        total_keys = total_keys if total_keys is not None else 0
        write_total = write_total if write_total is not None else 0
        read_total = read_total if read_total is not None else 0
        write_time_total = write_time_total if write_time_total is not None else 0
        read_time_total = read_time_total if read_time_total is not None else 0
        write_oh_time_total = write_oh_time_total if write_oh_time_total is not None else 0
        read_oh_time_total = read_oh_time_total if read_oh_time_total is not None else 0
        total_methods = total_methods if total_methods is not None else 0
        eval_time_total = eval_time_total if eval_time_total is not None else 0
        expected_eval_time = expected_eval_time if expected_eval_time is not None else 0
        method_oh_time_total = method_oh_time_total if method_oh_time_total is not None else 0
        eval_total = eval_total if eval_total is not None else 0
        elapsed_time = elapsed_time if elapsed_time is not None else 0
        eval_inds = eval_inds if eval_inds is not None else 0

        # Calculate total time spent
        cache_time_total = eval_time_total + read_time_total + write_time_total + write_oh_time_total + read_oh_time_total + method_oh_time_total
        
        overhead_time = time.time() - start_time
        
        # Write to CSV
        if non_init:
            keys_added = total_keys - df.loc[len(df) - 1]['totalKeys']
            try:
                 df.loc[len(df)] = [cache_size, total_keys, valid_keys, keys_added, 
                                    reuse_total, eval_total, write_total, read_total, 
                                    write_time_total, read_time_total, write_oh_time_total, 
                                    read_oh_time_total, method_oh_time_total, cache_time_total, 
                                    expected_eval_time, total_methods, overhead_time, eval_inds, elapsed_time]
            except Exception as e:
                 print('Dataframe columns are', [col for col in df.columns])
                 raise e
            df.to_csv(filename, index=False)
        else:
            df.loc[len(df)] = [cache_size, total_keys, valid_keys, 0, 
                               reuse_total, eval_total, write_total, 
                               read_total, write_time_total, read_time_total, 
                               write_oh_time_total, read_oh_time_total, method_oh_time_total, 
                               cache_time_total, expected_eval_time, total_methods, overhead_time, 
                               eval_inds, elapsed_time]
            df.to_csv(filename, index=False)

    def optimize_cache(self, host, max_size):
        """
        Removes data entry from table
        Args:
            host:     id of host machine ('central' if centralized)
            max_size: maximum cache size
        """
        start_time = time.time()
        # Note: all non-removal queries read-only and do not lock rows
        # This means cache information may change during the cache optimization
        # Cache is optimized on the state of the change when query_full is read

        # Decides what cache optimization to use
        # TODO: Make this a XML parameter
        mode = "timsort"
        node_list = []
        filename = "cache_opt_info{}.txt".format(os.getpid())
            
        self.sessions[os.getpid()].execute("START TRANSACTION;")
        cache_size = self.sessions[os.getpid()].execute("SELECT SUM(size) from cache WHERE host_id='{}' and dirty=0;".format(host)).scalar()
        self.sessions[os.getpid()].execute("ROLLBACK;")
        
        # Convert None variables to 0
        cache_size = cache_size if cache_size is not None else 0
        
        total_value = None

        # If cache optimization is needed
        if cache_size >= max_size:
            # Query all valid cache rows
            query_full = self.sessions[os.getpid()].query(self.Cache).filter(self.Cache.host_id == host, self.Cache.dirty == 0)
            # Calculate max of hits_last_gen
            max_hits_last_gen = max([int(row.hits_last_gen) for row in query_full])
            # Iterate over valid cache entries and calculate their fitness
            for row in query_full:
                # Calculate maximum eval time of everything tied to cache row
                mcm = self.sessions[os.getpid()].query(self.MCM).filter(self.MCM.cache_id == row.id)
                methods = []
                for e in mcm:
                    m = self.sessions[os.getpid()].query(self.Method).filter(self.Method.host_id == host, self.Method.id == e.method_id).first()
                    if m is not None:
                        methods.append(m)
                # Length of methods should always be > 0
                if methods is None:
                    raise Exception("Found cached data with no methods tied to it")

                # Calculate estimate of time saved and sum it over every method associated with the cache entry
                my_sum = 0
                for method in methods:
                    my_sum += (method.eval_time / method.num_eval) * (method.ref - method.num_eval)
                    
                # Query row from db and lock it
                not_queried = True
                while (not_queried):
                    try:
                        specific_row = self.sessions[os.getpid()].query(self.Cache).with_for_update().filter(self.Cache.id == row.id, self.Cache.host_id == host).first()
                        not_queried = False
                    except:
                        print("Error when querying cache row for value func calc. (Most likely a lock timeout)")
                        time.sleep(10+random())

                # Increase gens since last hit if row had 0 hits last gen
                if specific_row.hits_last_gen == 0:
                    specific_row.gens_since_last_hit += 1
                else:
                    specific_row.gens_since_last_hit = 0

                # Value function calculation
                value = (float(my_sum) * float(row.num_reads)) * ((row.hits_last_gen / (max_hits_last_gen + 1)) + (1 / (row.gens_since_last_hit + 1)))
                
                # Reset hits_last_gen back to 0
                specific_row.hits_last_gen = 0
                # Release Lock
                self.sessions[os.getpid()].commit()
                
                # Adjust weights and values to ints if mode is knapsack
                value = int(np.ceil(value) * 1000000) if mode == "knapsack" else value
                weight = int(np.ceil(row.size)) if mode == "knapsack" else row.size
                
                # Create nodes from database row information
                node_list.append(Node(value, weight, row.id))
            
            # Remove keys/data from cache
            if mode == "timsort":
                total_value = self.sort_and_greed(node_list, max_size, host)
            elif mode == "heap":
                self.heap_invalidation(node_list, max_size, host, cache_size)
            elif mode == "knapsack":
                self.knapsack(node_list, max_size, host)
            else:
                raise ValueError("Invalid cache optimization mode")
                
        # Make sure all db changes are committed
        self.commit()
        
        overhead_time = time.time() - start_time
        
        # Write time out to file
        with open(filename, "a") as f:
            f.write(str(overhead_time))
            if total_value is not None:
                f.write(", {}".format(total_value))
            f.write("\n")
