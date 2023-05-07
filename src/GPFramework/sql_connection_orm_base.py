from sqlalchemy import create_engine, func, and_, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.mutable import Mutable
from sqlalchemy import Column, Integer, String, PickleType, Float, Enum, DateTime, BigInteger
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from sqlalchemy.schema import ForeignKey
from sqlalchemy.dialects.mysql import MEDIUMBLOB, LONGTEXT
from sqlalchemy.sql import text
from time import strftime
from datetime import datetime
from enum import Enum as enum_Enum
from abc import ABC
import os
import uuid

from sqlalchemy.sql.expression import null
from sqlalchemy.sql.schema import BLANK_SCHEMA

class MutableObject(Mutable):
    """Acts as a wrapper class that allows for propagation of attribute change
    events of creator.Individual objects to SQLAlchemy.
    Has only been tested with creator.Individual subclassing list.
    """

    def __init__(self, obj):
        self.__dict__['obj'] = obj

    @classmethod
    def coerce(cls, key, value):
        if not isinstance(value, MutableObject):
            return MutableObject(value)
        else:
            return value

    def __setattr__(self, name, value):
        try:
            setattr(self.__dict__['obj'], name, value)
            self.changed()
        except ValueError:
            super(Mutable, self).__setattr__(name, value)

    def __getattr__(self, name):
        try:
            return getattr(self.__dict__['obj'], name)
        except ValueError:
            return super(Mutable, self).__setattr__(name, value)

    def __getstate__(self):
        return self.__dict__['obj']

    def __setstate__(self, state):
        self.__dict__['obj'] = state

    def __getitem__(self, indices):
        return self.__dict__['obj'][indices]

class MyPickleType(PickleType):
    impl = MEDIUMBLOB

class IndividualStatus(enum_Enum):
    """
    Enum of possible states for individuals in database for tracking evaluation progress.
    """
    NOT_EVALUATED = 1
    IN_PROGRESS = 2
    WAITING_FOR_MASTER = 3
    EVALUATED = 4

def get_session(connection_str, reuse, fitness_names, dataset_names, statistics_dict, cache_dict, is_worker, includeADFS=False):
    """
    Creates SQLAlchemy connection to database and ORM mappings
    Args:
        connection_str: database URL for SQLAlchemy. See http://docs.sqlalchemy.org/en/latest/core/engines.html
        reuse: whether to reuse the database tables or drop them and start new ones
        fitness_names: name of objectives
        dataset_names: name of datasets
        statistics_dict: dictionary mapping statistic names to their type
        cache_dict: dictionary mapping cache parameters
    Returns:
        SQLAlchemy session, Individual/History/ParetoFront ORM tables, and IndividualStatus enum
    """

    # Create SQLAlchemy engine connection. Isolation level should be 'READ COMMITTED'.
    if connection_str == 'sqlite':
        engine = create_engine('sqlite:///EMADE_' + strftime('%m-%d-%Y_%H-%M-%S') + '.db', connect_args={'timeout': 500.0})
    else:
        if 'sqlite' in connection_str:
            engine = create_engine(connection_str, connect_args={'timeout': 500.0})
        elif 'mysql' in connection_str:
            engine = create_engine(connection_str, isolation_level='READ COMMITTED', connect_args={'connect_timeout': 500}, pool_recycle=60)
            engine.execute(text('SET innodb_lock_wait_timeout = ' + cache_dict['timeout']))

        else:
            engine = create_engine(connection_str)

    # Drop tables if reuse is false
    if not reuse:
        engine.execute(text('DROP TABLE IF EXISTS paretofront'))
        engine.execute(text('DROP TABLE IF EXISTS history'))
        engine.execute(text('DROP TABLE IF EXISTS mcm'))
        engine.execute(text('DROP TABLE IF EXISTS mim'))
        engine.execute(text('DROP TABLE IF EXISTS NNLearnerStatistics'))
        engine.execute(text('DROP TABLE IF EXISTS individuals'))
        engine.execute(text('DROP TABLE IF EXISTS `modules`'))
        engine.execute(text('DROP TABLE IF EXISTS cache'))
        engine.execute(text('DROP TABLE IF EXISTS method'))
        engine.execute(text('DROP TABLE IF EXISTS statistics'))
        engine.execute(text('DROP TABLE IF EXISTS host'))
    elif not is_worker:
        try:
            engine.execute(text("UPDATE individuals SET evaluation_status = 'NOT_EVALUATED' WHERE evaluation_status = 'IN_PROGRESS'"))

        except:
            print('No IN_PROGRESS individuals to revert to NOT_EVALUATED')

    Base = declarative_base()

    class Individual(Base):
        """
        Individual ORM mapper with individual specific values.
        """
        __tablename__ = 'individuals'
        hash = Column(String(64), primary_key=True, index=True)
        elapsed_time = Column(Float)
        retry_time = Column(Float)
        age = Column(Float, nullable=False)
        evaluation_status = Column(Enum(IndividualStatus))
        evaluation_gen = Column(Integer)
        evaluation_start_time = Column(DateTime)
        tree = Column(LONGTEXT, nullable=False)
        error_string = Column(LONGTEXT)
        pickle = Column(MutableObject.as_mutable(MyPickleType), nullable=False)

        def __repr__(self):
            return ', '.join(str(key) + ' ' + str(type(value)) + ' ' + str(value)for key, value in vars(self).items() if not key.startswith('__'))

    class Modules(Base):
        """
        Stores info on modules shared by individuals. 
        Once a module is trained as part of an individual, weights/hyperparameters are saved here and loaded the next time they're used.
        If a better individual uses the module, it overwrites the weights/hyperparameters.
        If a module is mated/mutated, its weights/hyperparameters are reset

        Weights correspond to a keras model that can be called like a layer as part of a larger model. Relevant Keras documentation snippet:

        "You can treat any model as if it were a layer by invoking it on an Input or on the output of another layer. 
        By calling a model you aren't just reusing the architecture of the model, you're also reusing its weights."
        """
        __tablename__ = "modules"
        mod_num = Column(Integer, primary_key=True, index=True)
        age = Column(Float, nullable=False)
        num_occur = Column(Integer)
        tree = Column(LONGTEXT, nullable=False)
        error_string = Column(LONGTEXT)
        weights = Column(MutableObject.as_mutable(MyPickleType))
        pickle = Column(MutableObject.as_mutable(MyPickleType), nullable=False)

        def __repr__(self):
            return ', '.join(str(key) + ' ' + str(type(value)) + ' ' + str(value)for key, value in vars(self).items() if not key.startswith('__'))

    class History(Base):
        """
        History ORM mapper with time information about all considered individuals
        """
        __tablename__ = 'history'
        id = Column(Integer, primary_key=True, index=True)
        optimization = Column(Integer, nullable=False)
        generation = Column(Integer, nullable=False)
        hash = Column(String(64), ForeignKey('individuals.hash'), nullable=False)

    class ParetoFront(Base):
        """
        ParetoFront ORM mapper with time information about ParetoFront individuals
        """
        __tablename__ = 'paretofront'
        id = Column(Integer, primary_key=True, index=True)
        optimization = Column(Integer, nullable=False)
        generation = Column(Integer, nullable=False)
        hash = Column(String(64), ForeignKey('individuals.hash'), nullable=False)

    class Host(Base):
        """
        Host ORM mapper for storing information about worker machines
        """
        __tablename__ = 'host'
        id = Column(String(512), primary_key=True, index=True)

    class Cache(Base):
        """
        Cache Data ORM mapper used to manage resources
        """
        __tablename__ = 'cache'
        id = Column(String(512), nullable=False, unique=True, primary_key=True, index=True) # Unique id, hash of stored data
        host_id = Column(String(32), ForeignKey('host.id'), nullable=False) # id of machine this method was evaluated on
        write_time = Column(Float) # Time spent in write IO
        read_time = Column(Float) # Time spent in read IO
        write_overhead_time = Column(Float) # Time spent in write method outside of IO
        read_overhead_time  = Column(Float) # Time spent in read method outside of IO
        size = Column(Float) # Size of stored data in kB (kilobytes)
        ref = Column(Integer) # Number of times this cache entry is read
        num_reads = Column(Integer) # Number of times this cache entry is read
        num_writes = Column(Integer) # Number of times this cache entry was written to disk
        hits_last_gen = Column(Integer) # Number of cache hits last gen
        gens_since_last_hit = Column(Integer) # Number of gens since last hit
        dirty = Column(Integer, nullable=False) # 0 if data is currently cached (stored in memory), 1 otherwise
        train_directory = Column(String(512)) # Training directory of cached data (identifies fold)
        test_directory = Column(String(512)) # Testing directory of cached data (identifies fold)

    class Method(Base):
        """
        Method ORM mapper for storing information about specific method calls
        """
        __tablename__ = 'method'
        id = Column(String(512), primary_key=True, index=True) # Unique id, hash of incoming data + method name + input arguments
        host_id = Column(String(32), ForeignKey('host.id'), nullable=False) # id of machine this method was evaluated on
        eval_time = Column(Float) # Code evaluation done regardless of whether cache is used
        overhead_time = Column(Float) # all other code evaluation (cache overhead)
        ref = Column(Integer) # Number of times the method id is referenced by an individual (Default 1)
        num_eval = Column(Integer) # Number of times the method id is referenced AND the method is evaluated (Default 1)
        error = Column(Integer, nullable=False) # 1 if an error occurred during method evaluation, 0 otherwise
        error_string = Column(LONGTEXT) # if an error occurred, the string error output is stored here

    class MCM(Base):
        """
        Method to Cache Map (MCM) ORM mapper with a mapping between cachedata and methods
        One cache entry can have multiple methods
        One method can only have one cache entry
        Many to One relationship
        """
        __tablename__ = 'mcm'
        method_id = Column(String(512), ForeignKey('method.id'), primary_key=True, index=True)
        cache_id = Column(String(512), ForeignKey('cache.id'), nullable=False)

    class MIM(Base):
        """
        Method to Individual Map (MIM) ORM mapper with a mapping between individuals and cache ids
        Many to Many Relationship
        """
        __tablename__ = 'mim'
        method_id = Column(String(512), ForeignKey('method.id'), primary_key=True)
        ind_hash = Column(String(64), ForeignKey('individuals.hash'), primary_key=True)

    class Statistics(Base):
        """
        Statistics ORM mapper with time information about population statistics
        """
        __tablename__ = 'statistics'
        id = Column(Integer, primary_key=True, index=True)
        generation = Column(Integer, nullable=False)
        layer_frequencies = Column(PickleType)
        layer_frequencies_str = Column(String(512))
    
    class NNStatistics(Base):
        """
        NNLearner specific statistics table. Keeps track of nnlearners by hash. 
        Accumulates past individual strings in parseable string.
        """
        __tablename__ = 'NNLearnerStatistics'
        hash = Column(String(64), primary_key=True, index=True)
        age = Column(Float, nullable=False, primary_key=True, index=True)
        parents = Column(LONGTEXT)
        current_tree = Column(LONGTEXT, nullable=False)
        error_string = Column(LONGTEXT)

    # Dynamically set names for objective columns in table
    for dataset in dataset_names:
        for fitness in fitness_names:
            setattr(Individual, dataset + ' ' + fitness, Column(Float))
            setattr(Modules, dataset + ' ' + fitness, Column(Float))

    # Dynamically set statistics columns
    for stat_name in statistics_dict:
        stat_type = statistics_dict[stat_name]['type']
        col_type = None
        if stat_type == 'int':
            col_type = Column(Integer)
        elif stat_type == 'long':
            col_type = Column(BigInteger)
        elif stat_type == 'float':
            col_type = Column(Float)
        elif stat_type == 'object':
            col_type = Column(MutableObject.as_mutable(PickleType), nullable=False)
        if col_type is not None:
            setattr(Statistics, stat_name, col_type)

    Base.metadata.create_all(engine)

    session_maker = sessionmaker(bind=engine)
    return session_maker, Individual, History, ParetoFront, IndividualStatus, Host, Cache, Method, MCM, MIM, Statistics, Modules, NNStatistics

class Borg(object):
    """
    More flexible Singleton acting as a data sink for objects
    """
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class ConnectionSetup(Borg):
    """
    This class sets up shared objects in memory for SQLConnections on the same process
    """
    def __init__(self):
        """
        Initializes the session dictionary for every connection created by this process
        """
        print("Performing connection setup for PID: {}".format(os.getpid()))
        
        super().__init__() # Joins the borg :)
        
        # Initializes session dictionary for all Borg subclasses
        self.sessions = {}

class SQLConnection(Borg, ABC):
    """
    This class is an interface to the database storing all the individuals.
    It creates one engine for every machine
    Every process is then mapped a Session object
    If connection_str is None then no new connection engine is made
    It is assumed the caller is either master_algorithm or worker_algorithm
    
    This class is an abstract base class and cannot be instantiated
    """

    def __init__(self, connection_str=None, reuse=None, fitness_names=None, dataset_names=None, statistics_dict=None, cache_dict=None, is_worker=False, is_cache=False, ind_hash=None, includeADFS=True):
        """
        Initializes the SQLConnection object. Assigns return values from get_session as instance variables
        """
        if not isinstance(connection_str, str):
            raise Exception("connection_str must be a string")
        
        super().__init__() # Joins the borg :)
        
        # self.session_maker, self.Individual, self.History, self.ParetoFront, self.IndividualStatus, self.Host, self.Cache, self.Method, self.MCM, self.MIM, self.Statistics, self.NNStatistics = get_session(connection_str, reuse, fitness_names,
        #                                                                                                                                                                                   dataset_names, statistics_dict,
        #                                                                                                                                                                                   cache_dict, is_worker)
        self.session_maker, self.Individual, self.History, self.ParetoFront, self.IndividualStatus, self.Host, self.Cache, self.Method, self.MCM, self.MIM, self.Statistics, self.Modules, self.NNStatistics = get_session(connection_str, reuse, fitness_names,
                                                                                                                                                                                          dataset_names, statistics_dict,
                                                                                                                                                                                          cache_dict, is_worker)                                                                                                                                                                                          

        # Create a unique db session for the current process if one does not already exist
        # If a session already exists use the existing one
        if os.getpid() not in self.sessions:
            self.sessions[os.getpid()] = self.session_maker()
            self.sessions[os.getpid()].uuid = uuid.uuid4()
            print("Open Database Connection", "UUID:", self.sessions[os.getpid()].uuid, "PID:", os.getpid(), "Ind Hash:", ind_hash, "TimeStamp:", datetime.now())
        
        if fitness_names is not None: self.fitness_names = fitness_names
        if dataset_names is not None: self.dataset_names = dataset_names
        if statistics_dict is not None: self.statistics = statistics_dict
        
        if not is_worker:
            # Update optimization for each new run with last optimization number + 1
            self.optimization = self.sessions[os.getpid()].query(func.max(self.History.optimization)).first()[0]
            self.optimization = 0 if self.optimization is None else self.optimization + 1

    def commit(self):
        """
        Commits the current session
        """
        self.sessions[os.getpid()].commit()

    def close(self):
        """
        Closes the current session
        Then removes it from the session dictionary
        Assumes process no longer needs the session
        """
        # Save uuid before it is removed
        uuid_ = self.sessions[os.getpid()].uuid
        
        self.sessions[os.getpid()].close()
        self.sessions[os.getpid()].bind.dispose()
        self.sessions.pop(os.getpid())
        
        print("Closing Database Connection", "UUID:", uuid_, "PID:", os.getpid(), "TimeStamp:", datetime.now())
        
    def commit_pid(self, pid):
        """
        Commits the current session
        
        Args:
            pid: process id of the session
        """
        self.sessions[pid].commit()

    def close_pid(self, pid):
        """
        Closes the current session
        Then removes it from the session dictionary
        Assumes process no longer needs the session
        
        Args:
            pid: process id of the session
        """
        # Save uuid before it is removed
        uuid_ = self.sessions[pid].uuid
        
        self.sessions[pid].close()
        self.sessions[pid].bind.dispose()
        self.sessions.pop(pid)
        
        print("Closing Database Connection", "UUID:", uuid_, "PID:", pid)

    def insert_statistics(self, gen, stats_dict):
        """
        Inserts statistics for a generation into the database
        Args:
            gen: current generation
            stats_dict: dictionary from statistic name to their values
        """
        stats = self.Statistics(generation=gen)
        for stat_name in stats_dict:
            setattr(stats, stat_name, stats_dict[stat_name])
        self.sessions[os.getpid()].add(stats)
        self.sessions[os.getpid()].commit()

    def select_statistics(self):
        """Retrieves dictionary of statistics to their values over time"""
        stats = {}
        for stat in self.sessions[os.getpid()].query(self.Statistics).all():
            for stat_name in self.statistics:
                if stat_name not in stats:
                    stats[stat_name] = []
                stats[stat_name].append(getattr(stat, stat_name))
        return stats
    
    def get_total_cache_hits(self, host):
        """Returns total cache hit count"""
        value = self.sessions[os.getpid()].execute("SELECT SUM(num_reads) from cache WHERE host_id='{}';".format(host)).scalar()
        value = value if value is not None else 0
        return value
