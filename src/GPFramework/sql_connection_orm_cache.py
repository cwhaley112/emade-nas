from GPFramework.sql_connection_orm_base import SQLConnection
from datetime import datetime
import socket
import sys
import os

class SQLConnectionCache(SQLConnection):
    """
    This class is a Singleton interface to the database storing all the individuals.
    """

    def __init__(self, connection_str=None, reuse=None, fitness_names=None, dataset_names=None, statistics_dict=None, cache_dict=None, is_worker=True, ind_hash=None):
        """
        Initializes the SQLConnection object. Assigns return values from get_session as instance variables
        """
        super().__init__(connection_str, reuse, fitness_names, dataset_names, statistics_dict, cache_dict, is_worker=is_worker, ind_hash=ind_hash)

    def check_cache_size(self, host):
        """
        Returns the current cache size of cache table

        Args:
            host: id of host machine ('central' if centralized)
        """
        print("Before Cache Size Query | " + str(datetime.now()))
        sys.stdout.flush()
        
        stored_keys = self.sessions[os.getpid()].query(self.Cache).filter(self.Cache.host_id == host, self.Cache.dirty == 0)
        
        print("After Cache Size Query | " + str(datetime.now()))
        sys.stdout.flush()
        return sum([row.size for row in stored_keys])

    def query_method(self, method_id, ind_hash, host):
        """
        Checks for method in database and adds method to database if not found
        Checks cache for whether this method is currently saved

        Args:
            method_id: method string used to uniquely identify method (name + args + data)
            ind_hash:  hash of individual stored in datapair
            host:      id of host machine ('central' if centralized)

        Returns:
            True if method is in database, else False
        """
        try:
            print("Before Method Query | Ind Hash: {} | Method id: {} | ".format(ind_hash, method_id) + str(datetime.now()))
            sys.stdout.flush()

            query = self.sessions[os.getpid()].query(self.Method).with_for_update().filter(self.Method.id == method_id, self.Method.host_id == host).first()

            print("After Method Query | Ind Hash: {} | Method id: {} | ".format(ind_hash, method_id) + str(datetime.now()))
            sys.stdout.flush()
        except Exception as e:
            print("Got exception:", e, "While querying Method id:", method_id, "with host:", host)
            # raise special exception for lock timeout
            raise Exception("lock timeout")
        
        if query is None:
            self.sessions[os.getpid()].add(self.Method(id=method_id, host_id=host,
                                                       eval_time=0, overhead_time=0, 
                                                       ref=1, num_eval=0,
                                                       error=0, error_string=None))
            try:
                print("Before Method Query | Ind Hash: {} | Method id: {} | ".format(ind_hash, method_id) + str(datetime.now()))
                sys.stdout.flush()

                query = self.sessions[os.getpid()].query(self.Method).with_for_update().filter(self.Method.id == method_id, self.Method.host_id == host).first()

                print("After Method Query | Ind Hash: {} | Method id: {} | ".format(ind_hash, method_id) + str(datetime.now()))
                sys.stdout.flush()
            except Exception as e:
                print("Got exception:", e, "While querying Method id:", method_id, "with host:", host)
                # raise special exception for lock timeout
                raise Exception("lock timeout")

            '''
            Query MIM
            '''
            try:
                check = self.sessions[os.getpid()].query(self.MIM).filter(self.MIM.method_id == method_id, self.MIM.ind_hash == ind_hash).first()
            except Exception as e:
                print("Got exception:", e, "While MIM querying Method id:", method_id, "and Individual hash:", ind_hash, "with host:", host)
                # raise special exception for lock timeout
                raise Exception("lock timeout")
            if not check:
                self.sessions[os.getpid()].add(self.MIM(method_id=method_id, ind_hash=ind_hash))
            else:
                # Note: this error should never occur on tiered datasets because the data hashes in the method ids will change
                raise Exception("An Individual hash with the same Method ID was evaluated twice.")

            return False, query, None
        else:
            if query.error == 1:
                query.ref += 1
                self.sessions[os.getpid()].commit()
                raise Exception('key known to cause error: ' + query.error_string)
            else:
                mq = self.sessions[os.getpid()].query(self.MCM).filter(self.MCM.method_id == method_id).first()
                if mq is not None:
                    try:
                        print("Before Cache Query | Ind Hash: {} | Method id: {} | ".format(ind_hash, method_id) + str(datetime.now()))
                        sys.stdout.flush()

                        cq = self.sessions[os.getpid()].query(self.Cache).with_for_update().filter(self.Cache.id == mq.cache_id, self.Cache.host_id == host).first()

                        print("After Cache Query | Ind Hash: {} | Method id: {} | ".format(ind_hash, method_id) + str(datetime.now()))
                        sys.stdout.flush()
                    except Exception as e:
                        print("Got exception:", e, "While querying Cache id:", mq.cache_id, "with host:", host)
                        # raise special exception for lock timeout
                        raise Exception("lock timeout")
                    
                    if cq is not None:
                        query.ref += 1
                        if cq.dirty == 1 or cq.dirty == 2:
                            return False, query, cq
                        else:
                            return True, query, cq

            query.ref += 1
            return False, query, None

    def query_write(self, host, method_id, data_hash, train_directory, test_directory):
        """
        Checks if data has already been stored into the cache

        Args:
            host:            id of host machine ('central' if centralized)
            method_id:       method string used to uniquely identify method (name + args + data)
            data_hash:       hash of data to save
            train_directory: base directory of cached data (used in optimization)
            test_directory:  base directory of cached data (used in optimization)

        Returns:
            True if key is in database, else False
        """
        try:
            print("Before Cache Query | Cache id: {} | Method id: {} | ".format(data_hash, method_id) + str(datetime.now()))
            sys.stdout.flush()

            query = self.sessions[os.getpid()].query(self.Cache).with_for_update().filter(self.Cache.id == data_hash, self.Cache.host_id == host).first()

            print("After Cache Query | Cache id: {} | Method id: {} | ".format(data_hash, method_id) + str(datetime.now()))
            sys.stdout.flush()
        except Exception as e:
            print("Got exception:", e, "While querying Cache id:", data_hash, "with host:", host)
            # raise special exception for lock timeout
            raise Exception("lock timeout")
        
        if query is None:
            '''
            Query MCM
            '''
            try:
                print("Before MCM Query | Cache id: {} | Method id: {} | ".format(data_hash, method_id) + str(datetime.now()))
                sys.stdout.flush()
                
                mcm = self.sessions[os.getpid()].query(self.MCM).filter(self.MCM.method_id == method_id).first()
                
                print("After MCM Query | Cache id: {} | Method id: {} | ".format(data_hash, method_id) + str(datetime.now()))
                sys.stdout.flush()
            except Exception as e:
                print("Got exception:", e, "While MCM querying Method id:", method_id, "and Cache id:", data_hash, "with host:", host)
                # raise special exception for lock timeout
                raise Exception("lock timeout")
            
            self.sessions[os.getpid()].add(self.Cache(id=data_hash, host_id=host,
                                                      write_time=0, read_time=0, 
                                                      write_overhead_time=0, read_overhead_time=0,
                                                      size=0,
                                                      ref=1, num_reads=0, num_writes=0,
                                                      hits_last_gen=0, gens_since_last_hit=0,
                                                      dirty=1,
                                                      train_directory=train_directory,
                                                      test_directory=test_directory))
            
            try:
                print("Before Cache Query 2ND | Cache id: {} | Method id: {} | ".format(data_hash, method_id) + str(datetime.now()))
                sys.stdout.flush()

                query = self.sessions[os.getpid()].query(self.Cache).with_for_update().filter(self.Cache.id == data_hash, self.Cache.host_id == host).first()

                print("After Cache Query 2ND | Cache id: {} | Method id: {} | ".format(data_hash, method_id) + str(datetime.now()))
                sys.stdout.flush()
            except Exception as e:
                print("Got exception:", e, "While querying Cache id after add:", data_hash, "with host:", host)
                raise
            
            if mcm is None:
                self.sessions[os.getpid()].add(self.MCM(method_id=method_id, cache_id=data_hash))
            else:
                raise Exception("Method ID produced at least two different output data hashes.")

            return True, query
        else:
            '''
            Query MCM
            '''
            try:
                print("Before MCM Query Row Exists | Cache id: {} | Method id: {} | ".format(data_hash, method_id) + str(datetime.now()))
                sys.stdout.flush()
                
                mcm = self.sessions[os.getpid()].query(self.MCM).filter(self.MCM.method_id == method_id).first()
                
                print("After MCM Query Row Exists | Cache id: {} | Method id: {} | ".format(data_hash, method_id) + str(datetime.now()))
                sys.stdout.flush()
            except Exception as e:
                print("Got exception:", e, "While MCM Querying Row Exists | Method id:", method_id, "and Cache id:", data_hash, "with host:", host)
                # raise special exception for lock timeout
                raise Exception("lock timeout")
            
            if mcm is None:
                self.sessions[os.getpid()].add(self.MCM(method_id=method_id, cache_id=data_hash))
            else:
                raise Exception("Method ID produced at least two different output data hashes.")
            
            query.ref += 1
            if query.dirty == 1 or query.dirty == 2:
                return True, query
            return False, query
        
    def set_dirty(self, row):
        """
        Sets dirty bit of cache row to 1

        Args:
            row: reference to cache row

        """
        # Set dirty bit
        row.dirty = 1
            
    def set_dirty_rollback(self, cache_id, pid, host):
        """
        Rollsback transaction
        Queries for a new lock on the cache row
        Sets dirty bit of cache row to 1

        Args:
            cache_id: id of cache row
            pid:      process id
            host:     cache host

        """
        # Rollback session to avoid autoflush and start a new transaction
        self.sessions[pid].rollback()
        
        try:
            print("Before Cache Query to Set Dirty 1 | Cache id: {} | Process id: {} | ".format(cache_id, pid) + str(datetime.now()))
            sys.stdout.flush()

            query = self.sessions[pid].query(self.Cache).with_for_update().filter(self.Cache.id == cache_id, self.Cache.host_id == host).first()

            print("After Cache Query to Set Dirty 1 | Cache id: {} | Process id: {} | ".format(cache_id, pid) + str(datetime.now()))
            sys.stdout.flush()
        except Exception as e:
            print("Got exception:", e, "While querying Cache id:", cache_id, "with host:", host)
            # raise special exception for lock timeout
            raise Exception("lock timeout")
        
        if query is not None:
            # Set dirty bit
            query.dirty = 1

    def update_time_size(self, row, write_time, size):
        """
        Updates write_time, num_writes, size, and dirty bit in cache row

        Args:
            row:         link to db row
            write_time:  time taken to write the data
            size:        size of the generated data

        """
        row.write_time += write_time
        row.num_writes += 1
        row.size = size
        row.dirty = 0
        
    def update_time_size_dirty(self, row, write_time):
        """
        Updates write_time, num_writes, and dirty bit in cache row

        Args:
            row:         link to db row
            write_time:  time taken to write the data

        """
        row.write_time += write_time
        row.num_writes += 1
        row.dirty = 1

    def update_ref_and_read(self, row, read_time):
        """
        Updates 'ref' and 'read-time' columns in cache row

        Args:
            row:       link to db row
            read_time: time taken to read data from cache

        """
        row.read_time += read_time
        row.ref += 1

    def set_error(self, row, error, time):
        """
        Sets error string and error bit in row

        Args:
            row:   link to db row
            error: string of error
            time:  time spent up until this method call

        """
        row.eval_time += time
        row.error = 1
        row.error_string = error

    def update_eval_time(self, row, time):
        """
        Updates eval time in method row

        Args:
            row:  link to db row
            time: time spent in seconds

        Returns:
            Running sum of eval_time
        """
        row.eval_time += time
        return row.eval_time
        
    def update_overhead_time(self, row, time):
        """
        Updates overhead time in method row

        Args:
            row:  link to db row
            time: time spent in seconds

        """
        row.overhead_time += time
        
    def update_read_overhead_time(self, row, time):
        """
        Updates read overhead time in cache row

        Args:
            row:  link to db row
            time: time spent in seconds

        """
        row.read_overhead_time += time
        
    def update_write_overhead_time(self, row, time):
        """
        Updates write overhead time in cache row

        Args:
            row:  link to db row
            time: time spent in seconds

        """
        row.write_overhead_time += time
        
    def increment_num_eval(self, row):
        """
        Increases row's num_eval by 1

        Args:
            row: link to db row

        """
        row.num_eval += 1
        
    def increment_num_reads(self, row):
        """
        Increases cache row's num_reads and hits_last_gen by 1

        Args:
            row: link to db row

        """
        row.num_reads += 1
        row.hits_last_gen += 1
        

    def set_special_error(self, row, string):
        """
        Used for placing specific error strings on a method row

        Args:
            row:    link to db row
            string: string to set error_string to

        """
        row.error_string = string
        self.commit()
        self.close()
