'''Programmed by Austin Dunn'''
from datetime import datetime
from shutil import rmtree
import numpy as np
import copy as cp
import hashlib
import pickle
import time
import sys
import os

CLEAN_FILE = "cleanup"

"""
Primitive Methods
"""

def check_cache_read(data_pair, database, method_key, oh_time_start, target=False):
    # Get fold/base directory of the data
    base_directory = data_pair.get_base_test_directory()

    # Check if method key exists and if it exists check if a cache entry for it exists
    cache_hit, method_row, cache_row = database.query_method(method_key,
                                                             data_pair.get_ind_hash(),
                                                             'central')
    
    # if the method key is currently cached
    if cache_hit:
        # Load the hash of the cached data into the data_pair
        data_hash = data_pair.set_hash(base_directory + cache_row.id)
        # Increment number of relevant reads
        database.increment_num_reads(cache_row)
        # End overhead time and update it on the method db row
        time_spent = time.time() - oh_time_start
        database.update_overhead_time(method_row, time_spent)
        # Read the cached data from the cache into the data_pair
        data_pair.load_data(data_hash, database, cache_row, target)
        print("Cache Hit after Read Before Commit | Method: {} | Cache: {} | TimeStamp | ".format(method_key, cache_row.id) + str(datetime.now()))
        # Save cache id to local memory
        cache_id = cp.deepcopy(cache_row.id)
        # Commit db changes and close the TCP connection with the db
        database.commit()
        try:
            database.close()
        except Exception as e:
            print("Error occurred while closing DB Session. TimeStamp:", datetime.now(), "Exception:", e)
            sys.stdout.flush()
        del database
        print("Cache Hit After Commit | Method: {} | Cache: {} | TimeStamp | ".format(method_key, cache_id) + str(datetime.now()))
        # The method key is currently cached. Return early.
        return time_spent, method_row, cache_row, True

    # Stop recording overhead time until cache store
    time_spent = time.time() - oh_time_start
    # Raise number of evaluations on method row
    database.increment_num_eval(method_row)
    # The method key is not currently cached. Evaluate it.
    return time_spent, method_row, cache_row, False

def check_cache_write(data_pair, database, method_row, cache_row, method_key, curr_oh_time, eval_time, target=False):
    # Start measuring overhead time
    oh_time_start = time.time()

    # Set hash in the datapair to match new data
    # This occurs before any additional database queries are made
    data_pair.update_hash()

    try:
        # Grab running eval_time sum
        eval_time_sum = database.update_eval_time(method_row, eval_time)

        # Return early if caching is off with only the method row updated
        if not data_pair.get_caching_mode():
            curr_oh_time += time.time() - oh_time_start
            database.update_overhead_time(method_row, curr_oh_time)
            database.commit()
            try:
                database.close()
            except Exception as e:
                print("Error occurred while closing DB Session. TimeStamp:", datetime.now(), "Exception:", e)
                sys.stdout.flush()
            return

        # Check if there is still room in the cache
        if database.check_cache_size('central') <= data_pair.get_cache_limit():
            # Boolean expression for whether eval_time_sum should be compared
            dirty_2 = cache_row is not None and cache_row.dirty == 2

            # Set overhead time in method row
            curr_oh_time += time.time() - oh_time_start
            database.update_overhead_time(method_row, curr_oh_time)

            # Check if primitive eval time is greater than minimum threshold to cache
            if eval_time >= data_pair.get_threshold():
                print("Eval Time Passed Threshold | Method: {} | TimeStamp | ".format(method_key) + str(datetime.now()))
                
                # Write Data into Cache
                cache_row = data_pair.save_data(data_pair.get_train_data(), data_pair.get_test_data(),
                                                                method_key, eval_time,
                                                                database, method_row, target)

            # Check if primitive sum total eval time is greater than minimum threshold to cache
            elif eval_time_sum >= data_pair.get_threshold() and not dirty_2:
                print("Eval Time Sum Passed Threshold | Method: {} | TimeStamp | ".format(method_key) + str(datetime.now()))

                # Write Data into Cache
                cache_row = data_pair.save_data(data_pair.get_train_data(), data_pair.get_test_data(),
                                                                method_key, eval_time,
                                                                database, method_row, target)

                # Read Data from Cache
                database.increment_num_reads(cache_row)
                data_pair.load_data(data_pair.get_hash(), database, cache_row)

                # If time required to read the data in is greater than the eval_time
                # Wipe the data and set dirty bit to special 'ignore' status
                if ((cache_row.read_time + cache_row.read_overhead_time) / cache_row.num_reads) > (eval_time_sum / method_row.num_eval):
                    # Remove data from cache
                    cache_row.dirty = 2
                    rmtree("{}/{}".format(data_pair.get_train_data().get_base_directory(), data_pair.get_hash()))
                    rmtree("{}/{}".format(data_pair.get_test_data().get_base_directory(), data_pair.get_hash()))
            # Primitive should not be cached
            else:
                print("Evaluated But Not Cached | Method: {} | TimeStamp | ".format(method_key) + str(datetime.now()))
        else:
            print("Evaluated But Not Cached | Method: {} | TimeStamp | ".format(method_key) + str(datetime.now()))

        # Commit changes to the database
        database.commit()
    except Exception as e:
        print("An Exception occurred during check_cache_write after the EmadeDataPair's hash was updated. method_key: {} | output data hash: {} Exception: {} TimeStamp: {}".format(method_key, data_pair.get_hash(), e, datetime.now()))

        if cache_row is not None:
            # define directories and id
            train_dir = "{}/{}".format(data_pair.get_train_data().get_base_directory(), data_pair.get_hash())
            test_dir = "{}/{}".format(data_pair.get_test_data().get_base_directory(), data_pair.get_hash())

            # Attempt to remove data
            try:
                rmtree(train_dir)
            except Exception as f:
                print("Failed to remove train data because it was already empty. Exception:", f)
            try:
                rmtree(test_dir)
            except Exception as f:
                print("Failed to remove test data because it was already empty. Exception:", f)
    
            # Make sure cache row shows no data being cached
            database.set_dirty(cache_row)
   
        # Remove cleanup file now that db session is cleaned up
        try:
            os.remove("{}/{}.txt".format(CLEAN_FILE, data_pair.get_ind_hash()))
        except Exception as f:
            # Note: This exception is expected when an error occurs before any data was written
            print("Failed to remove cleanup file after commit failed. Exception:", f)
            
        
        # Commit current changes to the database
        try:
            database.commit()
        except Exception as f:
            print("Error occurred while committing DB Session. check_cache_write Error Handler. TimeStamp:", datetime.now(), "Exception:", f)
            sys.stdout.flush()

        # Close database session
        try:
            database.close()
        except Exception as f:
            print("Error occurred while closing DB Session. check_cache_write Error Handler. TimeStamp:", datetime.now(), "Exception:", f)
            sys.stdout.flush()

        raise Exception("pr0c33d " + str(e))

    # Close database session
    try:
        database.close()
    except Exception as e:
        print("Error occurred while closing DB Session. check_cache_write success. TimeStamp:", datetime.now(), "Exception:", e)
        sys.stdout.flush()

    # Remove cleanup file now that db session is cleaned up
    try:
        os.remove("{}/{}.txt".format(CLEAN_FILE, data_pair.get_ind_hash()))
    except Exception as e:
        print("Failed to remove cleanup file. Exception:", e)
        
"""
IO Methods
"""
        
def load_cache(train_data, test_data, database, row, target):
    """
    Helper method for loading data stored locally

    Args:
        train_data: EmadeData containing training data
        test_data:  EmadeData containing testing data
        database:   database connection object (session)
        row:        reference to row in database
        target:     flag for whether predicted labels need to be loaded

    Returns:
        Updated EmadeData objects

    """

    train_gen_directory = train_data.get_gen_directory()
    train_fold = train_data.get_fold()

    test_gen_directory = test_data.get_gen_directory()
    test_fold = test_data.get_fold()

    start_time = time.time()

    # load data from files
    f = open('{}/feature_data_{}.npz'.format(train_gen_directory, train_fold), "rb")
    train_data_feature = np.load(f, allow_pickle=True)['arr']
    f.close()

    f = open('{}/feature_data_{}.npz'.format(test_gen_directory, test_fold), "rb")
    test_data_feature = np.load(f, allow_pickle=True)['arr']
    f.close()

    f = open('{}/stream_data_{}.npz'.format(train_gen_directory, train_fold), "rb")
    train_data_stream = np.load(f, allow_pickle=True)['arr']
    f.close()

    f = open('{}/stream_data_{}.npz'.format(test_gen_directory, test_fold), "rb")
    test_data_stream = np.load(f, allow_pickle=True)['arr']
    f.close()

    if target:
        with open(test_gen_directory + "/" + "labels.npz", "rb") as f:
            test_target = np.load(f, allow_pickle=True)['arr']
        with open(train_gen_directory + "/" + "labels.npz", "rb") as f:
            train_target = np.load(f, allow_pickle=True)['arr']

    read_time = time.time() - start_time
 
    start_time = time.time()

    # Release lock and update row
    database.update_ref_and_read(row, read_time)

    # set cached data into train data instances
    instances_1 = cp.deepcopy(train_data.get_instances())

    train_instances = []
    for f_data, s_data, instance in zip(train_data_feature, train_data_stream, instances_1):
        if len(f_data.shape) == 1 or len(f_data.shape) == 0:
            f_data = np.expand_dims(f_data, axis=0)
        if len(s_data.shape) == 1 or len(s_data.shape) == 0:
            s_data = np.expand_dims(s_data, axis=0)
        instance.get_features().set_data(f_data)
        instance.get_stream().set_data(s_data)
        train_instances.append(instance)

    instances_2 = cp.deepcopy(test_data.get_instances())

    # set cached data into test data instances
    test_instances = []
    for f_data, s_data, instance in zip(test_data_feature, test_data_stream, instances_2):
        if len(f_data.shape) == 1 or len(f_data.shape) == 0:
            f_data = np.expand_dims(f_data, axis=0)
        if len(s_data.shape) == 1 or len(s_data.shape) == 0:
            s_data = np.expand_dims(s_data, axis=0)
        instance.get_features().set_data(f_data)
        instance.get_stream().set_data(s_data)
        test_instances.append(instance)

    if target:
        for t, instance in zip(test_target, test_instances):
            if len(t.shape) == 1 or len(t.shape) == 0:
                t = np.expand_dims(t, axis=0)
            instance.set_target(t)

        for t, instance in zip(train_target, train_instances):
            if len(t.shape) == 1 or len(t.shape) == 0:
                t = np.expand_dims(t, axis=0)
            instance.set_target(t)

    train_data.set_instances(train_instances)
    test_data.set_instances(test_instances)

    overhead_time = time.time() - start_time
    database.update_read_overhead_time(row, overhead_time)

    return train_data, test_data

def save_cache(train_data, test_data, compress, method_id, ind_hash, database, row, target=False):
    """
    Helper method for saving data locally from array

    Args:
        train_data:   given EmadeData object
        test_data:    given EmadeData object
        compress:     whether to compress the data or not
        method_id:    method name + args + previous data
        ind_hash:     unique hash of an individual
        database:     database connection
        row:          link to db row
        target:       True if prediction of test data was updated False otherwise

    Returns:
        Updated EmadeData object
    """
    start_time = time.time()
 
    train_base_directory = train_data.get_base_directory()
    train_fold = train_data.get_fold()

    test_base_directory = test_data.get_base_directory()
    test_fold = test_data.get_fold()

    train_data_feature = []
    train_data_stream = []
    train_data_target = []
    for instance in train_data.get_instances():
        train_data_feature.append(np.array(instance.get_features().get_data()))
        train_data_stream.append(np.array(instance.get_stream().get_data()))
        train_data_target.append(np.array(instance.get_target()))

    test_data_feature = []
    test_data_stream = []
    test_data_target = []
    for instance in test_data.get_instances():
        test_data_feature.append(np.array(instance.get_features().get_data()))
        test_data_stream.append(np.array(instance.get_stream().get_data()))
        test_data_target.append(np.array(instance.get_target()))

    # hash feature and stream data separately then hash the addition of those hashes
    new_data_hash = functional_hash([test_data_feature, test_data_stream, test_data_target])

    new_train_gen_directory = train_base_directory + new_data_hash
    new_test_gen_directory = test_base_directory + new_data_hash

    # check cache
    write, c_row = database.query_write('central', method_id, new_data_hash,
                                        train_base_directory, test_base_directory)
 
    write_overhead_time = time.time() - start_time

    if write:
        start_time = time.time()
        size = 0

        with open("{}/{}.txt".format(CLEAN_FILE, ind_hash), "w") as f:
            f.write(new_train_gen_directory + ", " + new_test_gen_directory)

        try:
            os.makedirs(new_test_gen_directory)
        except FileExistsError:
            # remove directory (assuming whatever is stored there is not fully written due to process death)
            rmtree(new_test_gen_directory)
            # recreate the directory
            os.makedirs(new_test_gen_directory)
        
        try:
            os.makedirs(new_train_gen_directory)
        except FileExistsError:
            # remove directory (assuming whatever is stored there is not fully written due to process death)
            rmtree(new_train_gen_directory)
            # recreate the directory
            os.makedirs(new_train_gen_directory)

        # make some placeholder variables for cleaner code
        files = [new_train_gen_directory + "/feature_data_" + train_fold + ".npz",
                    new_train_gen_directory + "/stream_data_" + train_fold + ".npz",
                    new_test_gen_directory + "/feature_data_" + test_fold + ".npz",
                    new_test_gen_directory + "/stream_data_" + test_fold + ".npz"]

        datas = [train_data_feature, train_data_stream,
                    test_data_feature, test_data_stream]

        try:
            # save data onto the cache
            for file_name, data in zip(files, datas):
                f = open(file_name, "wb")
                if compress:
                    np.savez_compressed(file=f, arr=data)
                else:
                    np.savez(file=f, arr=data)
                f.close()
                info = os.stat(file_name)
                size += info.st_size / 1000.0

            # save the data hash
            with open(new_test_gen_directory + "/hash.txt", "wb") as f:
                pickle.dump(new_data_hash, f)

            if target:
                # save the label data list if it updated
                with open(new_test_gen_directory + "/" + "labels.npz", "wb") as f:
                    if compress:
                        np.savez_compressed(file=f, arr=test_data_target)
                    else:
                        np.savez(file=f, arr=test_data_target)
                    info = os.stat(new_test_gen_directory + "/" + "labels.npz")
                    size += info.st_size / 1000.0

                with open(new_train_gen_directory + "/" + "labels.npz", "wb") as f:
                    if compress:
                        np.savez_compressed(file=f, arr=train_data_target)
                    else:
                        np.savez(file=f, arr=train_data_target)
                    info = os.stat(new_train_gen_directory + "/" + "labels.npz")
                    size += info.st_size / 1000.0

            write_time = time.time() - start_time

            database.update_write_overhead_time(c_row, write_overhead_time)
            database.update_time_size(c_row, write_time, size)
        except Exception as e:
            print("Error occurred while writing data. Error:", e)
            # Remove data if error occurred and keep cache row as dirty 1
            try:
                rmtree(new_train_gen_directory)
            except Exception as g:
                print("Failed to remove train data because it was already empty. Exception:", g)
            try:
                rmtree(new_test_gen_directory)
            except Exception as g:
                print("Failed to remove test data because it was already empty. Exception:", g)         

            write_time = time.time() - start_time
            database.update_write_overhead_time(c_row, write_overhead_time)
            database.update_time_size_dirty(c_row, write_time)

    else:
        database.update_write_overhead_time(c_row, write_overhead_time)

    return train_data, test_data, new_data_hash, c_row

def load_hash(directory):
    """
    Helper method for loading hash locally

    Args:
        directory: directory to load data from

    """
    return pickle.load(open(directory + "/hash.txt", "rb"))

def cleanup_trash(ind_hash):
    """Removes corrupted cache data

    Args:
        ind_hash: name of text file to parse

    """
    cache_id = None
    if os.path.isfile("{}/{}.txt".format(CLEAN_FILE, ind_hash)):
        print("Individual trash being cleaned. Hash: " + ind_hash)
        print("TimeStamp | " + str(datetime.now()))
        sys.stdout.flush()
        with open("{}/{}.txt".format(CLEAN_FILE, ind_hash)) as f:
            # Parse file
            line = f.readline()
            parts = line.split(", ")

            # Parse cache_id
            index = parts[0].rfind('/')
            cache_id = parts[0][index+1:]
            
            # Delete cache data
            try:
                rmtree(parts[0])
            except Exception as e:
                print("Failed to remove train directory after process kill")
                print("Exception: {}".format(e))
            try:
                rmtree(parts[1])
            except Exception as e:
                print("Failed to remove test directory after process kill")
                print("Exception: {}".format(e))
        # Delete cleanup file
        os.remove("{}/{}.txt".format(CLEAN_FILE, ind_hash))
        
    return cache_id

"""
Hashing Methods
"""

def hash_array(array_to_hash):
    """Return the SHA256 hash of a numpy array
    Copied from the hash function in EMADE.py

    Returns:
        the SHA256 hash of numpy array
    """
    return hashlib.sha256(array_to_hash.tostring()).hexdigest()

def hash_string(my_string):
    """Return the SHA256 hash of a string
    Copied from the hash function in EMADE.py

    Returns:
        the SHA256 hash of string
    """
    return hashlib.sha256(my_string.encode()).hexdigest()

def functional_hash(my_list):
    """Return the SHA256 hash of a fold (feature_data + stream_data + target)

    Args:
        my_list: list of list of numpy arrays

    Returns:
        the SHA256 hash of string
    """
    for i in my_list:
        for j in i:
            j.flags.writeable = False
    return hash_string(''.join([hash_string(''.join(map(hash_array, l))) for l in my_list]))

