"""
Programmed by Jason Zutty
Modified by VIP Team
Implements a new data type for use with DEAP
"""
import copy
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
import time
import os
import pickle
import hashlib
import re
import gzip
import csv
import itertools
import operator
from pathlib import Path
from GPFramework.sql_connection_orm_cache import SQLConnectionCache
import scipy.sparse as sci

from GPFramework.cache_methods import save_cache, load_cache, load_hash, functional_hash

def load_feature_data_from_file(filename, use_cache=False, compress=False, hash_data=False):
    """
    Loads feature data written in a typical csv format:
    First row is the header starting with pound, or no header
    remaining rows are data.
    The filename should be a file under dataset/<dataset_name>/filename
    with train_<fold_num> or test_<fold_num> somewhere in the filename

    Args:
        filename: name of data file
        use_cache: if enabled, sets up directories within the datasets folder to store feature data for later use;
                   set to false by default.
        compress:  if enabled, compresses the data to slim down the space the cache will take up.;
                   However, it takes more time to write the data (time vs space tradeoff);
                   set to false by default.
        hash_data: if enabled, stores the hash of the entire fold of data. typically only used for test folds.;
                   set to false by default.

    Returns:
        A EMADE Object

    """
    save_data = False
    if use_cache:
        # Setup directories for data assuming that folders are prepped properly
        directory = os.fsdecode(filename)
        # Figure out if the file is test data or train data
        train_index = directory.find("train")
        test_index = directory.find("test")
        fold = None

        # If can't find any train data (meaning this file is test data), create the test folder for this fold
        # else do the opposite, note that the "test_index + 6" is literally the length of len("test_0")
        if train_index == -1:
            fold = re.sub('[_]', '', directory[test_index:test_index+6])
        else:
            fold = re.sub('[_]', '', directory[train_index:train_index+7])

        # If directory is set up correctly, base_directory should reference "/datasets/dota"
        # instead of "/datasets/dota/dota_data_set_..." using dota as an example, this is NOT "home/user"!
        base_directory = directory.split("/")
        base_directory = "/".join(base_directory[:2]) + "/" + fold + "/"
        gen_directory = os.fsencode(base_directory + "gen_data")

        if not os.path.exists(base_directory + "gen_data"):
            os.makedirs(base_directory + "gen_data")
            save_data = True

    feature_array = []
    label_list = []
    points = []
    with gzip.open(filename, 'rb') as my_file:
        first_line = True
        class_data = np.array([])
        feature_data = np.array([[]])
        for line in my_file:
            line = line.decode('utf-8')
            line = line.strip().split(',')
            if first_line and line[0].startswith('#'):
                first_line = False
                continue
            elif first_line:
                first_line = False
            else:
                class_data = np.array([np.float(line[-1])])
                feature_data = np.array([line[:-1]], dtype='d')

                feature_array.append(feature_data)
                label_list.append(class_data)

                point = EmadeDataInstance(target=class_data)
                point.set_stream(
                    StreamData(np.array([[]]))
                    )
                point.set_features(
                    FeatureData(feature_data)
                    )
                points.append(point)

    if use_cache:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        if save_data:
            # save data to cache
            f1 = open(os.fsdecode(gen_directory) + "/feature_" + "data_" + fold + ".npz", "wb")
            f2 = open(os.fsdecode(gen_directory) + "/stream_" + "data_" + fold + ".npz", "wb")
            f3 = open(os.fsdecode(gen_directory) + "/label_" + fold + ".npz", "wb")
            if compress:
                np.savez_compressed(file=f1, arr=feature_list)
                np.savez_compressed(file=f2, arr=stream_list)
                np.savez_compressed(file=f3, arr=label_list)
            else:
                np.savez(file=f1, arr=feature_list)
                np.savez(file=f2, arr=stream_list)
                np.savez(file=f3, arr=label_list)
            f1.close()
            f2.close()
            f3.close()

        if hash_data:
            # hash data
            data_hash = functional_hash([feature_list, stream_list, target_list])

            if save_data:
                # save the data hash for future
                with open(os.fsdecode(gen_directory) + "/" + "hash.txt", "wb") as f:
                    pickle.dump(str(data_hash), f)

            return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), data_hash)

        return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), None)

    elif False and hash_data:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        # hash data
        data_hash = functional_hash([feature_list, stream_list, target_list])
        return (EmadeData(points), data_hash)

    return (EmadeData(points), None)

def load_text_data_from_file(filename, use_cache=False, hash_data=False, compress=False):
    save_data = False
    if use_cache:
        # Setup directories for data assuming that folders are prepped properly
        directory = os.fsdecode(filename)
        # Figure out if the file is test data or train data
        train_index = directory.find("train")
        test_index = directory.find("test")
        fold = None

        # If can't find any train data (meaning this file is test data), create the test folder for this fold
        # else do the opposite, note that the "test_index + 6" is literally the length of len("test_0")
        if train_index == -1:
            fold = re.sub('[_]', '', directory[test_index:test_index+6])
        else:
            fold = re.sub('[_]', '', directory[train_index:train_index+7])

        # If directory is set up correctly, base_directory should reference "/datasets/dota"
        # instead of "/datasets/dota/dota_data_set_..." using dota as an example, this is NOT "home/user"!
        base_directory = directory.split("/")
        base_directory = "/".join(base_directory[:2]) + "/" + fold + "/"
        gen_directory = os.fsencode(base_directory + "gen_data")

        if not os.path.exists(base_directory + "gen_data"):
            os.makedirs(base_directory + "gen_data")
            save_data = True

    feature_array = []
    label_array = []
    points = []
    with gzip.open(filename, 'rt', encoding='UTF-8') as my_file:
        csv_reader = csv.reader(my_file,delimiter=',')
        first_line = True
        class_data = np.array([])
        feature_data = np.array([[]])
        for li, line in enumerate(csv_reader):
            #line = line.decode('utf-8')
            #line = line.strip().rsplit(',', 1)
            if first_line and line[0].startswith('#'):
                first_line = False
                continue
            elif first_line:
                first_line = False
            else:
                
                
                # class_data = np.array(line[-1][1:len(line[-1]) - 1].split(', ')).astype(np.float)
                class_data = np.array(line[1:]).astype(np.float)
                feature_data = np.array([line[0]], dtype=object)
                # print(line)
                # print("feature_data",feature_data)
                # print("class_data",class_data)
               

                feature_array.append(feature_data)
                label_array.append(class_data)

                #print(f"line[{li}]: {line}")
                #print(f"label_array[-1]: {label_array[-1]}")

                point = EmadeDataInstance(target=class_data)
                point.set_stream(
                    StreamData(np.array([[]]))
                    )
                point.set_features(
                    TextData(feature_data)
                    )
                points.append(point)

    if use_cache:
        feature_array = np.vstack([instance.get_features().get_data() for instance in points])
        stream_array = np.vstack([instance.get_stream().get_data() for instance in points])
        if save_data:
            # save data to cache
            f = open(os.fsdecode(gen_directory) + "/feature_" + "data_" + fold + ".npy", "wb")
            np.save(file=f, arr=feature_array)
            f.close()
            f = open(os.fsdecode(gen_directory) + "/stream_" + "data_" + fold + ".npy", "wb")
            np.save(file=f, arr=stream_array)
            f.close()
            # save labels
            f = open(os.fsdecode(gen_directory) + "/label_" + fold + ".npy", "wb")
            np.save(file=f, arr=np.vstack(label_array))
            f.close()

        if hash_data:
            # hash data
            data_hash = hash_function(hash_function(feature_array) + hash_function(stream_array))

            if save_data:
                # save the data hash for future
                with open(os.fsdecode(gen_directory) + "/" + "hash.txt", "wb") as f:
                    pickle.dump(str(data_hash), f)

    if use_cache:
        if hash_data:
            return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), data_hash)
        else:
            return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), None)
    else:
        return (EmadeData(points), None) 


    

def load_text_data_for_summary(filename, use_cache=False, hash_data=False, compress=False):
    save_data = False
    if use_cache:
        # Setup directories for data assuming that folders are prepped properly
        directory = os.fsdecode(filename)
        # Figure out if the file is test data or train data
        train_index = directory.find("train")
        test_index = directory.find("test")
        fold = None

        # If can't find any train data (meaning this file is test data), create the test folder for this fold
        # else do the opposite, note that the "test_index + 6" is literally the length of len("test_0")
        if train_index == -1:
            fold = re.sub('[_]', '', directory[test_index:test_index+6])
        else:
            fold = re.sub('[_]', '', directory[train_index:train_index+7])

        # If directory is set up correctly, base_directory should reference "/datasets/dota"
        # instead of "/datasets/dota/dota_data_set_..." using dota as an example, this is NOT "home/user"!
        base_directory = directory.split("/")
        base_directory = "/".join(base_directory[:2]) + "/" + fold + "/"
        gen_directory = os.fsencode(base_directory + "gen_data")

        if not os.path.exists(base_directory + "gen_data"):
            os.makedirs(base_directory + "gen_data")
            save_data = True

    feature_array = []
    label_array = []
    points = []
    with gzip.open(filename, 'rt') as my_file:
        csv_reader = csv.reader(my_file,delimiter=',')
        first_line = True
        class_data = np.array([])
        feature_data = np.array([[]])
        for line in csv_reader:
            #line = line.decode('utf-8')
            #line = line.strip().rsplit(',', 1)
            if first_line and line[0].startswith('#'):
                first_line = False
                continue
            elif first_line:
                first_line = False
            else:
                #class_data = np.array([np.float(line[-1])])
                #feature_data = np.array([line[:-1]], dtype=object)

                t = line[-1][1:len(line[-1]) - 1]
                #class_data = np.array(line[-1][1:len(line[-1]) - 1].split(', ')).astype(np.float)
                class_data = np.array([float(x) for x in t.split(" ")], dtype=np.float32)
                
                #feature_data = np.array([line[0].split('||')], dtype=object)
                feature_data = np.array([line[0]], dtype=np.float32)
                # results = list(map(int, results))
                # class_data = np.array(line[-1])
                

                feature_array.append(feature_data)
                label_array.append(class_data)

                point = EmadeDataInstance(target=class_data)
                point.set_stream(
                    StreamData(np.array([[]]))
                    )
                point.set_features(
                    TextData(feature_data)
                    )
                points.append(point)

    if use_cache:
        feature_array = np.vstack([instance.get_features().get_data() for instance in points])
        stream_array = np.vstack([instance.get_stream().get_data() for instance in points])
        if save_data:
            # save data to cache
            f = open(os.fsdecode(gen_directory) + "/feature_" + "data_" + fold + ".npy", "wb")
            np.save(file=f, arr=feature_array)
            f.close()
            f = open(os.fsdecode(gen_directory) + "/stream_" + "data_" + fold + ".npy", "wb")
            np.save(file=f, arr=stream_array)
            f.close()
            # save labels
            f = open(os.fsdecode(gen_directory) + "/label_" + fold + ".npy", "wb")
            np.save(file=f, arr=np.vstack(label_array))
            f.close()

        if hash_data:
            # hash data
            data_hash = hash_function(hash_function(feature_array) + hash_function(stream_array))

            if save_data:
                # save the data hash for future
                with open(os.fsdecode(gen_directory) + "/" + "hash.txt", "wb") as f:
                    pickle.dump(str(data_hash), f)

    if use_cache:
        if hash_data:
            return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), data_hash)
        else:
            return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), None)
    else:
        return (EmadeData(points), None) 



def load_many_to_one_from_file(filename, use_cache=False, compress=False, hash_data=False):
    """
    Loads stream data written in a many to one format:
        time,data,time,data,time,data,...,truth_val
    per sample

    Args:
        filename: name of data file
        use_cache: if enabled, sets up directories within the datasets folder to store feature data for later use;
                   set to false by default.
        compress:  if enabled, compresses the data to slim down the space the cache will take up.;
                   However, it takes more time to write the data (time vs space tradeoff);
                   set to false by default.
        hash_data: if enabled, stores the hash of the entire fold of data. typically only used for test folds.;
                   set to false by default.

    Returns:
        A EmadeData Object

    """
    save_data = False
    if use_cache:
        # Setup directories for data assuming that folders are prepped properly
        directory = os.fsdecode(filename)
        train_index = directory.find("train")
        test_index = directory.find("test")
        fold = None

        if train_index == -1:
            fold = re.sub('[_]', '', directory[test_index:test_index+6])
        else:
            fold = re.sub('[_]', '', directory[train_index:train_index+7])

        base_directory = directory.split("/")
        base_directory = "/".join(base_directory[:2]) + "/" + fold + "/"
        gen_directory = os.fsencode(base_directory + "gen_data")

        if not os.path.exists(base_directory + "gen_data"):
            os.makedirs(base_directory + "gen_data")
            save_data = True

    label_list = []
    points = []
    with gzip.open(filename, 'r') as my_file:
        # Start by reading the first line which specifies how many rows
        # per column, i.e. data will be written col,row,row,row,...col,...
        header = my_file.readline().decode('utf-8')
        # Check first line for N=Number
        match = re.search(r"N=(\d+)", header)
        if match:
            # Assign N to be what the file specifies
            N = int(match.group(1))
        else:
            # Default is one row per col, i.e. a vector
            N = 1
            # Reset the file pointer to the beginning of file
            my_file.seek(0)
        for line in my_file:
            line = line.decode('utf-8')
            counts = []
            line = line.strip().split(',')
            depth = np.array([float(line[-1])])
            #waveform = [float(elem) for elem in line[:-1]]
            for row in np.arange(N):
                counts.append(line[(row+1):-1:(N+1)])

            counts = np.array(counts, dtype='d')
            label_list.append(depth)
            point = EmadeDataInstance(target=depth)
            point.set_stream(
                ManyToOne(counts)
                )
            point.set_features(
                FeatureData(np.array([[]]))
                )
            points.append(point)

    if use_cache:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        if save_data:
            # save data to cache
            f1 = open(os.fsdecode(gen_directory) + "/feature_" + "data_" + fold + ".npz", "wb")
            f2 = open(os.fsdecode(gen_directory) + "/stream_" + "data_" + fold + ".npz", "wb")
            f3 = open(os.fsdecode(gen_directory) + "/label_" + fold + ".npz", "wb")
            if compress:
                np.savez_compressed(file=f1, arr=feature_list)
                np.savez_compressed(file=f2, arr=stream_list)
                np.savez_compressed(file=f3, arr=label_list)
            else:
                np.savez(file=f1, arr=feature_list)
                np.savez(file=f2, arr=stream_list)
                np.savez(file=f3, arr=label_list)
            f1.close()
            f2.close()
            f3.close()

        if hash_data:
            # hash data
            data_hash = functional_hash([feature_list, stream_list, target_list])

            if save_data:
                # save the data hash for future
                with open(os.fsdecode(gen_directory) + "/" + "hash.txt", "wb") as f:
                    pickle.dump(str(data_hash), f)

            return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), data_hash)

        return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), None)

    elif False and hash_data:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        # hash data
        data_hash = functional_hash([feature_list, stream_list, target_list])
        return (EmadeData(points), data_hash)

    return (EmadeData(points), None)

def load_many_to_many_from_file(filename, use_cache=False, compress=False, hash_data=False):
    """
    This method is designed to read in data that uses a "stream"
    of data to predict a "stream" of data. A potential example of this paradigm
    would be a filter design problem
    The format of this data will be interlaced, such that you have:
    Sample_Data
    Truth_Data
    Sample_Data
    Truth_Data
    etc...

    Therefore we will always have an even number of lines

    Args:
        filename: name of data file
        use_cache: if enabled, sets up directories within the datasets folder to store feature data for later use;
                   set to false by default.
        compress:  if enabled, compresses the data to slim down the space the cache will take up.;
                   However, it takes more time to write the data (time vs space tradeoff);
                   set to false by default.
        hash_data: if enabled, stores the hash of the entire fold of data. typically only used for test folds.;
                   set to false by default.

    Returns:
        A EmadeData object

    """
    save_data = False
    if use_cache:
        # Setup directories for data assuming that folders are prepped properly
        directory = os.fsdecode(filename)
        train_index = directory.find("train")
        test_index = directory.find("test")
        fold = None

        if train_index == -1:
            fold = re.sub('[_]', '', directory[test_index:test_index+6])
        else:
            fold = re.sub('[_]', '', directory[train_index:train_index+7])

        base_directory = directory.split("/")
        base_directory = "/".join(base_directory[:2]) + "/" + fold + "/"
        gen_directory = os.fsencode(base_directory + "gen_data")

        if not os.path.exists(base_directory + "gen_data"):
            os.makedirs(base_directory + "gen_data")
            save_data = True

    label_list = []
    points = []
    with gzip.open(filename, 'r') as my_file:
        # Start by reading the first line which specifies how many rows
        # per column, i.e. data will be written col,row,row,row,...col,...
        header = my_file.readline().decode('utf-8')
        # Check first line for N=Number
        match = re.search(r"N=(\d+)", header)
        if match:
            # Assign N to be what the file specifies
            N = int(match.group(1))
        else:
            # Default is one row per col, i.e. a vector
            N = 1
            # Reset the file pointer to the beginning of file
            my_file.seek(0)
        for lines in grouper(my_file, 2):
            lines = [line.decode('utf-8') for line in lines]
            if lines[1] is None:
                break
            line_1 = lines[0].strip().split(',')
            line_2 = lines[1].strip().split(',')

            counts_1 = []
            counts_2 = line_2[1::2]

            for row in np.arange(N):
                counts_1.append(line_1[(row+1)::(N+1)])

            counts_1 = np.array(counts_1, dtype='d')
            counts_2 = np.array(counts_2, dtype='d')

            label_list.append(counts_2)
            point = EmadeDataInstance(target=counts_2)
            point.set_stream(
                ManyToMany(counts_1)
                )
            point.set_features(
                FeatureData(np.array([[]]))
                )
            points.append(point)

    if use_cache:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        if save_data:
            # save data to cache
            f1 = open(os.fsdecode(gen_directory) + "/feature_" + "data_" + fold + ".npz", "wb")
            f2 = open(os.fsdecode(gen_directory) + "/stream_" + "data_" + fold + ".npz", "wb")
            f3 = open(os.fsdecode(gen_directory) + "/label_" + fold + ".npz", "wb")
            if compress:
                np.savez_compressed(file=f1, arr=feature_list)
                np.savez_compressed(file=f2, arr=stream_list)
                np.savez_compressed(file=f3, arr=label_list)
            else:
                np.savez(file=f1, arr=feature_list)
                np.savez(file=f2, arr=stream_list)
                np.savez(file=f3, arr=label_list)
            f1.close()
            f2.close()
            f3.close()

        if hash_data:
            # hash data
            data_hash = functional_hash([feature_list, stream_list, target_list])

            if save_data:
                # save the data hash for future
                with open(os.fsdecode(gen_directory) + "/" + "hash.txt", "wb") as f:
                    pickle.dump(str(data_hash), f)

            return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), data_hash)

        return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), None)

    elif False and hash_data:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        # hash data
        data_hash = functional_hash([feature_list, stream_list, target_list])
        return (EmadeData(points), data_hash)

    return (EmadeData(points), None)

def load_many_to_some_from_file(filename, use_cache=False, compress=False, hash_data=False):
    """
    This method is designed to read in data that uses a "stream"
    of data to predict a "stream" of data. A potential example of this paradigm
    would be a filter design problem
    The format of this data will be interlaced, such that you have:
    Sample_Data
    Truth_Data
    Sample_Data
    Truth_Data
    etc...
    Therefore we will always have an even number of lines

    Args:
        filename: name of data file
        use_cache: if enabled, sets up directories within the datasets folder to store feature data for later use;
                   set to false by default.
        compress:  if enabled, compresses the data to slim down the space the cache will take up.;
                   However, it takes more time to write the data (time vs space tradeoff);
                   set to false by default.
        hash_data: if enabled, stores the hash of the entire fold of data. typically only used for test folds.;
                   set to false by default.

    Returns:
        A EmadeData object
    """
    save_data = False
    if use_cache:
        # Setup directories for data assuming that folders are prepped properly
        directory = os.fsdecode(filename)
        train_index = directory.find("train")
        test_index = directory.find("test")
        fold = None

        if train_index == -1:
            fold = re.sub('[_]', '', directory[test_index:test_index+6])
        else:
            fold = re.sub('[_]', '', directory[train_index:train_index+7])

        base_directory = directory.split("/")
        base_directory = "/".join(base_directory[:2]) + "/" + fold + "/"
        gen_directory = os.fsencode(base_directory + "gen_data")

        if not os.path.exists(base_directory + "gen_data"):
            os.makedirs(base_directory + "gen_data")
            save_data = True

    points = []
    label_list = []
    with gzip.open(filename, 'r') as my_file:
        # N Represents how many cols in the flattened stream data
        # M Represents how many cols in the truth data
        header = my_file.readline().decode('utf-8')

        # Check first line for N=Number, M=Number
        match_N = re.search(r"N=(\d+)", header)
        N = int(match_N.group(1)) if match_N else 1
        match_M = re.search(r"M=(\d+)", header)
        M = int(match_M.group(1)) if match_M else N

        if not match_N and not match_M:
            # Reset the file pointer to the beginning of file
            my_file.seek(0)

        for lines in grouper(my_file, 2):
            lines = [line.decode('utf-8') for line in lines]
            if lines[1] is None:
                break

            stream_data = []
            truth_data = []
            line_1 = lines[0].strip().split(',')
            line_2 = lines[1].strip().split(',')

            rows = len(line_1)//N
            for row in np.arange(rows):
                stream_data.append(line_1[(N*row):(N*row+N)])
            rows = len(line_2)//M
            for row in np.arange(rows):
                truth_data.append(line_2[(M*row):(M*row+M)])

            stream_data = np.array(stream_data, dtype='d')

            truth_data = np.array(truth_data, dtype='d')
            label_list.append(truth_data)

            #if stream_data.shape[1] != truth_data.shape[1]:
            #    raise ValueError('Sample and truth data length mismatched: {}, {}'.format(stream_data.shape[1], truth_data.shape[1]))

            point = EmadeDataInstance(target=truth_data)
            point.set_stream(
                ManyToMany(stream_data)
                )
            point.set_features(
                FeatureData(np.array([[]]))
                )
            points.append(point)

    if use_cache:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        if save_data:
            # save data to cache
            f1 = open(os.fsdecode(gen_directory) + "/feature_" + "data_" + fold + ".npz", "wb")
            f2 = open(os.fsdecode(gen_directory) + "/stream_" + "data_" + fold + ".npz", "wb")
            f3 = open(os.fsdecode(gen_directory) + "/label_" + fold + ".npz", "wb")
            if compress:
                np.savez_compressed(file=f1, arr=feature_list)
                np.savez_compressed(file=f2, arr=stream_list)
                np.savez_compressed(file=f3, arr=label_list)
            else:
                np.savez(file=f1, arr=feature_list)
                np.savez(file=f2, arr=stream_list)
                np.savez(file=f3, arr=label_list)
            f1.close()
            f2.close()
            f3.close()

        if hash_data:
            # hash data
            data_hash = functional_hash([feature_list, stream_list, target_list])

            if save_data:
                # save the data hash for future
                with open(os.fsdecode(gen_directory) + "/" + "hash.txt", "wb") as f:
                    pickle.dump(str(data_hash), f)

            return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), data_hash)

        return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), None)

    elif False and hash_data:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        # hash data
        data_hash = functional_hash([feature_list, stream_list, target_list])
        return (EmadeData(points), data_hash)

    return (EmadeData(points), None)

def load_image_csv_from_file(filename, use_cache=False, compress=False, hash_data=False):
    """
    This method is designed to read in data that uses a "stream"
    of data to predict a "stream" of data.

    This method supports any image data with the shape (c, m, n)
    where m x n is the dimensions of a single image
    and c is the number of channels with a minimum = 1

    The format of this data will be interlaced, such that you have:
    Sample_Data
    Truth_Data
    Sample_Data
    Truth_Data
    etc...
    Therefore we will always have an even number of lines

    Args:
        filename: name of data file
        use_cache: if enabled, sets up directories within the datasets folder to store feature data for later use;
                   set to false by default.
        compress:  if enabled, compresses the data to slim down the space the cache will take up.;
                   However, it takes more time to write the data (time vs space tradeoff);
                   set to false by default.
        hash_data: if enabled, stores the hash of the entire fold of data. typically only used for test folds.;
                   set to false by default.

    Returns:
        A EmadeData object
    """
    save_data = False
    if use_cache:
        # Setup directories for data assuming that folders are prepped properly
        directory = os.fsdecode(filename)
        train_index = directory.find("train")
        test_index = directory.find("test")
        fold = None

        if train_index == -1:
            fold = re.sub('[_]', '', directory[test_index:test_index+6])
        else:
            fold = re.sub('[_]', '', directory[train_index:train_index+7])

        base_directory = directory.split("/")
        base_directory = "/".join(base_directory[:2]) + "/" + fold + "/"
        gen_directory = os.fsencode(base_directory + "gen_data")

        if not os.path.exists(base_directory + "gen_data"):
            os.makedirs(base_directory + "gen_data")
            save_data = True

    # get lists of numpy arrays
    image_data = []
    truth_data = []
    with gzip.open(filename, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        m, n, num_channels = next(reader)
        m = int(m)
        n = int(n)
        num_channels = int(num_channels)
        next(reader)

        i = 0
        toggle = True
        for row in reader:
            row = np.array([float(i) for i in row])
            if i % 2 == 0:
                if toggle:
                    # reconstruct data from flat row
                    if num_channels == 1 or num_channels == 0:
                        image_data.append(row.reshape((m, n)))
                    else:
                        image_data.append(row.reshape((num_channels, m, n)).reshape((m, n, num_channels)))
                    toggle = False
                else:
                    truth_data.append(np.stack(np.split(row, len(row) / 2), axis=0))
                    toggle = True
            i += 1

    points = []
    label_list = []
    for image, truth in zip(image_data, truth_data):
        label_list.append(truth)
        # standardize truth format
        # if len(truth.shape) >= 1:
        #     truth = np.expand_dims(truth, axis=0)

        point = EmadeDataInstance(target=truth)
        point.set_stream(
            ManyToMany(image)
            )
        point.set_features(
            FeatureData(np.array([[]]))
            )
        points.append(point)


    if use_cache:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        if save_data:
            # save data to cache
            f1 = open(os.fsdecode(gen_directory) + "/feature_" + "data_" + fold + ".npz", "wb")
            f2 = open(os.fsdecode(gen_directory) + "/stream_" + "data_" + fold + ".npz", "wb")
            f3 = open(os.fsdecode(gen_directory) + "/label_" + fold + ".npz", "wb")
            if compress:
                np.savez_compressed(file=f1, arr=feature_list)
                np.savez_compressed(file=f2, arr=stream_list)
                np.savez_compressed(file=f3, arr=label_list)
            else:
                np.savez(file=f1, arr=feature_list)
                np.savez(file=f2, arr=stream_list)
                np.savez(file=f3, arr=label_list)
            f1.close()
            f2.close()
            f3.close()

        if hash_data:
            # hash data
            data_hash = functional_hash([feature_list, stream_list, target_list])

            if save_data:
                # save the data hash for future
                with open(os.fsdecode(gen_directory) + "/" + "hash.txt", "wb") as f:
                    pickle.dump(str(data_hash), f)

            return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), data_hash)

        return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), None)

    elif False and hash_data:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        # hash data
        data_hash = functional_hash([feature_list, stream_list, target_list])
        return (EmadeData(points), data_hash)

    return (EmadeData(points), None)


'''Programmed by Austin Dunn'''
def load_images_from_file(base_directory, use_cache=False, compress=False, hash_data=False):
    """
    Loads image files in a typical format

    Args:
        filename: name of data file
        use_cache: if enabled, sets up directories within the datasets folder to store feature data for later use;
                   set to false by default.
        hash_data: if enabled, stores the hash of the entire fold of data. typically only used for test folds.;
                   set to false by default.
        compress:  if enabled, compresses the data to slim down the space the cache will take up.;
                   However, it takes more time to write the data (time vs space tradeoff);
                   set to false by default.

    Returns:
        A EmadeData object

    """
    save_data = False

    # Setup directories for data assuming that folders are prepped properly
    directory = os.fsencode(base_directory)
    gen_directory = os.fsencode(base_directory + "gen_data")
    fold = re.sub('[/]', '', base_directory[-7:])

    if not os.path.exists(base_directory + "gen_data"):
        os.makedirs(base_directory + "gen_data")
        save_data = True

    # format image files in correct order
    dir = os.listdir(directory)
    trimmed_dir = []
    image_dir = []
    for file in dir:
        file = os.fsdecode(file)
        if file.endswith(".jpeg") or file.endswith(".jpg") or \
           file.endswith(".png") or file.endswith(".tif"):
            image_dir.append(file)
            trimmed_dir.append(re.sub(r"[A-Za-z.]", "", file))
    for i in range(len(trimmed_dir)):
        trimmed_dir[i] = trimmed_dir[i].split("_")
        trimmed_dir[i] = (int(trimmed_dir[i][0]), int(trimmed_dir[i][1]))

    mapping = dict(zip(trimmed_dir, image_dir))
    sorted_list = []
    for key in sorted(mapping.keys(), key=operator.itemgetter(0,1)):
        sorted_list.append(mapping[key])

    # convert image files into numpy arrays
    data = []
    for f in sorted_list:
        filename = os.fsdecode(f)
        img = load_img(base_directory + filename)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        data.append(x)

    # open labels csv file
    labels = pd.read_csv(base_directory + "labels_" + fold +".csv", header=None).values

    label_data = []
    for i in range(labels.shape[0]):
        row = labels[i, :]
        label_data.append(row)

    # create instances
    points = []
    for i in range(len(data)):
        points.append(EmadeDataInstance(features=FeatureData(data=np.array([[]])), stream=StreamData(data=data[i]), target=label_data[i]))

    if use_cache:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        if save_data:
            # save data to cache
            f1 = open(os.fsdecode(gen_directory) + "/feature_" + "data_" + fold + ".npz", "wb")
            f2 = open(os.fsdecode(gen_directory) + "/stream_" + "data_" + fold + ".npz", "wb")
            f3 = open(os.fsdecode(gen_directory) + "/label_" + fold + ".npz", "wb")
            if compress:
                np.savez_compressed(file=f1, arr=feature_list)
                np.savez_compressed(file=f2, arr=stream_list)
                np.savez_compressed(file=f3, arr=target_list)
            else:
                np.savez(file=f1, arr=feature_list)
                np.savez(file=f2, arr=stream_list)
                np.savez(file=f3, arr=target_list)
            f1.close()
            f2.close()
            f3.close()

        if hash_data:
            # hash data
            data_hash = functional_hash([feature_list, stream_list, target_list])

            if save_data:
                # save the data hash for future
                with open(os.fsdecode(gen_directory) + "/" + "hash.txt", "wb") as f:
                    pickle.dump(str(data_hash), f)

            return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), data_hash)

        return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), None)

    elif False and hash_data:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        # hash data
        data_hash = functional_hash([feature_list, stream_list, target_list])
        return (EmadeData(points), data_hash)

    return (EmadeData(points), None)

def load_feature_pickle_from_file(filename, use_cache=False, compress=False, hash_data=False):
    save_data = False
    if use_cache:
        # Setup directories for data assuming that folders are prepped properly
        directory = os.fsdecode(filename)
        train_index = directory.find("train")
        test_index = directory.find("test")
        fold = None

        if train_index == -1:
            fold = re.sub('[_]', '', directory[test_index:test_index+6])
        else:
            fold = re.sub('[_]', '', directory[train_index:train_index+7])

        base_directory = directory.split("/")
        base_directory = "/".join(base_directory[:2]) + "/" + fold + "/"
        gen_directory = os.fsencode(base_directory + "gen_data")

        if not os.path.exists(base_directory + "gen_data"):
            os.makedirs(base_directory + "gen_data")
            save_data = True

    # load .npz file
    full_data = np.load(filename, allow_pickle=True)['arr']
    points = []
    for row in full_data:
        feature, truth = (row[0], row[1])
        # print(i, row[0],row[1])
        feature_data = np.array([row.astype(np.float64) for row in feature])
        #point = EmadeDataInstance(target=np.expand_dims(truth, axis=0))
        point = EmadeDataInstance(target = truth)
        point.set_stream(
            StreamData(np.array([[]]))
            )
        point.set_features(
            FeatureData(feature_data)
            )
        points.append(point)

    if use_cache:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        if save_data:
            # save data to cache
            f1 = open(os.fsdecode(gen_directory) + "/feature_" + "data_" + fold + ".npz", "wb")
            f2 = open(os.fsdecode(gen_directory) + "/stream_" + "data_" + fold + ".npz", "wb")
            f3 = open(os.fsdecode(gen_directory) + "/label_" + fold + ".npz", "wb")
            if compress:
                np.savez_compressed(file=f1, arr=feature_list)
                np.savez_compressed(file=f2, arr=stream_list)
                np.savez_compressed(file=f3, arr=target_list)
            else:
                np.savez(file=f1, arr=feature_list)
                np.savez(file=f2, arr=stream_list)
                np.savez(file=f3, arr=target_list)
            f1.close()
            f2.close()
            f3.close()

        if hash_data:
            # hash data
            data_hash = functional_hash([feature_list, stream_list, target_list])

            if save_data:
                # save the data hash for future
                with open(os.fsdecode(gen_directory) + "/" + "hash.txt", "wb") as f:
                    pickle.dump(str(data_hash), f)

            return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), data_hash)

        return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), None)

    elif False and hash_data:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        # hash data
        data_hash = functional_hash([feature_list, stream_list, target_list])
        return (EmadeData(points), data_hash)

    return (EmadeData(points), None)

def load_pickle_from_file(filename, use_cache=False, compress=False, hash_data=False):
    """
    This method is designed to read in data that uses a common pickled format

    [ [examples, truths] <- one instance
      [examples, truths]
      ...
      [examples, truths] ]

    where examples: [ example 1,
                      example 2,
                      ...
                      example n ]

    And

    where truths:    [ truth 1,
                       truth 2,
                       ...
                       truth n ]

    Each example and truth are numpy arrays with arbitrary shape and dtype

    Note: if n = 1 then you can load in normal feature/stream datasets such as titanic

    Args:
        filename: name of data file
        use_cache: if enabled, sets up directories within the datasets folder to store feature data for later use;
                   set to false by default.
        compress:  if enabled, compresses the data to slim down the space the cache will take up.;
                   However, it takes more time to write the data (time vs space tradeoff);
                   set to false by default.
        hash_data: if enabled, stores the hash of the entire fold of data. typically only used for test folds.;
                   set to false by default.

    Returns:
        A EmadeData object
    """
    save_data = False
    if use_cache:
        # Setup directories for data assuming that folders are prepped properly
        directory = os.fsdecode(filename)
        train_index = directory.find("train")
        test_index = directory.find("test")
        fold = None

        if train_index == -1:
            fold = re.sub('[_]', '', directory[test_index:test_index+6])
        else:
            fold = re.sub('[_]', '', directory[train_index:train_index+7])

        base_directory = directory.split("/")
        base_directory = "/".join(base_directory[:2]) + "/" + fold + "/"
        gen_directory = os.fsencode(base_directory + "gen_data")

        if not os.path.exists(base_directory + "gen_data"):
            os.makedirs(base_directory + "gen_data")
            save_data = True

    # load .npz file
    full_data = np.load(filename, allow_pickle=True)['arr']
    points = []
    for row in full_data:
        stream, truth = (row[0], row[1])
        # print(i, row[0],row[1])
        stream = np.array([row.astype(np.float16) for row in stream])
        #point = EmadeDataInstance(target=np.expand_dims(truth, axis=0))
        point = EmadeDataInstance(target = truth)
        point.set_stream(
            ManyToMany(np.array(stream))
            )
        point.set_features(
            FeatureData(np.array([[]]))
            )
        points.append(point)
    if use_cache:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        if save_data:
            # save data to cache
            f1 = open(os.fsdecode(gen_directory) + "/feature_" + "data_" + fold + ".npz", "wb")
            f2 = open(os.fsdecode(gen_directory) + "/stream_" + "data_" + fold + ".npz", "wb")
            f3 = open(os.fsdecode(gen_directory) + "/label_" + fold + ".npz", "wb")
            if compress:
                np.savez_compressed(file=f1, arr=feature_list)
                np.savez_compressed(file=f2, arr=stream_list)
                np.savez_compressed(file=f3, arr=target_list)
            else:
                np.savez(file=f1, arr=feature_list)
                np.savez(file=f2, arr=stream_list)
                np.savez(file=f3, arr=target_list)
            f1.close()
            f2.close()
            f3.close()

        if hash_data:
            # hash data
            data_hash = functional_hash([feature_list, stream_list, target_list])

            if save_data:
                # save the data hash for future
                with open(os.fsdecode(gen_directory) + "/" + "hash.txt", "wb") as f:
                    pickle.dump(str(data_hash), f)

            return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), data_hash)

        return (EmadeData(points, base_directory=base_directory, gen_directory=os.fsdecode(gen_directory), fold=fold), None)

    elif False and hash_data:
        feature_list = [instance.get_features().get_data() for instance in points]
        stream_list = [instance.get_stream().get_data() for instance in points]
        target_list = [instance.get_target() for instance in points]
        # hash data
        data_hash = functional_hash([feature_list, stream_list, target_list])
        return (EmadeData(points), data_hash)

    return (EmadeData(points), None)

def grouper(iterable, n, fillvalue=None):
    """
    A helper function that creates a list of iter objects from `iterable`, of size `n`;
    Returns an iterator that aggregates elements from each of the iterables.

    Args:
        iterable: object from which to create a list of duplicates of iter objects.
        n: size of returned iterator
        fillvalue: if the iterables are of an odd length, the rest of the iterator is filled in with this value.

    Returns:
        an itertools.Iterator that aggregates elements from n * [iter(iterable)].

    """
    args = [iter(iterable)]*n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

class EmadeDataObject(object):
    """
    Serves as the parent class for EMADE data objects.

    Represented by three attributes:
        _data: an array containing data.
        _labels: an array labeling the data.

    """

    def __init__(self, data=np.array([[]])):
        # Let's call the methods of the class to set the data
        self.set_data(data)


    def get_data(self):
        """
        Returns the data stored in the object.

        """
        return self._data

    def set_data(self, data):
        """
        Sets the data object.
        Optionally takes in a corresponding set of labels for the data.

        Args:
            data: data object

        Raises:
            ValueError: if data is not all finite

        """
        # For debugging purposes I am going to raise an exception if the data is not all finite,
        # I may leave this in, as the sci-kit methods seem to require all data be finite
        if isinstance(data, list) or (isinstance(data, np.ndarray) and data.dtype == object):
            for i in data:
                if isinstance(i, np.ndarray) and i.dtype == object:
                    i = [j for j in i]
                elif type(i) == dict:
                    i = [i[k] for k in i]
                if not np.all(np.isfinite(i)):
                    raise ValueError('Non-finite data produced: ' + str(data) + ' ' + str(type(data)))
        elif not np.all(np.isfinite(data)):
            raise ValueError('Non-finite data produced: ' + str(data) + ' ' + str(type(data)))

        self._data = data

    def type_check(self, other):
        """
        Checks to see if the types of self and other match.

        Raises:
            ValueError: if types of self and other do not match

        """
        if type(self) is not type(other):
            raise ValueError('The two objects are of different classes.')


    def __add__(self, other):
        """
        ADDITION

        Returns:
            A new object with (self data PLUS other data) and labels

        Raises:
            ValueError: if data sizes do not match

        """

        if isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = type(self)(
                    data=self.get_data() + other
                    )
            return new
        else:
            self.type_check(other)

            if self.get_data().shape == other.get_data().shape or (other.get_data().shape[0] == self.get_data().shape[0] and (other.get_data().shape[1] == 1 or self.get_data().shape[1] == 1)):
                new = type(self)(
                        data=self.get_data() + other.get_data()
                        )
                return new
            else:
                raise ValueError('Data sizes did not match.')

    def __sub__(self, other):
        """
        SUBTRACTION

        Returns:
            A new object with (self data MINUS other data) and labels

        Raises:
            ValueError: if data sizes do not match

        """

        if isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = type(self)(
                    data=self.get_data() - other
                )
            return new
        else:
            self.type_check(other)

            if self.get_data().shape == other.get_data().shape or (other.get_data().shape[0] == self.get_data().shape[0] and (other.get_data().shape[1] == 1 or self.get_data().shape[1] == 1)):
                new = type(self)(
                        data=self.get_data() - other.get_data()
                        )
                return new
            else:
                raise ValueError('Data sizes did not match.')

    def __mul__(self, other):
        """
        MULTIPLICATION

        Returns:
            A new object with (self data TIMES other data) and labels

        Raises:
            ValueError: if data sizes do not match

        """

        if isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = type(self)(
                    data=self.get_data() * other
                )
            return new
        else:
            self.type_check(other)

            if self.get_data().shape == other.get_data().shape or (other.get_data().shape[0] == self.get_data().shape[0] and (other.get_data().shape[1] == 1 or self.get_data().shape[1] == 1)):
                new = type(self)(
                        data=self.get_data() * other.get_data()
                        )
                return new
            else:
                raise ValueError('Data sizes did not match.')

    def __truediv__(self, other):
        """
        DIVISION

        Returns:
            A new object with (self data DIVIDED BY other data) and labels

        Raises:
            ValueError: if data sizes do not match

        """

        if isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = type(self)(
                    data=self.get_data() / other
                )
            return new
        else:
            self.type_check(other)
            # We can divide if shapes match or if either has one column, i.e. is a scalar
            if self.get_data().shape == other.get_data().shape or (other.get_data().shape[0] == self.get_data().shape[0] and (other.get_data().shape[1] == 1 or self.get_data().shape[1] == 1)):
                new = type(self)(
                        data=self.get_data() / other.get_data()
                        )
                return new
            else:
                raise ValueError('Data sizes did not match.')

    def __repr__(self):
        return '\n'.join(['Data:',str(self._data)])

class FeatureData(EmadeDataObject):
    """
    This class represents the parent class for feature type data.
    """
    def __init__(self,
            data=np.array([[]])
            ):
        super(FeatureData, self).__init__(data)

class TextData(FeatureData):
    """
    This class represents the parent class for feature type data.
    """
    def __init__(self,
            data=np.array([[]])
            ):
        super(TextData, self).__init__(data)

    def set_data(self, data):
        """
        Sets the data object.
        Optionally takes in a corresponding set of labels for the data.

        Args:
            data: data object
        """
        self._data = data

class StreamData(EmadeDataObject):
    """
    This class represents the parent class for stream type data.
    """
    def __init__(self,
            data=np.array([[]])
            ):
        super(StreamData, self).__init__(data)

#WARNING: The slice method of this data type is not cached (Do not use as is)
class OneToOne(StreamData):
    """
    This class represents stream data where the target data lines up
    on a one to one basis with the raw data, i.e. each point has a target value
    in the time series.
    """
    def __init__(self,
            data=np.array([[]])
            ):

        super(OneToOne, self).__init__(data)

    def slice(self, step, win):
        """
        Slices each instance into pieces of size 'win'.
        - win must be < len(self.labels)
        - step must be < win

        Args:
            win: window size (size of each 'chunk')
            step: step size (how far forward to move chunk's starting position after each iteration)

        """
        new_data = []
        new_target = []

        self._data = self._data.flatten()
        self._target = self._target.flatten()

        # num_slices is an undefined variable :(
        for i in np.arange(num_slices):
            new_data.append(self._data[(i*step):(i*step+win)])
            new_target.append(self._target[(i*step):(i*step+win)])

        self._data = np.array(new_data)
        self._target = np.array(new_target)

class ManyToOne(StreamData):
    """
    This class represents stream data where there is only one point of
    target data per raw data, i.e. each series has a target value.
    """
    def __init__(self,
            data=np.array([[]])
            ):

        super(ManyToOne, self).__init__(data)

class ManyToMany(StreamData):
    """
    This class represents stream data where there is only one point of
    target data per raw data, i.e. each series has a target value.
    """
    def __init__(self,
            data=np.array([[]])
            ):
        super(ManyToMany, self).__init__(data)

    # def slice(self, step, win):
    #     """
    #     Slice each instance in to pieces of size N
    #     """
    #     new_labels = []
    #     new_data = []
    #     new_target = []

    #     self._labels = self._labels.flatten()
    #     self._data = self._data.flatten()
    #     self._target = self._target.flatten()

    #     num_slices = (len(self._labels)-win+step)/step

    #     for i in np.arange(num_slices):
    #         new_labels.append(self._labels[(i*step):(i*step+win)])
    #         new_data.append(self._data[(i*step):(i*step+win)])
    #         new_target.append(self._target[(i*step):(i*step+win)])

    #     self._labels = np.array(new_labels)
    #     self._data = np.array(new_data)
    #     self._target = np.array(new_target)

    # def combine_targets(self, other):
    #     if self.get_target().shape == other.get_target().shape:
    #         return np.logical_or(self.get_target(), other.get_target())
    #     else:
    #         raise ValueError('Target arrays have different sizes.')

#WARNING: Operators not safe for caching
class EmadeDataInstance(object):
    """
    This is a class where each implementation of it is a training or testing
    instance EMADE will operate on a deapDataInstance collection.
    Methods will handle the collection of these instances.

    Each deapDataInstance will contain two members
        data:  a numpy.ndarray
        truth: a scalar value on what the data should produce

    """

    def __init__(self,
            features=FeatureData(),
            stream=StreamData(),
            target=np.array([])
            ):
        self._features = None
        self._stream = None
        self._target = None
        self.set_features(features)
        self.set_stream(stream)
        self.set_target(target)

    def get_features(self):
        """
        Returns the feature object of the data instance.
        The feature object will be of some subclass of FeatureData.
        """
        return self._features

    def set_features(self, features):
        """
        Set the feature object of a data instance,
        features must be a subclass of FeatureData.

        Raises:
            ValueError: if feature is not a derived class of FeatureData

        """
        if isinstance(features, FeatureData):
            self._features = features
        else:
            raise ValueError(
                    'Features need to be a derived class of FeatureData.'
                    )

    def get_stream(self):
        """
        Returns the stream object of the data instance.
        The stream object will be of some subclass of StreamData.
        """
        return self._stream

    def set_stream(self, stream):
        """
        Set the stream object of a data instance.
        stream must be a subclass of StreamData.

        Raises:
            ValueError: if stream is not a derived class of StreamData

        """
        if isinstance(stream, StreamData):
            self._stream = stream
        else:
            raise ValueError('Stream needs to be a derived class of StreamData')

    def get_target(self):
        """
        Returns the target for the data, i.e. The value the data represents.
        """
        return self._target

    def set_target(self, target):
        """
        Set the target array
        """
        self._target = target

    # Pulled this from the EmadeDataObject class to keep one target for both stream
    # And features
    def combine_targets(self, other, func):
        """
        Abstract placeholder of parent method
        """
        if len(self.get_target().shape) == 1 and len(other.get_target().shape) == 1:
            return self.get_target() #or other.get_target()
        elif self.get_target().shape == other.get_target().shape:
            return func(self.get_target(), other.get_target())
        else:
            raise ValueError('Child class should implement this method')


    """
    WARNING: Do NOT use these operators in primitives with caching turned on
             These are safe for unit tests which do not use caching
    """
    def __add__(self, other):
        """
        ADDITION

        Returns:
            A new object with (self features/stream PLUS other features/stream)

        Raises:
            ValueError: if type of other is incompatible with operation

        """

        if isinstance(other, EmadeDataInstance):
            new = EmadeDataInstance(target=self.combine_targets(other, operator.__add__))
            new.set_features(self.get_features() + other.get_features())
            new.set_stream(self.get_stream() + other.get_stream())
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = EmadeDataInstance(target=self.get_target())
            new.set_features(self.get_features() + other)
            new.set_stream(self.get_stream() + other)
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __sub__(self, other):
        """
        SUBTRACTION

        Returns:
            A new object with (self features/stream MINUS other features/stream)

        Raises:
            ValueError: if type of other is incompatible with operation

        """

        if isinstance(other, EmadeDataInstance):
            new = EmadeDataInstance(target=self.combine_targets(other, operator.__sub__))
            new.set_features(self.get_features() - other.get_features())
            new.set_stream(self.get_stream() - other.get_stream())
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = EmadeDataInstance(target=self.get_target())
            new.set_features(self.get_features() - other)
            new.set_stream(self.get_stream() - other)
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __mul__(self, other):
        """
        MULTIPLICATION

        Returns:
            A new object with (self features/stream TIMES other features/stream)

        Raises:
            ValueError: if type of other is incompatible with operation

        """

        if isinstance(other, EmadeDataInstance):
            new = EmadeDataInstance(target=self.combine_targets(other, operator.__mul__))
            new.set_features(self.get_features() * other.get_features())
            new.set_stream(self.get_stream() * other.get_stream())
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = EmadeDataInstance(target=self.get_target())
            new.set_features(self.get_features() * other)
            new.set_stream(self.get_stream() * other)
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __truediv__(self, other):
        """
        TRUE DIVISION
        In python 3.X there is the / operator, which typically returns a float, as well
        as a // operator (__floordiv__), which typically returns an int.

        Returns:
            A new object with (self features/stream DIVIDED BY other features/stream)

        Raises:
            ValueError: if type of other is incompatible with operation

        """

        if isinstance(other, EmadeDataInstance):
            new = EmadeDataInstance(target=self.combine_targets(other, operator.__truediv__))
            new.set_features(self.get_features() / other.get_features())
            new.set_stream(self.get_stream() / other.get_stream())
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = EmadeDataInstance(target=self.get_target())
            new.set_features(self.get_features() / other)
            new.set_stream(self.get_stream() / other)
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __repr__(self):
        return '\n'.join(['Features:', str(self.get_features()),
                          'Stream:', str(self.get_stream()),
                          'Target:', str(self.get_target())])

#WARNING: Operators not safe for caching
'''Modified by Austin Dunn, Anish Thite, and Mohan Dodda'''
class EmadeData(object):
    """
    This class represents the collection of instances in to a single
    data set.

    Args:
        instances: object containing the numpy arrays for the data and target
        base_directory: local disk directory for locating fold of the data
                        for example - datasets/titanic/test0/
        gen_directory: local disk directory for locating fold of the data
                        for example - datasets/titanic/test0/signal_method_0_12.3_
        fold: unique string, usually in the format of [train/test][integer]
              for example - train0 or test3
    """
    def __init__(self,
            instances=np.array([]),
            base_directory=None,
            gen_directory=None,
            fold=None
            ):
        """
        Subject ids are used for folding the data.
        """
        self._instances = []
        self.set_instances(instances)
        print(self._instances[0]._stream._data.shape)
        self._base_directory = base_directory
        self._gen_directory = gen_directory
        self._fold = fold

    def get_instances(self):
        """
        Returns the instance list.
        """
        return self._instances

    def get_base_directory(self):
        """
        Returns the base directory.
        """
        return self._base_directory

    def get_gen_directory(self):
        """
        Returns the gen directory.
        """
        return self._gen_directory

    def set_gen_directory(self, directory):
        """
        Sets the gen directory of the data
        """
        self._gen_directory = directory

    def set_base_directory(self, directory):
        """
        Sets the base directory of the data
        """
        self._base_directory = directory

    def get_fold(self):
        """
        Returns the fold.
        """
        return self._fold

    def update_directories(self, base, gen):
        """
        Update base and gen directories
        """
        self._base_directory = base
        self._gen_directory = gen

    def set_instances(self, instances):
        """
        Sets the objects instances list with the given list of instances.

        Raises:
            ValueError: if instance not of EmadeDataInstances
        """

        if all(
            isinstance(instance, EmadeDataInstance) for instance in instances
            ):
            # The copy is here, because otherwise two EmadeData instances
            # Could share EmadeDataInstance instances, and a change to one
            # would propagate to the other
            self._instances = [instance for instance in instances]
        else:
            raise ValueError('Expected an array of EmadeDataInstances')

    def get_numpy(self):
        """
        Returns the data objects features as a numpy array.
        """
        first_row = self._instances[0].get_features().get_data()
        if (sci.issparse(first_row)):
            return sci.vstack([instance.get_features().get_data()
                         for instance in self._instances \
                         if len(instance.get_features().get_data()) > 0])
        else:
            return np.vstack([instance.get_features().get_data()
                         for instance in self._instances \
                         if len(instance.get_features().get_data()) > 0])

    def get_target(self):
        """
        Returns the data objects targets as a numpy array.
        """

        return np.squeeze(np.array([np.array(instance.get_target()) for instance in self._instances]))  #np.squeeze is to turn 1d target 1 dimensional
        #IMP: right now the instance level target is encapsulated by nunmpy array in case of detection data if we want code to be cleaner, we can deal with this in the evaluation method instead!
        #IMP: if np.squeeze is giving any problems, try the commented code below!
        # target = np.array([np.array(instance.get_target()) for instance in self._instances]) 
        # if len(target.shape)>1 and target.shape[1] == 1: #turn 1d target 1 dimensional
        #     target = target.flatten()
        # return target
        



    def get_flattenned_target(self): 
        """
        flattens target data into 1d - for target data in detection - if wanted can be part of get_target with a param indicating to do this
        """
        return np.hstack([instance.get_target()
                         for instance in self._instances \
                         if len(instance.get_target()) > 0])



    def set_target(self, labels):
        """
        Sets the data objects targets to appropriate values
        """
        if len(labels) != len(self._instances): #for detection data in learner, can turn this into separate method if wanted
            ind = 0
            for inst in self._instances:
                data = inst.get_features().get_data()
                if len(data) > 0:
                    r = len(data) if len(data.shape) > 1 else 1
                    target = np.array([labels[i] for i in range(ind, ind+r)])
                    ind += r
                    inst.set_target(target)
        else:
            for instance, row in zip(self._instances,labels):
                instance.set_target(np.array(row).reshape(-1,))


        
                                


    """
    WARNING: Do NOT use these operators in primitives with caching turned on
             These are safe for unit tests which do not use caching
    """
    def __add__(self, other):
        """
        ADDITION

        Returns:
            A new set with (self instances PLUS other instances)

        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if  (isinstance(other, EmadeData) and
              len(self.get_instances()) == len(other.get_instances())
            ):
            new.set_instances([inst_1 + inst_2 for inst_1, inst_2
                               in
                               zip(self.get_instances(), other.get_instances())
                               ])
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new.set_instances([inst_1 + other for inst_1
                                   in self.get_instances()])
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __sub__(self, other):
        """
        SUBTRACTION

        Returns:
            A new set with (self instances MINUS other instances)

        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if (isinstance(other, EmadeData)
             and len(self.get_instances()) == len(other.get_instances())
            ):
            new.set_instances([inst_1 - inst_2 for inst_1, inst_2
                               in
                               zip(self.get_instances(), other.get_instances())
                               ])
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new.set_instances([inst_1 - other for inst_1
                                   in self.get_instances()])
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __mul__(self, other):
        """
        MULTIPLICATION

        Returns:
            A new set with (self instances TIMES other instances)

        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if (isinstance(other, EmadeData)
            and len(self.get_instances()) == len(other.get_instances())
            ):
            new.set_instances([inst_1 * inst_2 for inst_1, inst_2
                               in
                               zip(self.get_instances(), other.get_instances())
                               ])
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new.set_instances([inst_1 * other for inst_1
                                   in self.get_instances()])
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __truediv__(self, other):
        """
        DIVISION

        Returns:
            A new set with (self instances DIVIDED BY other instances)

        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if (isinstance(other, EmadeData)
            and len(self.get_instances()) == len(other.get_instances())
            ):
            new.set_instances([inst_1 / inst_2 for inst_1, inst_2
                               in
                               zip(self.get_instances(), other.get_instances())
                               ])
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new.set_instances([inst_1 / other for inst_1
                                   in self.get_instances()])
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __repr__(self):
        return '\n'.join([str(instance) for instance in self.get_instances()])

class EmadeDataPair(object):
    """
    This class groups together a training and testing data object,
    so that they may be passed around the genetic programming structure
    in tandem.

    Args:
        train_data: tuple (EmadeData, string)
                    the string is always None (EmadeData, None)

        test_data: tuple (EmadeData, string)
                   the string is either None (EmadeData, None) or
                   the hash of the test data numpy array (EmadeData, hash)

    Non-init Args:
        _caching_mode: True if caching is being used
                       False otherwise
        _central: True if cache is being stored in one central drive
                  False if each host is storing its own cache
        _db_map: dictionary storing all the arguments for creating
                 a database connection
        _time_threshold: parameter for minimum time needed to store data in cache
        _ind_hash: individual hash used to keep track of methods the datapair has gone through
        _truth_data: truth data: used for early stopping (which is used in nnlearner for now)
        _datatype: type of data being used
        _multilabel: True if data is multilabel
                     False otherwise
        _num_params: number of parameters for NNLearners (None by default)
        _regression: True if data is regression task
                      False if data is not regressiont task (classification)
    """

    def __init__(self, train_data, test_data):
        self._hash = test_data[1]
        self._train_data = None
        self._test_data = None
        self._caching_mode = False
        self._central = True
        self._compression = False
        self._cache_limit = 0
        self._db_map= {}
        self._time_threshold = 0
        self._ind_hash = None
        self._db_use = False
        self._truth_data = None
        self._datatype = ''
        self._multilabel = False
        self._num_params = None
        self._regression = False
        self.modList = []

        self.set_train_data(train_data[0])
        self.set_test_data(test_data[0])

        if self._train_data is None:
            self._train_data = EmadeData()
        if self._test_data is None:
            self._test_data = EmadeData()

    def get_train_data(self):
        """
        Return the training data set,
        data is of class EmadeData.
        """
        return self._train_data

    def set_train_data(self, train_data):
        """
        Set the training data set,
        train_data must be of type EmadeData.
        """
        if isinstance(train_data, EmadeData):
            self._train_data = train_data

    def get_test_data(self):
        """
        Return the testing data set,
        data is of class EmadeData.
        """
        return self._test_data

    def set_test_data(self, test_data):
        """
        Set the testing data set,
        data is of class EmadeData.
        """
        if isinstance(test_data, EmadeData):
            self._test_data = test_data

    def get_hash(self):
        """
        Return current hash of data
        """
        return self._hash

    def get_sha1(self):
        """
        Return the sha1 hash of the data pair.
        """
        pickled_string = pickle.dumps(self, -1)
        hash_function = hashlib.sha1()
        hash_function.update(pickled_string)
        return hash_function.hexdigest()

    def get_caching_mode(self):
        """
        Return the caching mode
        """
        return self._caching_mode

    def get_central(self):
        """
        Return the centralization mode
        """
        return self._central

    def set_caching_mode(self, mode):
        """
        Set the caching mode
        """
        self._caching_mode = mode

    def set_central(self, value):
        """
        Set the centralization mode
        """
        self._central = value

    def get_connection(self):
        """
        Return a connection to the database
        """
        return SQLConnectionCache(**self._db_map, ind_hash=self._ind_hash)

    def set_db_map(self, map):
        """
        Set the database info map
        """
        self._db_map = map

    def set_threshold(self, th):
        """
        Set the time threshold for caching
        """
        self._time_threshold = th

    def get_threshold(self):
        """
        Return the time threshold for caching
        """
        return self._time_threshold

    def set_compression(self, a_bool):
        """
        Set the compression mode for caching
        """
        self._compression = a_bool

    def get_compression(self):
        """
        Return whether to use compression for caching
        """
        return self._compression

    def set_cache_limit(self, value):
        """
        Set the size limit for caching
        """
        self._cache_limit = value

    def get_cache_limit(self):
        """
        Return cache size limit
        """
        return self._cache_limit * 2

    def save_data(self, train_data, test_data, method_string, time, connection, row, target=False):
        """
        Save data to cache and store location
        """
        self._train_data, self._test_data, self._hash, c_row = save_cache(train_data, test_data,
                                                                          self._compression,
                                                                          method_string, self._ind_hash,
                                                                          connection,
                                                                          row, target=target)
        return c_row

    def load_data(self, hash, connection, row, target=False):
        self._train_data.set_gen_directory(self._train_data.get_base_directory() + hash)
        self._test_data.set_gen_directory(self._test_data.get_base_directory() + hash)
        self._train_data, self._test_data = load_cache(self._train_data,
                                                       self._test_data,
                                                       connection,
                                                       row, target)

    def update_hash(self):
        """
        Update hash using current data
        """
        feature = [np.array(i.get_features().get_data()) for i in self._test_data.get_instances()]
        stream = [np.array(i.get_stream().get_data()) for i in self._test_data.get_instances()]
        target = [np.array(i.get_target()) for i in self._test_data.get_instances()]
        self._hash = functional_hash([feature, stream, target])

    def set_hash(self, directory):
        """
        Set the current hash of test data
        """
        self._hash = load_hash(directory)
        return self._hash

    def set_ind_hash(self, hash):
        """
        Set the hash of the individual being run
        """
        self._ind_hash = hash

    def get_ind_hash(self):
        """
        Get the hash of the individual being run
        """
        return self._ind_hash

    def get_base_test_directory(self):
        """
        Return base directory of testing data
        """
        return self._test_data.get_base_directory()

    def get_base_train_directory(self):
        """
        Return base directory of training data
        """
        return self._train_data.get_base_directory()

    def set_db_use(self):
        """
        Sets Database Use to True
        This always gets called in EMADE
        And is optional outside of EMADE (such as unit tests)
        """
        # TODO: Change this back when submodule primitives are fixed
        self._db_use = False

    def get_db_use(self):
        """
        Returns Database Use Boolean
        """
        return self._db_use

    def set_datatype(self, datatype):
        """
        Sets datatype to type of data being loaded for future use in EMADE
        """
        self._datatype = datatype

    def get_datatype(self):
        """
        Returns datatype of the EmadeDataPair
        """
        return self._datatype

    def set_multilabel(self, multilabel):
        """
        Sets boolean to indicate if data is multilabel data or not
        """
        self._multilabel = multilabel

    def get_multilabel(self):
        """
        Returns if the data is multilabel data or not
        """
        return self._multilabel

    def set_num_params(self, num_params):
        """
        Set number of parameters (if NNLearner)
        If there are aready number parameters logged it will add to it
        """
        if self._num_params is None:
            self._num_params = num_params
        else:
            self._num_params += num_params

    def get_num_params(self):
        """
        Return the number of Parameters (if NNLearner)
        """
        return self._num_params
    def get_truth_data(self): 
        """
        Return the truth data
        """
        return self._truth_data
    def set_truth_data(self):
        """
        Only use in the loading data process!!!!!!
        """
        self._truth_data = copy.deepcopy(self._test_data.get_target())

    def get_regression(self):
        return self._regression 

    def set_regression(self, regression):
        self._regression = regression
    def __add__(self, other):
        """
        ADDITION
        Returns:
            A new GTMOEPDataPair with (self train_data/test_data PLUS other train_data/test_data)
        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if isinstance(other, EmadeDataPair):
            new = EmadeDataPair(
                self.get_train_data() + other.get_train_data(),
                self.get_test_data() + other.get_test_data()
                )
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = EmadeDataPair(
                self.get_train_data() + other,
                self.get_test_data() + other
                )
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __sub__(self, other):
        """
        SUBTRACTION
        Returns:
            A new GTMOEPDataPair with (self train_data/test_data MINUS other train_data/test_data)
        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if isinstance(other, EmadeDataPair):
            new = EmadeDataPair(
                self.get_train_data() - other.get_train_data(),
                self.get_test_data() - other.get_test_data()
                )
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = EmadeDataPair(
                self.get_train_data() - other,
                self.get_test_data() - other
                )
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __mul__(self, other):
        """
        MULTIPLICATION
        Returns:
            A new GTMOEPDataPair with (self train_data/test_data TIMES other train_data/test_data)
        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if isinstance(other, EmadeDataPair):
            new = EmadeDataPair(
                self.get_train_data() * other.get_train_data(),
                self.get_test_data() * other.get_test_data()
                )
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = EmadeDataPair(
                self.get_train_data() * other,
                self.get_test_data() * other
                )
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __truediv__(self, other):
        """
        DIVISION
        Returns:
            A new GTMOEPDataPair with (self train_data/test_data DIVIDED BY other train_data/test_data)
        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if isinstance(other, EmadeDataPair):
            new = EmadeDataPair(
                self.get_train_data() / other.get_train_data(),
                self.get_test_data() / other.get_test_data()
                )
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = EmadeDataPair(
                self.get_train_data() / other,
                self.get_test_data() / other
                )
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __repr__(self):
        return '\n'.join(['Train', str(self.get_train_data()),
                          'Test', str(self.get_test_data())])






class EmadeDataPairNN(EmadeDataPair):
    """
    Copy of emade data pair used specifically by Neural Network primitives

    Hypothetically, this constructor won't run (and emade only cares about the existence
    of this object), but we'll have it return an actual data pair just in case
    """
    def __init__(self, train_data, test_data):
        super().__init__(train_data, test_data)


class EmadeDataPairNNF(EmadeDataPair):
    """
    Copy of emade data pair used specifically by Neural Network primitives

    Hypothetically, this constructor won't run (and emade only cares about the existence
    of this object), but we'll have it return an actual data pair just in case
    """
    def __init__(self, train_data, test_data):
        super().__init__(train_data, test_data)