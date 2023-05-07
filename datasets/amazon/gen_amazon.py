import random
import csv
import bz2
import gzip
import os

import sklearn.model_selection
import kaggle
import numpy as np

# Downloads from https://www.kaggle.com/bittlingmayer/amazonreviews
# Follow instructions on https://github.com/Kaggle/kaggle-api to set up kaggle

# total test.ft.txt size is 400,000 lines
# total train.ft.txt size is 3,600,000 lines
# we take the full test dataset, and only 180,000 lines from train

# CHANGE THIS if you want a different percentage of the train.ft.txt as your train.csv, total size is 400,000
train_size = 0.05

kaggle.api.dataset_download_cli('bittlingmayer/amazonreviews', unzip=True)

# I don't think these do anything because we specify a random_state in the StratifiedShuffleSplit
random.seed(111)
np.random.seed(111)

def transform_data(line):
    line = line.decode('utf-8').rstrip('\n')
    return [line[11:], int(line[9])-1]

with bz2.BZ2File('./test.ft.txt.bz2', 'r') as test_in, gzip.open('./test.csv.gz', 'wt') as test_out:
    writer = csv.writer(test_out, delimiter=',')
    for line in test_in:
        writer.writerow(transform_data(line))
if os.path.isfile('./test.ft.txt.bz2'):
    os.remove('./test.ft.txt.bz2')

train_Y = []
# populate labels so that we can create a stratified sample of 5% of data (we can't load all data into memory)
with bz2.BZ2File('./train.ft.txt.bz2', 'r') as train_in:
    for line in train_in:
        review, label = transform_data(line)
        train_Y.append(label)

sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=111)
train_inds = sorted(list(sss.split(np.zeros(len(train_Y)), train_Y))[0][0])  # get the indices of the train split
# now that we know the stratified split, we can get the X data cooresponding to train_inds
with bz2.BZ2File('./train.ft.txt.bz2', 'r') as train_in, gzip.open('./train.csv.gz', 'wt') as train_out:
    cur_train_ind = 0
    cur_train_file_ind = 0
    writer = csv.writer(train_out, delimiter=',')
    for line in train_in:
        if cur_train_ind >= len(train_inds):
            break
        if cur_train_file_ind == train_inds[cur_train_ind]:
            writer.writerow(transform_data(line))
            cur_train_ind += 1
        cur_train_file_ind += 1
if os.path.isfile('./train.ft.txt.bz2'):
    os.remove('./train.ft.txt.bz2')
