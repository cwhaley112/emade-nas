import numpy as np
import copy as cp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GRU, Activation, Conv2D, LeakyReLU

from tensorflow.keras.activations import linear
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.utils import to_categorical
from GPFramework.learner_methods import makeFeatureFromClass
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras_pickle_wrapper import KerasPickleWrapper
import GPFramework.text_processing_methods as ps
from tensorflow.keras.initializers import RandomUniform, Constant



from collections import UserList

#Pytorch imports
import torch
import torch.nn as nn

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')



import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import pickle
import dill
import copy as cp
import numpy as np
import math
import sys
import time
import re
import gc
from GPFramework.general_methods import mod_select, target_value_check
import scipy.sparse as sci
from lightgbm import LGBMClassifier
import cv2      
import matplotlib.pyplot as plt 
from PIL import Image   
import numpy as np

MAXLEN=500
NUMWORDS=5000
BATCH_LIMIT=200
EPOCH_LIMIT=6


import tensorflow as tf
import keras.backend.tensorflow_backend as tfb

POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned






def weighted_binary_crossentropy(target, output):
        """
        Weighted binary crossentropy between an output tensor
        and a target tensor. POS_WEIGHT is used as a multiplier
        for the positive targets.

        Combination of the following functions:
        * keras.losses.binary_crossentropy
        * keras.backend.tensorflow_backend.binary_crossentropy
        * tf.nn.weighted_cross_entropy_with_logits
        """
        # transform back to logits
        _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.math.log(output / (1 - output))
        # compute weighted loss
        loss = tf.nn.weighted_cross_entropy_with_logits(labels=target,
                                                                                                        logits=output,
                                                                                                        pos_weight=POS_WEIGHT)
        return tf.reduce_mean(loss, axis=-1)


#define a layer list so that we can pass the list up to the Sequential()
#model. Layer primitives will add layer functions to the list 

class LayerList(UserList):
    """ 
    Wrapper class for a list to hold layers
    """
    def __init__(self):
        super().__init__()


def LinearLayer(out_dim, layerlist):    
        """Creates linear layer 
        Args:   
                out_dim: ouput dimension        
                activation_func: object of class Activation, specifies activation function to use       
        Returns:        
                torch Linear Layer      
        """     
        # model.model.add(Dense(out_dim, activation=activation_func))   
        out_dim = abs(out_dim)  
        layerlist.mylist.append({"DENSE": [out_dim, 'relu']})   
        return layerlist

def LSTMLayer(out_dim, layerlist):
        """Creates lstm layer

        Args:
                out_dim: ouput dimension

        Returns:
                Keras LSTM Layer
        """
        out_dim = abs(out_dim)+1
        layerlist.append({'LSTM': [out_dim]})
        return layerlist

def GRULayer(out_dim, layerlist):
        """Creates gru layer

        Args:
                out_dim: ouput dimension

        Returns:
                Keras GRU Layer
        """
        out_dim = abs(out_dim)+1
        # model.model.add(GRU(out_dim))
        layerlist.mylist.append({'GRU': [out_dim]})
        return layerlist

def DropoutLayer(rate, layerlist):
        """Creates Dropout layer
        Args:
                rate: float between 0 and 1 that indicates fraction of the input units to drop.

        Returns:
                Dropout Layer    
        """
        rate =rate%1
        layerlist.mylist.append({'DROPOUT': [rate]})
        return layerlist

def OutputLayer(data_pair, layerlist, bitmask=True):
        """Creates Output layer which matches the output dimension of the predictions
        Currently only supports 1 mode (bitmask mode)
        TODO: architecture mode.
        """
        out_dim = data_pair.get_train_data().get_target().shape[1]
        layerlist.mylist.append({'OUTPUT': [out_dim]})
        return layerlist



def ReluLayer(layerlist):
        """Creates ReLU layer

        Returns:
                Keras ReLU activation layer
        """
        layerlist.mylist.append({'RELU': ['relu']})
        return layerlist

def ELUActivationLayer(layerlist):
        '''Creates Activation layer using ELU function

        Returns:
                Keras Activation layer using ELU
        '''
        layerlist.mylist.append({'ELU': ['elu']})
        return layerlist

def LeakyReLULayer(alpha, layerlist):
        """Creates Leaky ReLU layer

        Args:
                alpha= value that f(x) equals when x<0

        Returns:
                Keras Leaky ReLU Layer
        """
        layerlist.mylist.append({'LEAKYRELU': [alpha]})
        return layerlist

def get_glove_embedding_matrix(out_dim, vocab_size, tokenizer): 
        """Creates glove embedding matrix

        Returns:
                glove embedding matrix
        """

        embeddings_dictionary = dict()  
        glove_file = open('glove.6B.100d.txt', encoding="utf8") 
        for line in glove_file: 
                records = line.split()  
                word = records[0]       
                vector_dimensions = np.asarray(records[1:], dtype='float32')    
                embeddings_dictionary[word] = vector_dimensions 
        glove_file.close()      
        embedding_matrix = np.zeros((vocab_size, 100))  
        for word, index in tokenizer.word_index.items():        
                embedding_vector = embeddings_dictionary.get(word)      
                if embedding_vector is not None:        
                        embedding_matrix[index] = embedding_vector      
        return embedding_matrix 

#TODO: Add in glove to EmbeddingLayer 
def EmbeddingLayer(out_dim, data_pair, use_glove):      
        """Creates Embedding layer      
        Args:   
                empty_model: empty_model as terminal    
                data_pair: given dataset        
                out_dim: ouput dimension        
        Returns:        
                Keras Embedding Layer   
        """     
        out_dim = abs(out_dim)
        data_pair, vocab_size, tok = ps.tokenizer(data_pair, MAXLEN, NUMWORDS)
        empty_list = LayerList()
        empty_list.append(nn.Embedding(vocab_size, out_dim))
        return empty_list
        #maxlen = MAXLEN        
        #numwords=NUMWORDS      
        #out_dim = abs(out_dim) 
        #data_pair, vocab_size, tok  = ps.tokenizer(data_pair, maxlen, numwords)        
        #if use_glove is True:  
        #       initializer = Constant(get_glove_embedding_matrix(out_dim, vocab_size, tok))    
        #else:  
        #       initializer = RandomUniform()   
        #empty_layerlist = LayerList()  
        #empty_layerlist.mylist.append({'EMBEDDING': [vocab_size, out_dim, maxlen, initializer]})       
        #return empty_layerlist

def MaxPoolingLayer(pool_dim, pool_size, strides):
        """Creates MaxPooling layer

        Args:
                pool_dim: output dimension string, including 1D, 2D and 3D
                pool_size: pooling size
                strides: stride for pooling operations
                
        Returns:
                Keras MaxPooling Layer
        """
        layerlist.mylist.append({'MAXPOOLING': [pool_dim, pool_size, strides]})
        return layerlist

def Conv1DLayer(output_dim, kernel_dim, strides, activation):
        """Creates Conv1D layer
        
        Args:
                output_dim: output dimension
                kernel_dim: kernel size
                strides: stride for conv operations
                activation: activation function
                
        Returns:
           Keras Conv1D Layer
        """
        layerlist.mylist.append({'CONV1D': [output_dim, kernel_dim, strides, activation]})
        return layerlist

def Conv2DLayer(output_dim, kernel_dim, strides, activation):
        """Creates Conv2D layer

        Args:
                output_dim: output dimension
                kernel_dim: kernel size
                strides: stride for conv operations
                activation: activation function
                
        Returns:
                Keras Conv2D Layer
        """
        layerlist.mylist.append({'CONV2D': [output_dim, kernel_dim, strides, activation]})
        return layerlist

def NNLearner(data_pair, layerlist, batch_size, epochs):
    """
    Core method for all neural nets
    Creates the model
    Handles caching functions
    Loads and store the data and classification

    Args:
        data_pair: data structure storing train and test data
        model: model with layers already added
        batch_size: batch size for training (will be modded with BATCH_LIMIT)
        epochs: epochs for training (will be modded with EPOCH_LIMIT)

    Returns:
        updated data pair with new data
    """
    data_pair = cp.deepcopy(data_pair)




    """
    Cache [Load]
    """
    if data_pair.get_caching_mode():
        previous_hash = data_pair.get_hash()
        base_directory = data_pair.get_base_test_directory()
        l_params = re.sub("[,:]", "_", re.sub("[_ '}{]", "", str(learner.learnerParams)))
        e_params = re.sub("[,:]", "_", re.sub("[_ '}{]", "", str(learner.ensembleParams)))
        method_string = learner.learnerName + "_" + l_params + "_" + learner.ensemble + "_" + e_params + "_"
        file_string = method_string + previous_hash
        data_pair.add_to_key_list(file_string)
        key_directory = base_directory + file_string

        database = data_pair.get_connection()

        cache_hit, data_row = database.query_data(data_pair.get_central(), file_string,
                                                  data_pair.get_base_train_directory(),
                                                  base_directory, data_pair.get_threshold())
        if cache_hit:
            data_pair.set_hash(key_directory)
            data_pair.load_data(file_string, database, data_row, learner=True)
            return data_pair

    start_time = time.time()
    try:
        '''
        Main Block
        '''
        model = []
        model = Sequential()
        #add all of the layers in from layerlist
        for layer in layerlist.mylist:
            name = list(layer.keys())[0]
            params = layer[name]
            if  name == 'DENSE':
                model.add(Dense(params[0], activation=params[1]))
            elif name == 'GRU':
                model.add(GRU(params[0]))
            elif name == 'LSTM':
                model.add(LSTM(params[0]))
            elif name == 'EMBEDDING':
                model.add(Embedding(params[0], params[1], input_length=params[2], embeddings_initializer=params[3]))

            elif name == 'CONV1D': 
                # TODO: look into how this can be used with activation that's to be integrated to EMADE
                model.add(Conv1D(params[0], params[1], strides=params[2], padding='SAME', activation=params[3]))

            elif name == 'MAXPOOLING': # might not be the best way to add stuff
                if params[0] == 1: 
                    int_size = params[1][0] # check if this works
                    model.add(MaxPooling1D(pool_size=params[1], strides=params[2], padding='SAME', data_format='channels_last')) # last is the default; might use first for performance
                elif params[0] == 2: 
                    model.add(MaxPooling2D( pool_size=tuple([params[1], params[1]]), 
                                            strides=tuple([params[2], params[2]]), 
                                            padding='SAME', 
                                            data_format='channels_last'))
                elif params[0] == 3: 
                    model.add(MaxPooling3D(pool_size=tuple([params[1], params[1], params[1]]),
                                            strides=tuple([params[2], params[2], params[2]]), 
                                            padding='SAME', 
                                            data_format='channels_last'))
            
            elif name == 'DROPOUT':
                model.add(Dropout(params[0]))
            elif name == 'OUTPUT':
                model.add(Dense(params[0], activation=sigmoid))
            elif name == 'CONV2D': 
                model.add(Conv2D(params[0], params[1], strides=params[2], padding='SAME', activation=params[3]))
            elif name == 'RELU':
                model.add(Activation(params[0]))
            elif name == 'ELU':
                model.add(Activation(params[0]))
            elif name == 'LEAKYRELU':
                model.add(LeakyReLU(alpha=params[0]))
                
        #get size of multilabel targets in bitvector form.
        
        #TODO: if multilabel, change the metrics
        ml = data_pair.get_multilabel()
        if ml:
            model.compile(loss = weighted_binary_crossentropy, optimizer = 'adam', metrics= ['accuracy'])
        else:
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #wrap model in kpw for pickling, from here on access model from model()
        model = KerasPickleWrapper(model)
        
        """
        Load data
        Validate data
        """
        tokenized_data_pair, vocab_size, tok = ps.tokenizer(data_pair, MAXLEN, NUMWORDS)
        training_object = tokenized_data_pair.get_train_data()
        testing_object = tokenized_data_pair.get_test_data()
        # # training_data = np.vstack([instance.get_features().get_data() for instance in training_object.get_instances()])
        # # testing_data = np.vstack([instance.get_features().get_data() for instance in testing_object.get_instances()])
        # # target_values = np.vstack([inst.get_target() for inst in training_object.get_instances()])
        training_data = training_object.get_numpy()
        testing_data = testing_object.get_numpy()
        target_values = training_object.get_target()
        target_values = data_pair.get_train_data().get_target()
        # # Check for multiple target values
        # target_value_check(target_values)
        # # For debugging purposes let's print out name of the function and dimensions of the data
        print('NN' + ': training', training_data.shape)
        sys.stdout.flush()
        print('NN' + ': labels', target_values.shape)
        print(target_values)

        # Temporary fix for problem with EmadeData. A more elegant solution may exist.
        # EmadeData stores a set of EmadeDataInstance in a python list, called self._inst
        # EmadeData.get_target() calls np.hstack on a [i.get_target() for i in self._inst]
        # Since EmadeDataInstance.get_target() is a 1D np.array, np.hstack
        # behaves like np.concatenate(axis=1). see scipy docs for more details.
        #
        # Basically, the output of EmadeData.get_target() needs to be reshaped into a 2D np.array.
        # if ml_size > 0:
        #     target_values = target_values.reshape(-1, ml_size)

        print(testing_data.shape)
        sys.stdout.flush()

        """
        Fit estimator to training data
        Predict labels of testing data
        """


        prepredicted_classes = model().predict(testing_data)
        # TODO: pick single label training metric.
        # metric = "weighted_binary_crossentropy" if ml_size > 0 else "acc"

        #set batch sizes and epochs based on limit
        batch_size = batch_size % BATCH_LIMIT
        epochs = epochs % EPOCH_LIMIT
        history = model().fit(training_data, target_values, batch_size=batch_size, epochs=epochs)
        print('NN' + ': testing', testing_data.shape)
        sys.stdout.flush()
        predicted_classes = model().predict(testing_data)
        
        #round predictions to get classes
        predicted_classes = np.round(predicted_classes)
        #[inst.set_target(target) for inst, target in zip(data_pair.get_test_data().get_instances(), predicted_classes)]
        
        #set num params in data pair
        data_pair.set_num_params(model().count_params())   

        #import tensorflow.keras.backend as bek #:)
        #del model
        #clear_session()
        #gc.collect()
        
        for inst, target in zip(data_pair.get_test_data().get_instances(), predicted_classes):
            # if ml_size > 0:
            #     inst.set_target(target)
            # else:
            #     inst.set_target([target])
            inst.set_target(target)
        """
        Add new feature to data based on predicted labels
        """
        # Let's make the predictions a feature through use of the make feature from class,
        # But then restore the training data to the class
        # Set the self-predictions of the training data
        # trained_classes = model().predict(training_data)
        # [inst.set_target([target]) for inst, target in
        # zip(data_pair.get_train_data().get_instances(), trained_classes)]

        # data_pair = makeFeatureFromClass(data_pair, name='NN')
        # # Restore the training data
        # [inst.set_target([target]) for inst, target in
        # zip(data_pair.get_train_data().get_instances(), target_values)]

        """
        Cache [Store]
        """
        if data_pair.get_caching_mode():
            gen_time = time.time() - start_time
            if gen_time > data_pair.get_threshold():
                data_pair.save_data(data_pair.get_train_data(), data_pair.get_test_data(),
                                    method_string, gen_time, database, data_row,
                                    target=[[i] for i in predicted_classes])
            else:
                database.update_dirty(gen_time, data_row)


    except (KeyboardInterrupt, Exception) as e:
        """
        Handle Errors
        """
        if data_pair.get_caching_mode():
            if isinstance(e, KeyboardInterrupt):
                database.set_method_time(data_row, time.time() - start_time)
                del database
            else:
                if "kill_process" in str(e):
                    database.set_special_error("time/memory errored occurred \
                                               but data was saved")
                    del database
                    # after the data is saved we need to kill the process
                    # before it goes further past the time/memory limit
                    os.kill(os.getpid(), 9)
                database.set_error(data_row, str(e))
                del database
        raise

    if data_pair.get_caching_mode():
        del database

    gc.collect();
    np.save('predictionsalldone', predicted_classes)
    return data_pair
   
def bbox_grid(top_left_x, top_left_y, width, height):   
        """ Prepares grid for cropping from bounding box information    
        Args:   
                top_left_x: top left x coordinate of bounding box       
                top_left_y: top left y coordinate of bounding box       
                width: width of bouding box     
                height: height of bounding box  
                        
        Returns:        
                Grid tuple with top left coordinate and bottom right coordinate 
        """     
        bottom_right_x = top_left_x + width     
        bottom_right_y = top_left_y + height    
        grid = (top_left_x, top_left_y, bottom_right_x, bottom_right_y) 
        return grid     
        
def crop_img(raw_img_file, grid):       
        """ Crops given image file      
        Args:   
                raw_img_file: raw image file    
                grid: grid with top left coordinate and bottom right coordinate 
        Returns:        
                Cropped image as array  
        """     
        raw_img = Image.open(raw_img_file)      
        cropped_img = raw_img.crop(grid)        
        cropped_img = np.asarray(cropped_img)   
        return cropped_img
        
def warp_img(raw_img, cols=227, rows=227):      
        """ Warps an image to given dimensions  
        Args:   
                raw_img: raw image as array     
                cols: # of columns (width) to set image, 227 by default 
                rows: # of rows (height) to set image, 227 by default   
        Returns:        
                Warped image as array   
        """     
        warped_img = cv2.resize(raw_img, (cols, rows))  
        return warped_img

def compiled(num_classes):
        """provides the list of arguments for the for the model compiler

        Args:
                num_classes: the number of classes within the target data

        Returns:
                return arguments the configuration of the model including the optimizer, the loss function and the list of metric to be evaluated
        """
        if num_classes <= 2:
                return ['binary_crossentropy', 'adam', ['accuracy']]
        else:
                return ['categorical_crossentropy', 'adam', ['accuracy']]
