#!/usr/bin/env python3
#-*- coding: utf-8 *-*

# -------------------------------- #
# Built-in Imports                 #
# -------------------------------- #

import datetime
import json
import yaml
import logging
import os
import sys
import time

from pprint import pprint

# -------------------------------- #
# Machine Learning Imports         #
# -------------------------------- #

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import auc as auc_test

from tensorflow import keras

# -------------------------------- #
# Functions                        #
# -------------------------------- #

def get_compiled_model(params_dict: dict):
    version : str = params_dict['version']
    if version == 'get_compiled_model_v1':
        return get_compiled_model_v1(params_dict=params_dict)
    if version == 'get_compiled_model_v2':
        return get_compiled_model_v2(params_dict=params_dict)
    raise Exception(f"ERROR: '{version}' is not allowed.")

def get_compiled_model_v1(params_dict: dict = None):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=5, output_dim=16))
    model.add(tf.keras.layers.LSTM(32,
        return_sequences=True,
        input_shape=(16, 1)))

    model.add(tf.keras.layers.LSTM(16, activation='relu'))
    model.add(tf.keras.layers.Dense(72))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy'])
    
    print(model.summary())
    tf.keras.utils.plot_model(model, 'model_graph.png', show_shapes=True)

    return model

def compile_model(model, num_classes: int = -1, type_problem: str = 'regression'):

    # Regression
    if type_problem == 'regression':
            model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
    
    elif type_problem == 'classification':
        if num_classes > 2:
            model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
            # model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy', 'categorical_crossentropy'])
            model.compile(
                optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'categorical_crossentropy'])
        
        elif num_classes == 2:
            model.add(tf.keras.layers.Dense(num_classes-1,activation='sigmoid'))
    
            model.compile(
                optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0),
                loss='binary_crossentropy',
                metrics=['accuracy', 'binary_crossentropy'])
        else:
            raise Exception(f"ERROR: '{num_classes}' is not allowed.")
    return model

def get_compiled_model_v2(params_dict: dict = None):

    embedding_size: int = params_dict['embedding_size']
    vocab_size: int = params_dict['vocab_size']
    lstm_units: int = params_dict['lstm_units']
    input_length: int = params_dict['maxlen']
    num_classes: int = params_dict['num_classes'] if 'num_classes' in params_dict.keys() else -1
    type_problem: int = params_dict['type_problem'] if 'type_problem' in params_dict.keys() else 'regression'

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_size, input_length = input_length))
    model.add(tf.keras.layers.SpatialDropout1D(0.4))
    model.add(tf.keras.layers.LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2))
    
    print(model.summary())
    tf.keras.utils.plot_model(model, 'model_graph.png', show_shapes=True)

    model = compile_model(model, num_classes, type_problem)

    return model