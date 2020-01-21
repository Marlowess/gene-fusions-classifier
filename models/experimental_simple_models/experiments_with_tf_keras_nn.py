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

def get_compiled_model_v2(params_dict: dict = None):

    embedding_size: int = params_dict['embedding_size']
    vocab_size: int = params_dict['vocab_size']
    lstm_units: int = params_dict['lstm_units']
    input_length: int = params_dict['maxlen']

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_size, input_length = input_length))
    model.add(tf.keras.layers.SpatialDropout1D(0.4))
    model.add(tf.keras.layers.LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy', 'categorical_crossentropy'])
    
    print(model.summary())
    tf.keras.utils.plot_model(model, 'model_graph.png', show_shapes=True)

    return model