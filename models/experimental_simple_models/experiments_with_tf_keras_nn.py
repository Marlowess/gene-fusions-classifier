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

def get_compiled_model_v1():
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