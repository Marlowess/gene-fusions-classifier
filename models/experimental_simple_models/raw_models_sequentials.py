import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import auc as auc_test
from models.attlayer import AttentionWeightedAverage
from models.metrics import f1_m, precision_m, recall_m

def _get_optimizer(optimizer_name='adam', lr=0.001, clipnorm=1.0):
    if optimizer_name == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(lr, clipnorm=clipnorm)
    if optimizer_name == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(lr, clipnorm=clipnorm)
    elif optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr, clipnorm=clipnorm)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(lr, clipnorm=clipnorm)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr, clipnorm=clipnorm)
    elif optimizer_name == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(lr, clipnorm=clipnorm)
    else:
        raise ValueError(f'ERROR: specified optimizer name {optimizer_name} is not allowed.')
    return optimizer

def _getcallbacks(program_params) -> list:

    train_result_path: str = program_params['train_result_path']
    callbacks_list = [            
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(train_result_path, 'my_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
            keras.callbacks.CSVLogger(os.path.join(train_result_path, 'history.csv')),
            keras.callbacks.ReduceLROnPlateau(
                patience=10,
                monitor='val_loss',
                factor=0.75,
                verbose=1,
                min_lr=5e-6)
        ]
    return callbacks_list

def _build_model(model_params: dict):

    # ----------------------------# 
    # Model's Params              #
    # ----------------------------#

    # Equal to timesteps
    sequence_len: int = model_params['maxlen']

    # Mapping a vocab-size to embbedding-size
    vocab_size: int = model_params['vocab_size']
    embedding_size: int = model_params['embedding_size']
    
    # LSTM units per layer
    lstms_units: list = model_params['lstms_units']

    # Regularization factors
    seeds: list = model_params['seeds']
    l2: list = model_params['l2']
    droputs_rates: list = model_params['droputs_rates']

    # ----------------------------# 
    # Build model                 #
    # ----------------------------#
    model = keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(sequence_len,)))
    model.add(tf.keras.layers.Masking(mask_value=0, name="masking_layer"))

    # Embedding Input
    model.add(tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size, # mask_zero=model_params['mask_zero'],
            embeddings_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[0]),
            embeddings_regularizer=tf.keras.regularizers.l2(l2[0]),
            name=f'embedding_layer_in{vocab_size}_out{embedding_size}'))
    
    # First LSTM layer
    model.add(tf.keras.layers.LSTM(
        units=lstms_units[0],
        return_sequences=True,
        unit_forget_bias=True,
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[1]),
        bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[1]),
        kernel_regularizer=tf.keras.regularizers.l2(l2[1]),
        name=f'lstm_1_units{lstms_units[0]}'))
    
    # Second LSTM layer
    model.add(tf.keras.layers.LSTM(
        units=lstms_units[1],
        return_sequences=False,
        unit_forget_bias=True,
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[2]),
        bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[2]),
        kernel_regularizer=tf.keras.regularizers.l2(l2[1]),
        name=f'lstm_2_units{lstms_units[0]}'))
    
    # Dropout after the lstm layer
    model.add(tf.keras.layers.Dropout(droputs_rates[0], seed=seeds[0]))

    # Fully connected (prediction) layer
    model.add(tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[3]),
        bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[3]),
        kernel_regularizer=tf.keras.regularizers.l2(l2[2]),
        name=f'dense_activation_sigmoid'))
    return model

def get_compiled_model(model_params: dict, program_params: dict):
    
    callbacks = _getcallbacks(program_params)
    model = _build_model(model_params)

    optimizer_name: str = model_params['optimizer']
    clip_norm: float = model_params['clip_norm']
    lr: float = model_params['lr']

    optimizer = _get_optimizer(
        optimizer_name=optimizer_name.lower(),
        lr=lr,
        clipnorm=clip_norm)

    model.compile(loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'binary_crossentropy', f1_m, precision_m, recall_m])

    return model, callbacks