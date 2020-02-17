from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import pandas as pd
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

def _getcallbacks(model_params) -> list:

    train_result_path: str = model_params['result_base_dir']
    callbacks_list = [            
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                min_delta=0,
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
                patience=5,
                monitor='val_loss',
                factor=0.75,
                verbose=1,
                min_lr=5e-6)
        ]
    return callbacks_list
    # return list()


def _build_model(model_params: dict):

    # ----------------------------# 
    # Model's Params              #
    # ----------------------------#

    # Equal to timesteps
    sequence_len: int = model_params['maxlen']

    # Mapping a vocab-size to embbedding-size
    vocab_size: int = model_params['vocab_size']
    embedding_size: int = model_params['embedding_size']

    # RNN units
    rnns_units: list = model_params['rnns_units']
    

    conv_filters: list = model_params['conv_filters']
    conv_kernel_size: list = model_params['conv_kernel_size']
    conv_activations: list = model_params['conv_activations']

    # Regularization factors
    seeds: list = model_params['seeds']
    l1: list = model_params['l1']
    l2: list = model_params['l2']
    dropouts_rates: list = model_params['droputs_rates']
    recurrent_dropouts_rates: list = model_params['recurrent_dropouts_rates']

    # ----------------------------# 
    # Build model                 #
    # ----------------------------#

    inputs = tf.keras.Input(shape=(sequence_len, vocab_size))
    x = tf.keras.layers.Masking(
        mask_value=[1.0, 0.0, 0.0, 0.0, 0.0],
        name="masking_layer")(inputs)

    # *** Convolutional 1D block of layers.
    x = tf.keras.layers.Conv1D(
        filters=conv_filters[0],
        kernel_size=conv_kernel_size[0],
        activation=conv_activations[0],
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[0]),
        bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[0]),
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1[0], l2[0]),
    )(inputs)
    # x = tf.keras.layers.AveragePooling1D()(x)
    x = tf.keras.layers.MaxPool1D()(x)
    # x = tf.keras.layers.Dropout(
    #     rate=dropouts_rates[0],
    #     seed=seeds[0]
    # )(x)

    # *** Block made from stack of LSTM layers
    x = tf.keras.layers.LSTM(
        rnns_units[0],
        dropout=recurrent_dropouts_rates[0],
        # recurrent_dropout=recurrent_dropouts_rates[0],
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[1]),
        recurrent_initializer=tf.keras.initializers.orthogonal(seed=seeds[1]),
        # kernel_regularizer=tf.keras.regularizers.l1(l1[1]),
        # kernel_regularizer=tf.keras.regularizers.l2(l2[1]),
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1[1], l2[1]),
    )(x)
    x = tf.keras.layers.Dropout(
        rate=dropouts_rates[1],
        seed=seeds[0]
    )(x)
    x = tf.keras.layers.Flatten()(x)

    # *** Last nn block, classification block by means of dense layers.
    x = tf.keras.layers.Dense(
        units=64,
        activation='relu',
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[2]),
        bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[2]),
        # kernel_regularizer=tf.keras.regularizers.l1(l1[2]),
        # kernel_regularizer=tf.keras.regularizers.l2(l2[2]),
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1[2], l2[2]),
    )(x)
    x = tf.keras.layers.Dropout(
        rate=dropouts_rates[2],
        seed=seeds[0]
    )(x)
    outputs = tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[3]),
        bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[3]),
        # kernel_regularizer=tf.keras.regularizers.l1(l1[3]),
        # kernel_regularizer=tf.keras.regularizers.l2(l2[3]),
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1[3], l2[3]),
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='dna_genes_fusions_model')

    return model


def get_compiled_model(model_params: dict, program_params: dict):
    
    callbacks = _getcallbacks(model_params)
    model = _build_model(model_params)

    optimizer_name: str = model_params['optimizer']
    clip_norm: float = model_params['clip_norm']
    lr: float = model_params['lr']

    optimizer = _get_optimizer(
        optimizer_name=optimizer_name.lower(),
        lr=lr,
        clipnorm=clip_norm)

    model.compile(loss='binary_crossentropy',
        # optimizer=optimizer_name.lower(),
        optimizer=optimizer,
        metrics=['accuracy', 'binary_crossentropy', f1_m, precision_m, recall_m])

    return model, callbacks