import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import pandas as pd
from models.metrics import f1_m, precision_m, recall_m

from tensorflow.keras.callbacks import Callback

class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='acc', baseline=0.9, mode='max'):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline
        self.mode = mode

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        monitored_measure = logs.get(self.monitor)
        if monitored_measure is not None:
            if self.mode == 'max':
                if monitored_measure >= self.baseline:
                    print('Epoch %d: Reached baseline, terminating training' % (epoch))
                    self.model.stop_training = True
            elif self.mode == 'min':
                if monitored_measure <= self.baseline:
                    print('Epoch %d: Reached baseline, terminating training' % (epoch))
                    self.model.stop_training = True

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
                patience=5,
                monitor='val_loss',
                factor=0.05,
                verbose=1,
                min_lr=5e-6),
            tf.keras.callbacks.TerminateOnNaN(),
            TerminateOnBaseline(monitor="loss", baseline=0.2, mode='min'),
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

    # Conv1D
    filters = model_params['filters']
    
    # LSTM units per layer
    lstms_units: list = model_params['lstms_units']

    # Regularization factors
    seeds: list = model_params['seeds']
    l1: list = model_params['l2']
    l2: list = model_params['l2']
    dropouts_rates: list = model_params['droputs_rates']
    lstms_dropouts: list = model_params['lstms_dropouts']

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
            embeddings_regularizer=tf.keras.regularizers.L1L2(l1[0], l2[0]),
            # name=f'embedding_layer_in{vocab_size}_out{embedding_size}'))
    ))

    # Conv1D layer
    # model.add(tf.keras.layers.Conv1D(
    #     kernel_size=model_params["kernel_size"],
    #     activation=model_params["activation"],
    #     strides=model_params["strides"],
    #     kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[1]),
    #     # kernel_regularizer=tf.keras.regularizers.l2(l2[0]),
    #     filters=filters,
    # ))

    # Regularization MaxPooling
    # model.add(tf.keras.layers.MaxPool1D())
    # model.add(tf.keras.layers.AveragePooling1D())
    # model.add(tf.keras.layers.BatchNormalization())
    
    # First LSTM layer
    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
            units=lstms_units[0],
            return_sequences=model_params["return_sequences"][0],
            # unit_forget_bias=True,
            dropout=lstms_dropouts[0],
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[2]),
            bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[1]),
            kernel_regularizer=tf.keras.regularizers.L1L2(l1[1], l2[1]),
            # name=f'lstm_1_units{lstms_units[0]}')
            )
        )
    )
    
    # Second LSTM layer
    # model.add(tf.keras.layers.LSTM(
    #     units=lstms_units[1],
    #     return_sequences=model_params["return_sequences"][1],
    #     # unit_forget_bias=True,
    #     dropout=lstms_dropouts[0],
    #     kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[2]),
    #     # bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[2]),
    #     kernel_regularizer=tf.keras.regularizers.l2(l2[1]),
    #     name=f'lstm_2_units{lstms_units[1]}'))
    
    # Dropout after the lstm layer
    model.add(tf.keras.layers.Dropout(dropouts_rates[0], seed=seeds[0]))

    # Fully connected (prediction) layer
    model.add(tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[3]),
        bias_initializer=tf.keras.initializers.glorot_uniform(seed=seeds[3]),
        kernel_regularizer=tf.keras.regularizers.L1L2(l1[2], l2[2]),
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