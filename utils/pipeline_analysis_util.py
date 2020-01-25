import argparse
import datetime
import logging
import os
import sys
import time

from pprint import pprint

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

from utils.load_dataset_util import load_dataset
from utils.setup_analysis_environment_util import setup_analysis_environment
from utils.preprocess_dataset_util import preprocess_data

from utils.train_util import _holdout
from utils.train_util import _train, _test

from models.ModelFactory import ModelFactory

import numpy as np

# =============================================================================================== #
# Utils Functions                                                                                 #
# =============================================================================================== #

def _log_info_message(message: str, logger:  logging.Logger, skip_message: bool = False) -> None:
    """
    Params:
    -------
        :message: str,
        :logger: logging.Logger,
        :skip_message: bool, default = False
    """
    if logger is None:
        if skip_message is True: return
        print(message)
    else:
        logger.info(message)
    pass

def _test_dataset(x_train, y_train, x_val, y_val) -> None:
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_VAL = 64
    BUFFER_SIZE = 100
    embedding_dim = 256

    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    # dataset_list: list = list(dataset)
    # for ii, element in enumerate(dataset_list):
    for ii, element in enumerate(dataset.as_numpy_iterator()):
        if ii == 5: break
        print(element)

    # ----> Shuffle and batch
    # Prepare the training dataset
    train_dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE_TRAIN)

    # Prepare the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE_VAL)
    pass

# =============================================================================================== #
# Load Datasets - Functions                                                                       #
# =============================================================================================== #

def _pipeline_load_data(conf_load_dict, main_logger) -> object:

    _log_info_message(f"----> Dataset Load.", main_logger)

    data = load_dataset(conf_load_dict, main_logger=main_logger)

    _log_info_message(f" [*] Dataset Load: Done.", main_logger, skip_message=True)

    return data

# =============================================================================================== #
# Preprocess Datasets - Functions                                                                 #
# =============================================================================================== #

def _pipeline_preprocess_data(data, conf_preprocess_dict, main_logger):

   _log_info_message(f"----> Preprocess Data.", main_logger)

   x_train, y_train, x_val, y_val, x_test, y_test, tokenizer = \
        preprocess_data(data, conf_preprocess_dict, main_logger=main_logger)

   _log_info_message(f" [*] Preprocess Data: Done.", main_logger, skip_message=True)

   return x_train, y_train, x_val, y_val, x_test, y_test, tokenizer

# =============================================================================================== #
# Train Datasets - Functions                                                                      #
# =============================================================================================== #

def _get_callbacks_list(history_filename: str) -> list:
    callbacks_list: list = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=2,
                        restore_best_weights=True
                    ),
                    keras.callbacks.ModelCheckpoint(
                        filepath='my_model.h5',
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=0
                    ),
                    keras.callbacks.CSVLogger(history_filename),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=5,
                        monitor='val_loss',
                        factor=0.75,
                        verbose=1,
                        min_lr=5e-6)
    ]
    return callbacks_list



def _pipeline_train(x_train, y_train, x_val, y_val, conf_load_dict, cmd_line_params,
                    network_params, meta_info_project_dict, tokenizer, main_logger):

    model = None
    _log_info_message(f"----> Perform Analysis...", main_logger)

    _log_info_message(network_params, main_logger)
    
    if cmd_line_params.validation is True:
        model, epochs_trained = _holdout(
            x_train,
            y_train,
            x_val,
            y_val,
            conf_load_dict,
            cmd_line_params,
            network_params,
            meta_info_project_dict,
            tokenizer,
            main_logger)
        
        # calculate number of steps for early stopping in training
        steps = epochs_trained * np.ceil(x_train.shape[0] / network_params['batch_size'])
        _log_info_message("trained for {} steps".format(steps), main_logger) 
    if cmd_line_params.train is True:
        # we take steps from early stopping the holdout validation otherwise must be specified from command line 
        if ('steps' not in locals()):
            steps = cmd_line_params.steps 
        if ('epochs_trained' not in locals()):
            epochs_trained = cmd_line_params.early_stopping_epoch
            print(epochs_trained)
       
        # adding bin 4 to training set and pass the shape of len of bins 1,2,3 in order to 
        # train the same number of steps before overfitting in holdout
        model = _train(
            np.concatenate((x_train, x_val), axis=0),
            np.concatenate((y_train, y_val), axis=0),
            x_train.shape[0],
            conf_load_dict,
            cmd_line_params,
            network_params,
            meta_info_project_dict,
            tokenizer,
            main_logger,
            epochs_trained=epochs_trained
        )
    
    _log_info_message(f" [*] Perform Analysis: Done.", main_logger, skip_message=True)
    
    return model

def _pipeline_test(model, x_test, y_test, conf_load_dict, cmd_line_params,
                   network_params, meta_info_project_dict, main_logger):
    
    if cmd_line_params.test is True:
        if ('model' not in locals() or model is None):
            if (cmd_line_params.pretrained_model is None):
                raise ValueError("In order to perform test a pretrained model " \
                                "must be specified")
            # Todo check whether the model actually exists
            model = ModelFactory.getModelByName(cmd_line_params.load_network, network_params)
            model.build(main_logger)
        
        _test(
            model,
            x_test,
            y_test,
            conf_load_dict,
            cmd_line_params,
            network_params,
            meta_info_project_dict,
            main_logger,
        )
# =============================================================================================== #
# Run pipeline on Datasets - Function                                                             #
# =============================================================================================== #

def run_pipeline(conf_load_dict: dict, conf_preprocess_dict: dict, cmd_line_params, network_params: dict, meta_info_project_dict: dict, main_logger: logging.Logger = None) -> None:
    """Run pipeline."""
    
    pprint(conf_load_dict)

    pprint(conf_preprocess_dict)
    
    # Fetch Data.
    data = _pipeline_load_data(conf_load_dict, main_logger=main_logger)

    # Preprocessing Data.
    x_train, y_train, x_val, y_val, x_test, y_test, tokenizer = \
        _pipeline_preprocess_data(data, conf_preprocess_dict, main_logger=main_logger)
    # return

    # Print for debugging Data.
    # _test_dataset(x_train, y_train, x_val, y_val)

    # Train Data.
    model = _pipeline_train(
        x_train,
        y_train,
        x_val,
        y_val,
        conf_load_dict,
        cmd_line_params,
        network_params,
        meta_info_project_dict,
        tokenizer,
        main_logger)
    
    _pipeline_test(
        model,
        x_test,
        y_test,
        conf_load_dict,
        cmd_line_params,
        network_params,
        meta_info_project_dict,
        main_logger
    )
    pass
