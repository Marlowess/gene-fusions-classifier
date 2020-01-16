import argparse
import datetime
import logging
import os
import sys
import time

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

from utils_dir.load_dataset_util import load_dataset
from utils_dir.setup_analysis_environment_util import setup_analysis_environment
from utils_dir.preprocess_dataset_util import preprocess_data

from ModelFactory import ModelFactory

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
                        patience=10,
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

def _holdout(x_train, y_train, x_val, y_val, conf_load_dict: dict, cmd_line_params, network_params: dict, meta_info_project_dict: dict, tokenizer: Tokenizer, logger: logging.Logger, message: str = 'Performing first phase (holdout)...') -> None:
    
    # Some logs recorded.
    _log_info_message(f" [*] {message}", logger)

    train_bins, val_bins = conf_load_dict['train_bins'], conf_load_dict['val_bins']
    _log_info_message("Training on bins: {}, validation on {}".format(train_bins, val_bins), logger)

    vocabulary_len: int = len(tokenizer.index_word) + 1
    network_params['vocabulary_len'] = vocabulary_len

    # Get Callbacks.
    base_dir: str = meta_info_project_dict['base_dir']
    history_filename: str = os.path.join(base_dir, 'history.csv')
    callbacks_list = _get_callbacks_list(history_filename)

    # Get Model from ModelFactory Static class.
    network_model_name: str = cmd_line_params.load_network
    model = ModelFactory.getModelByName(network_model_name, network_params)

    # Build model.
    _log_info_message(f"> build model (holdout).", logger)
    model.build()
    model.plot_model()
    
    sys.exit(0)

    # Train model.
    _log_info_message(f"> train model (holdout)...", logger)

    model.fit(
        x_train=x_train,
        y_train=y_train,
        callback_list=callbacks_list,
        validation_data=(x_val, y_val))
    _log_info_message(f"> train model (holdout): Done.", logger)

    # Eval model.
    _log_info_message(f"> eval model (holdout).", logger)
    scores = model.model.evaluate(x_val, y_val)
    _log_info_message("{}: {}".format(model.model.metrics_names[1], scores[1] * 100), logger)

    _log_info_message(f" [*] {message} Done.", logger)
    pass

def _pipeline_train(x_train, y_train, x_val, y_val, conf_load_dict, cmd_line_params, network_params, meta_info_project_dict, tokenizer, main_logger):

    _log_info_message(f"----> Perform Analysis...", main_logger)

    if cmd_line_params.validation is True:
        _holdout(x_train,
            y_train,
            x_val,
            y_val,
            conf_load_dict,
            cmd_line_params,
            network_params,
            meta_info_project_dict,
            tokenizer,
            main_logger)

    _log_info_message(f" [*] Perform Analysis: Done.", main_logger, skip_message=True)

    pass

# =============================================================================================== #
# Run pipeline on Datasets - Function                                                             #
# =============================================================================================== #

def run_pipeline(conf_load_dict: dict, conf_preprocess_dict: dict, cmd_line_params, network_params: dict, meta_info_project_dict: dict, main_logger: logging.Logger = None) -> None:
    """Run pipeline."""
    
    # Fetch Data.
    data = _pipeline_load_data(conf_load_dict, main_logger=main_logger)

    # Preprocessing Data.
    x_train, y_train, x_val, y_val, x_test, y_test, tokenizer = \
        _pipeline_preprocess_data(data, conf_preprocess_dict, main_logger=main_logger)

    # Print for debugging Data.
    _test_dataset(x_train, y_train, x_val, y_val)

    # Train Data.
    _pipeline_train(
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
    pass