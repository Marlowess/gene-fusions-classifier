import argparse
import datetime
import logging
import os
import sys
import time
import numpy as np

from ModelFactory import ModelFactory
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

def gen(X, y, batch_size=32, shuffle=True, verbose=0, seed=None):
    """
    Convert dataset in generator for training model a specific number of step
    
    Params:
    -------
    :param X: feature matrix X on which we want to extract batches
    :param y: class label on which we want to extract batches
    :shuffle: shuffle dateset after each epoch
    :verbose: verbose level
    :seed: seed for dataset shuffling 
    """

    N = X.shape[0]
    idxs = np.arange(N)
    while(True):
        if shuffle:
            if (seed is not None):
                np.random.seed(seed)
            np.random.shuffle(idxs)
            X = X[idxs]
            y = y[idxs]
        for i in range(0, N, batch_size):
            yield (X[i:i+batch_size], y[i:i+batch_size])
        
        if (verbose != 0):
            print("epoch finished")

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

def _holdout(
    x_train,
    y_train,
    x_val,
    y_val,
    conf_load_dict: dict,
    cmd_line_params,
    network_params: dict,
    meta_info_project_dict: dict,
    tokenizer: Tokenizer,
    logger: logging.Logger,
    message: str = 'Performing first phase (holdout)...') -> object:
    
    """
    Instantiate a new model and train it for the number of epoch specified in the model 
    configuration file in order to subsequently validate
    
    Params:
    -------
    :x_train: training feature matrix X
    :y_train: training class labels
    :x_val: validation feature matrix X
    :y_val: validation class labels
    :conf_load_dict:
    :cmd_line_params: command line parameters
    :network_params: model configuration read from file
    :meta_info_project_dict:
    :tokenizer: tokenizer for translating text to sequences
    :logger:
    :message:
    """
     
    # Some logs recorded.
    _log_info_message(f" [*] {message}", logger)

    train_bins, val_bins = conf_load_dict['train_bins'], conf_load_dict['val_bins']
    _log_info_message("Training on bins: {}, validation on {}".format(train_bins, val_bins), logger)

    vocabulary_len: int = len(tokenizer.index_word) + 1
    network_params['vocabulary_len'] = vocabulary_len

    # Get Callbacks.
    base_dir: str = meta_info_project_dict['base_dir']
    history_filename: str = os.path.join(base_dir, 'history.csv')
    network_params['result_base_dir'] = meta_info_project_dict['val_result_path']    

    # TODO: callbacks are defined for each model --> remove them from all dictionaries
    callbacks_list = _get_callbacks_list(history_filename)

    # Get Model from ModelFactory Static class.
    network_model_name: str = cmd_line_params.load_network
    model = ModelFactory.getModelByName(network_model_name, network_params)

    # Build model.
    _log_info_message(f"> build model (holdout).", logger)
    
    # It compiles the model and print its summary (architecture's structure)
    model.build(logger)

    # It plots on a file the model's structure
    model.plot_model()

    # Train model.
    _log_info_message(f"> train model (holdout)...", logger)
    
    history, trained_epochs = model.fit(
        X_tr=x_train,
        y_tr=y_train,
        epochs=cmd_line_params.num_epochs,
        callbacks_list=callbacks_list,
        validation_data=(x_val, y_val)
        )
    _log_info_message(f"> train model (holdout): Done.", logger)
   
    # log number of epochs trained 
    _log_info_message("Trained for {} epochs".format(trained_epochs), logger)
    
    # Eval model.
    _log_info_message(f"> eval model (holdout).", logger)

    # scores contains [loss, accuracy, f1_score, precision, recall]
    results_dict = model.evaluate(x_val, y_val)
    res_string = ", ".join(f'{k}:{v}' for k,v in results_dict.items())
    _log_info_message("{}".format(results_str), logger)
    _log_info_message(f" [*] {message} Done.", logger)
    
    return model, trained_epochs

def _train(
    x_train,
    y_train,
    steps,
    conf_load_dict: dict,
    cmd_line_params,
    network_params: dict,
    meta_info_project_dict: dict,
    tokenizer: Tokenizer,
    logger: logging.Logger,
    message: str = 'Performing training phase...') -> object:
    
    """
    Instantiate a new model and train it for a specified number of update steps
    
    Params:
    -------
    :x_train: training feature matrix X
    :y_train: training class labels
    :steps: number of steps of training
    :conf_load_dict:
    :cmd_line_params: command line parameters
    :network_params: model configuration read from file
    :meta_info_project_dict:
    :tokenizer: tokenizer for translating text to sequences
    :logger:
    :message:
    """
    # Some logs recorded.
    _log_info_message(f" [*] {message}", logger)

    train_bins = conf_load_dict['train_bins']
    _log_info_message("Training on bins: {}".format(train_bins), logger)

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
    _log_info_message(f"> build model", logger)
    summary_model: str = model.build(logger)
    _log_info_message(f"\n{summary_model}", logger)
    model.plot_model()

    # Train for the specified amount of steps.
    # _log_info_message(f"> training model for {}".format(steps), logger)

    history = model.fit_generator(
        generator=gen(x_train, y_train),
        steps=steps,
        callbacks_list=callbacks_list,
        )
    _log_info_message(f"> train model: Done.", logger)
    _log_info_message(f" [*] {message} Done.", logger)
    
    return model

def _test(
    model,
    x_test,
    y_test,
    conf_load_dict: dict,
    cmd_line_params,
    network_params: dict,
    meta_info_project_dict: dict,
    logger: logging.Logger,
    message: str = 'Performing test phase...') -> object:
    """
    Test the model passed by argument and log evaluation metrics
    
    Params:
    -------
    :x_test: test feature matrix X
    :param y: test class labels
    :conf_load_dict:
    :cmd_line_params: command line parameters
    :network_params: model configuration read from file
    :meta_info_project_dict:
    :logger:
    :message:
    """ 
    # Some logs recorded.
    _log_info_message(f" [*] {message}", logger)

    test_bins = conf_load_dict['test_bins']
    _log_info_message("Testing on bins: {}".format(test_bins), logger)

    evaluation_metrics = model.evaluate(x_test, y_test)
    
    _log_info_message("Resulting metrics:", logger)
    for (k,v) in evaluation_metrics.items():
        _log_info_message("{}: {}".format(k, v), logger)
        