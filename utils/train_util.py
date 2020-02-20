import argparse
import datetime
import logging
import os
import sys
import time
import numpy as np
import pickle

from models.ModelFactory import ModelFactory
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from utils.plot_functions import plot_loss, plot_accuracy, plot_roc_curve

from sklearn.metrics import confusion_matrix

def gen(X, y, batch_size, shuffle=True, verbose=0, seed=None):
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

def _experimental_train(
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
    message: str = 'Performing Experimental Training...'
    ):

    _log_info_message(f" [*] {message}", logger)

    network_model_name: str = cmd_line_params.load_network
    model = ModelFactory.getModelByName(network_model_name, network_params)

    # epochs: int = network_params['epochs']
    batch_size: int = network_params['batch_size']

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(x_val,y_val),
        epochs=5,
        verbose=1)
    return model


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

    base_dir: str = meta_info_project_dict['base_dir']
    results_dir = meta_info_project_dict['val_result_path']
    history_filename: str = os.path.join(base_dir, 'history.csv')
    network_params['result_base_dir'] = results_dir   

    # TODO: callbacks are defined for each model --> remove them from all dictionaries
    # callbacks_list = _get_callbacks_list(history_filename)

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
        callbacks_list=[],
        validation_data=(x_val, y_val)
        )
    _log_info_message(f"> train model (holdout): Done.", logger)
   
    model.save_weights()
    # log number of epochs trained 
    _log_info_message("Trained for {} epochs".format(trained_epochs), logger)
    
    # Eval model.
    _log_info_message(f"> eval model (holdout).", logger)

    # plot graph of loss and accuracy
    plot_loss(history, results_dir, "Training and validation losses", "loss",
              savefig_flag=True, showfig_flag=False)
    plot_accuracy(history, results_dir, "Training and validation accuracies", "accuracy",
                  savefig_flag=True, showfig_flag=False)
    # serialize history
    with open(os.path.join(results_dir, "history"), 'wb') as history_pickle:
        pickle.dump(history.history, history_pickle)
    
    # scores contains [loss, accuracy, f1_score, precision, recall]
    results_dict = model.evaluate(x_val, y_val)    
    res_string = ", ".join(f'{k}:{v}' for k,v in results_dict.items())
    _log_info_message("{}".format(res_string), logger)
    _log_info_message(f" [*] {message} Done.", logger)
    
    return model, trained_epochs

def _train(
    subtrain,
    validation_data,
    x_subtrain_size,
    conf_load_dict: dict,
    cmd_line_params,
    network_params: dict,
    meta_info_project_dict: dict,
    tokenizer: Tokenizer,
    logger: logging.Logger,
    epochs_trained=None,
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

    train_bins = conf_load_dict['train_bins'] + conf_load_dict['val_bins']
    _log_info_message("Training on bins: {}".format(train_bins), logger)

    vocabulary_len: int = len(tokenizer.index_word) + 1
    network_params['vocabulary_len'] = vocabulary_len

    # Get Callbacks.
    base_dir: str = meta_info_project_dict['base_dir']
    results_dir = meta_info_project_dict['train_result_path']
    # history_filename: str = os.path.join(base_dir, 'history.csv')
    network_params['result_base_dir'] = results_dir
    
    # adding bin 5 to training data
    x_subtrain, y_subtrain = subtrain
    x_validation, y_validation = validation_data
    x_train = np.concatenate((x_subtrain, x_validation), axis=0)
    y_train = np.concatenate((y_subtrain, y_validation), axis=0)

    # Get Model from ModelFactory Static class.
    network_model_name: str = cmd_line_params.load_network
    if network_model_name == 'WrappedRawModel':
        model = ModelFactory.getRawModelByName(network_params, meta_info_project_dict)
    # if early stopping with loss val load trained model from holdout
    elif cmd_line_params.early_stopping_on_loss:
        _log_info_message("> loading holdout training weights", logger)
        # if train after validation in a single run 
        if network_params['pretrained_model'] == None:
            network_params['pretrained_model'] = os.path.join(base_dir,cmd_line_params.output_dir,
                                                               "results_holdout_validation/my_model_weights.h5")
        model = ModelFactory.getModelByName(network_model_name, network_params)
    else:
        model = ModelFactory.getModelByName(network_model_name, network_params)

    # Build model.
    _log_info_message(f"> build model", logger)
    model.build(logger)
    model.plot_model()

    # Train for the specified amount of steps.
    # _log_info_message(f"> training model for {}".format(steps), logger)

    if network_model_name == 'WrappedRawModel':
        history = model.train(x_train, y_train,
            epochs=cmd_line_params.num_epochs,
            batch_size=cmd_line_params.batch_size,
            validation_data=validation_data,
        )
    elif cmd_line_params.early_stopping_on_loss:
        early_stopping_loss = model.evaluate(x_subtrain, y_subtrain)['loss']
        history = model.fit_early_stopping_by_loss_val(x_train, y_train,
            epochs=cmd_line_params.num_epochs,
            early_stopping_loss=early_stopping_loss,
            callbacks_list=[], validation_data=validation_data
        )
    else:
        history = model.fit_generator2(
            generator=gen(x_train, y_train, batch_size=network_params['batch_size'], verbose=1),
            steps_per_epoch=np.floor(x_subtrain_size/network_params['batch_size']),
            epochs=epochs_trained,
            callbacks_list=[]
        )
    
    model.save_weights()
    
    # plot graph of loss and accuracy
    plot_loss(history, results_dir, "Training loss", "loss", savefig_flag=True, showfig_flag=False)
    # plot_accuracy(history, results_dir, "Training and validation accuracies", "accuracy", save_fig_flag=True)
    # serialize history
    with open(os.path.join(results_dir, "history"), 'wb') as history_pickle:
        pickle.dump(history.history, history_pickle)
        
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
        _log_info_message("{}: {:.2f}".format(k, v), logger)

    # plot roc curve and auc
    y_pred = model.predict(x_test)
    auc_value: float = plot_roc_curve(
        y_test,
        y_pred,
        title="Roc Curve Eval",
        fig_name="roc_curve_eval",
        fig_dir=meta_info_project_dict['test_result_path'],
        savefig_flag=True,
    )
    _log_info_message(f"TEST_AUC: {auc_value}", logger)

    # plot conf matrix
    target_names: list = ['Onco', 'Non-Onco']
    y_pred_classes = model.predict_classes(x_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_classes).ravel()

    conf_matrix_elem = "TN,FP,FN,TP".split(",")
    conf_matrix_elem_pairs: dict = dict(zip(conf_matrix_elem, [tn, fp, fn, tp]))

    _log_info_message(
        "CONFUSION MATRIX\n" + \
        '\n'.join([f"{k} {v}" for k,v in conf_matrix_elem_pairs.items()])
        ,logger
    )
    pass