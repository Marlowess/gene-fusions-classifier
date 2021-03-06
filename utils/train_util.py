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
from utils.plot_functions import plot_loss, plot_accuracy, plot_roc_curve, plot_precision_recall_curve, plot_confidence_graph

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


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
            print("\nepoch finished")

def _log_info_message(message: str, logger:  logging.Logger, skip_message: bool = False) -> None:
    """
    It writes any log either on a logger or on stdout 

    Params:
    -------
        :message: str.\n
        :logger: logging.Logger.\n
        :skip_message: bool, default = False.\n
    """

    if logger is None:
        if skip_message is True: return
        print(message)
    else:
        logger.info(message)
    pass

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
    It instantiates a new model and train it for the number of epoch specified in the model 
    configuration file in order to subsequently validate
    
    Params:
    -------
    :x_train: training feature matrix X.\n
    :y_train: training class labels.\n
    :x_val: validation feature matrix X.\n
    :y_val: validation class labels.\n
    :conf_load_dict: dictionary containing information about how input data have been loaded from source directory of data.\n
    :cmd_line_params: command line parameters.\n
    :network_params: model configuration read from file.\n
    :meta_info_project_dict: dictionary containing information about how project layout has been setted for collecting output data.\n
    :tokenizer: tokenizer for translating text to sequences.\n
    :logger: object used for loggin info, it might be None, if None it is just ignored by functions that are dealing with it.\n
    :message: str object used to either log or display on stdout a message about the fact train-phase is running.\n

    Returns:
    --------
        :object: representing the trained model by means of holdout procedure.
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

    # Get Model from ModelFactory Static class.
    network_model_name: str = cmd_line_params.load_network
    model = ModelFactory.getModelByName(network_model_name, network_params)

    # Build model.
    _log_info_message(f"> build model (holdout).", logger)
    
    # It compiles the model and print its summary (architecture's structure)
    model.build(logger)

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
    results_dict: dict = model.evaluate(x_val, y_val)        
    res_string = ", ".join(f'{k}:{v:.5f}' for k,v in results_dict.items())
    _log_info_message("{}".format(res_string), logger)
    _log_info_message(f" [*] {message} Done.", logger)
    
    return model, trained_epochs, res_string

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
    It instantiates a new model and train it for a specified number of update steps. The training phase executes:

    - either `algorithm 7.2*` if the command line flag specified for this phase is `--early_stopping_epoch`,
    - or `algorithm 7.3**` if the command line flag specified for this phase is `--early_stopping_on_loss`.

    * Algorithm 7.2 (Ian Goodfellow, Yoshua Bengio, and Aaron Courville. 2016. Deep Learning. The MIT Press, pp. 246-250.)\n
    ** Algorithm 7.3 (Ian Goodfellow, Yoshua Bengio, and Aaron Courville. 2016. Deep Learning. The MIT Press, pp. 246-250.)

    
    Params:
    -------
    :x_train: training feature matrix X.\n
    :y_train: training class labels.\n
    :steps: number of steps of training.\n
    :conf_load_dict: dictionary containing information about how input data have been loaded from source directory of data.\n
    :cmd_line_params: command line parameters.\n
    :network_params: model configuration read from file.\n
    :meta_info_project_dict: dictionary containing information about how project layout has been setted for collecting output data.\n
    :tokenizer: tokenizer for translating text to sequences.\n
    :logger: object used for loggin info, it might be None, if None it is just ignored by functions that are dealing with it.\n
    :message: str object used to either log or display on stdout a message about the fact train-phase is running.\n

    Returns:
    --------
        :object: representing the trained model.\n
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
    # if early stopping with loss val load trained model from holdout
    if cmd_line_params.early_stopping_on_loss:
        # Algorithm 7.3
        _log_info_message("> loading holdout training weights", logger)
        # if train after validation in a single run 
        if network_params['pretrained_model'] == None:
            network_params['pretrained_model'] = os.path.join(base_dir,cmd_line_params.output_dir,
                                                               "results_holdout_validation/model_checkpoint_weights.h5")
        model = ModelFactory.getModelByName(network_model_name, network_params)
    else:
        # Algorithm 7.2
        model = ModelFactory.getModelByName(network_model_name, network_params)

    # Build model.
    _log_info_message(f"> build model", logger)
    model.build(logger)    

    # Train for the specified amount of steps.
    # _log_info_message(f"> training model for {}".format(steps), logger)

    if cmd_line_params.early_stopping_on_loss:
        # Algorithm 7.3
        early_stopping_loss = model.evaluate(x_subtrain, y_subtrain)['loss']
        history = model.fit_early_stopping_by_loss_val(x_train, y_train,
            epochs=cmd_line_params.num_epochs,
            early_stopping_loss=early_stopping_loss,
            callbacks_list=[], validation_data=validation_data
        )
    else:
        # Algorithm 7.2
        history = model.fit_generator(
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
    message: str = 'Performing test phase...',
    roc_curve: bool = False,
    pr_curve: bool = True) -> object:

    """
    Test the model passed by argument and log evaluation metrics
    
    Params:
    -------
    :x_test: test feature matrix X.\n
    :param y: test class labels.\n
    :conf_load_dict: dictionary containing information about how input data have been loaded from source directory of data.\n
    :cmd_line_params: command line parameters.\n
    :network_params: model configuration read from file.\n
    :meta_info_project_dict: dictionary containing information about how project layout has been setted for collecting output data.\n
    :logger: object used for loggin info, it might be None, if None it is just ignored by functions that are dealing with it.\n
    :message: str object used to either log or display on stdout a message about the fact test-phase is running.\n
    """     
    # Some logs recorded.
    _log_info_message(f" [*] {message}", logger)

    test_bins = conf_load_dict['test_bins']
    _log_info_message("Testing on bins: {}".format(test_bins), logger)    

    evaluation_metrics = model.evaluate(x_test, y_test)
    
    _log_info_message("Resulting metrics:", logger)
    tmp_sol: list = list()
    for (k,v) in evaluation_metrics.items():
        _log_info_message("{}: {:.5f}".format(k, v), logger)
        tmp_sol.append("{}: {:.5f}".format(k, v))
    
    res_str_test: str = ','.join([str(xi) for xi in tmp_sol])
    _log_info_message(
        res_str_test
        , logger)
    
    # plot roc curve and auc
    y_pred = model.predict(x_test)
    
    if roc_curve :
        auc_value: float = plot_roc_curve(
            y_test,
            y_pred,
            title="Roc Curve Eval",
            fig_name="roc_curve_eval",
            fig_dir=meta_info_project_dict['test_result_path'],
            savefig_flag=True,
            showfig_flag=True,
        )
        _log_info_message(f"TEST_AUC: {auc_value}", logger)

    if pr_curve:
        # Plotting the precision-recall curve here    
        avg_precision_score: float = plot_precision_recall_curve(
            y_test,
            y_pred,
            title="Precision-Recall curve",
            fig_name="precison_recall_eval",
            fig_dir=meta_info_project_dict['test_result_path'],
            savefig_flag=True,
            showfig_flag=True,
        )
        _log_info_message(f"TEST_AVG_PREC_SCORE: {avg_precision_score}", logger)

    # plot conf matrix    
    y_pred_classes = model.predict_classes(x_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_classes).ravel()

    conf_matrix_elem = "TN,FP,FN,TP".split(",")
    conf_matrix_elem_pairs: dict = dict(zip(conf_matrix_elem, [tn, fp, fn, tp]))

    _log_info_message(
        "CONFUSION MATRIX\n" + \
        ','.join([f"{k} {v}" for k,v in conf_matrix_elem_pairs.items()])
        ,logger
    )

    test_dir: str = meta_info_project_dict['test_result_path']
    _calculate_confidence(y_test, y_pred, test_dir, logger)
    pass

def _calculate_confidence(y_test, y_pred, test_dir, logger):
    """
    It writes any log either on a logger or on stdout 

    Params:
    -------
        :y_test: Actual labels.\n
        :y_pred: Predicted labels.\n
        :test_dir: Folder where to put the images.\n
        :logger: Logger object.\n
    """


    _log_info_message("\n\nMetrics for element predicted with strong confidence (0.8 threshold)", logger)
    conf_y_prediction = y_pred[(y_pred < 0.2) | (y_pred >= 0.8)]
    conf_y_test = y_test[(y_pred < 0.2) | (y_pred >= 0.8)] 

    threshold_y_pred = np.rint(conf_y_prediction)
    _log_info_message("samples classified: {:.5f}".format(len(conf_y_prediction)/len(y_test)), logger)
    _log_info_message("accuracy {:.5f}".format(accuracy_score(conf_y_test, threshold_y_pred)), logger)
    _log_info_message("precision {:.5f}".format(precision_score(conf_y_test, threshold_y_pred)), logger)
    _log_info_message("recall {:.5f}".format(recall_score(conf_y_test, threshold_y_pred)), logger)
    _log_info_message("f1-score {:.5f}".format(f1_score(conf_y_test, threshold_y_pred)), logger)    
    
    plot_confidence_graph(
        pd.DataFrame(data=list(zip(y_test, y_pred)), columns=['Label', 'Prob']),
        fig_dir=test_dir,
        fig_name="scores_by_threshold",
        title="",
        savefig_flag=True,
        showfig_flag=False
        )
    pass
