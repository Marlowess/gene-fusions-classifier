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

def _log_info_message(message: str, logger:  logging.Logger, skip_message: bool = False) -> None:
    """
    Log message on stdout and on log file
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

def _pipeline_load_data(conf_load_dict, main_logger) -> object:
    """
    It is used to load data from a given source directory, and this data
    will be used for either train, test, or both phases.

    Params:
    -------
        :conf_load_dict: dictionary object used for knowing where to load and how to load input data.\n
        :main_logger: logger object that might be None, if it is None it will be ignored by those functions dealing with it
        and information to be displayed will be displayed directly to standard output.\n
    Returns:
    --------
        :object: dictionary containing the loaded input data.
    """

    _log_info_message(f"----> Dataset Load.", main_logger)

    data = load_dataset(conf_load_dict, main_logger=main_logger)

    _log_info_message(f" [*] Dataset Load: Done.", main_logger, skip_message=True)

    return data

def _pipeline_preprocess_data(data, conf_preprocess_dict, main_logger):
    """
    It is used to preprocess data loaded previously from a given source directory, and this data
    will be used for either train, test, or both phases. The constraints about how to treat and preprocess
    data are defined within `conf_preprocess_dict` input dictionary to this function.

    Params:
    -------
        :data: object containing data to be preprocessed.\n
        :conf_preprocess_dict: dictionary object used for knowing how to preprocess input data.\n
        :main_logger: logger object that might be None, if it is None it will be ignored by those functions dealing with it
        and information to be displayed will be displayed directly to standard output.\n
    
    Returns:
    --------
        x_train, y_train, x_val, y_val, x_test, y_test, tokenizer: preprocessed data, and tokenizer used for preprocessing input data
    """

    _log_info_message(f"----> Preprocess Data.", main_logger)

    x_train, y_train, x_val, y_val, x_test, y_test, tokenizer = \
        preprocess_data(data, conf_preprocess_dict, main_logger=main_logger)

    _log_info_message(f" [*] Preprocess Data: Done.", main_logger, skip_message=True)

    return x_train, y_train, x_val, y_val, x_test, y_test, tokenizer

def _pipeline_train(x_train, y_train, x_val, y_val, conf_load_dict, cmd_line_params,
                    network_params, meta_info_project_dict, tokenizer, main_logger):

    """
    Perform holdout and/or train depending on chosen step in cmd_line_params
    
    Params:
    -------
        :x_train: training samples
        :y_train: training labels
        :x_val: validation samples
        :y_val: validation labels
        :conf_load_dict: dict, configuration for loading the dataset: it specify the bin to use, the sequence type, dataset path 
        :cmd_line_params: dictionary, command line arguments
        :network_params:, dictionary, model configuration depending on the neural network architecture used 
        :meta_info_project_dict: dictionary, it contains path for the output for each stage of the pipeline 
        :tokenizer: object containing tokenization settings (vocabulary and mappings)
        :main_logger: output logger
        
    Returns:
    -------
        :model: model with trained weights
        :res_str_holdout: summary str for holdout
    """
    
    model = None
    _log_info_message(f"----> Perform Analysis...", main_logger)

    # _log_info_message(network_params, main_logger)
    res_str_holdout = ""
    
    if cmd_line_params.validation is True:
        model, epochs_trained, res_str_holdout = _holdout(
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
        # we take epochs from early stopping the holdout validation otherwise must be specified from command line 
        if ('epochs_trained' not in locals()):
            epochs_trained = cmd_line_params.early_stopping_epoch
            print(epochs_trained)
       
        # adding bin 4 to training set and pass the shape of len of bins 1,2,3 in order to 
        # train the same number of steps before overfitting in holdout
        model = _train(
            (x_train, y_train),
            (x_val, y_val),
            x_train.shape[0],
            conf_load_dict,
            cmd_line_params,
            network_params,
            meta_info_project_dict,
            tokenizer,
            main_logger,
            epochs_trained=epochs_trained,          
        )
    
    _log_info_message(f" [*] Perform Analysis: Done.", main_logger, skip_message=True)
    
    return model, res_str_holdout

def _pipeline_test(model, x_test, y_test, conf_load_dict, cmd_line_params,
                   network_params, meta_info_project_dict, main_logger):
    
    """
    Perform holdout and/or train depending on chosen step in cmd_line_params
    
    Params:
    -------
        :model: model to test 
        :x_train: testing samples
        :y_train: testing labels        
        :conf_load_dict: dict, configuration for loading the dataset: it specify the bin to use, the sequence type, dataset path 
        :cmd_line_params: dictionary, command line arguments
        :network_params:, dictionary, model configuration depending on the neural network architecture used 
        :meta_info_project_dict: dictionary, it contains path for the output for each stage of the pipeline         
        :main_logger: output logger
    """        

    if cmd_line_params.test is True:
        if ('model' not in locals() or model is None):
            if (cmd_line_params.pretrained_model is None):
                raise ValueError("In order to perform test a pretrained model " \
                                "must be specified")
            # Todo check whether the model actually exists

            # Get Callbacks.
            base_dir: str = meta_info_project_dict['base_dir']
            results_dir = meta_info_project_dict['test_result_path']
            network_params['result_base_dir'] = results_dir
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
    return

def run_pipeline(conf_load_dict: dict, conf_preprocess_dict: dict, cmd_line_params, network_params: dict, meta_info_project_dict: dict, main_logger: logging.Logger = None) -> None:
    """
    Run holdout validation and/or training and/or test depending on the choosen pipeline by command line parameters.
    
    Params:
    -------
        :conf_load_dict: dict, configuration for loading the dataset: it specify the bin to use, the sequence type, dataset path 
        :conf_preprocess_dict: dictionary, configuration for preprocessing in which we specify padding length, encoding, and sequence type (dna, protein, kmer)
        :cmd_line_params: dictionary, command line arguments
        :network_params:, dictionary, model configuration depending on the neural network architecture used 
        :meta_info_project_dict: dictionary, it contains path for the output for each stage of the pipeline 
        :main_logger: logging.Logger, logger reference
    """
    
    pprint(conf_load_dict)

    pprint(conf_preprocess_dict)
    
    # Fetch Data.
    data = _pipeline_load_data(conf_load_dict, main_logger=main_logger)

    # Preprocessing Data.
    x_train, y_train, x_val, y_val, x_test, y_test, tokenizer = \
        _pipeline_preprocess_data(data, conf_preprocess_dict, main_logger=main_logger)

    if cmd_line_params.train or cmd_line_params.validation:
        model, res_str_holdout = _pipeline_train(
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
        _log_info_message("Holdout: " + res_str_holdout, main_logger)
    else:
        model = None
        
    if cmd_line_params.test:
        network_params['only_test'] = True
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
