#!/usr/bin/env python3
#-*- coding: utf-8 *-*

import datetime
import json
import yaml
import logging
import os
import sys
import time

from pprint import pprint

import tensorflow as tf

from utils.parse_args_util import get_parsed_params
from utils.pipeline_analysis_util import run_pipeline
from utils.setup_analysis_environment_util import setup_analysis_environment

from models.ModelFactory import ModelFactory

import numpy as np
import random

def read_neural_network_params(cmd_line_params):
    """
    Handles and organizes data provided by the configuration file
    """
    if cmd_line_params.network_parameters is not None:
        network_params_path = cmd_line_params.network_parameters
    else:
        raise Exception('[ERROR] Please define a valid parameters\' filename')        
    
    # Parameters read from file
    network_params = get_neural_network_params_from_file(network_params_path)

    # It it exists, weights of a pre-trained model are loaded
    network_params['pretrained_model'] = cmd_line_params.pretrained_model
    network_params['lr'] = cmd_line_params.lr
    network_params['batch_size'] = cmd_line_params.batch_size
    return network_params

def get_neural_network_params_from_file(network_params_path: str) -> dict:
    """
    Loads parameters from external configuration file (JSON or yaml)
    """
    result_dict: dict = None

    with open(network_params_path, "r") as f:
        if network_params_path.endswith('json'):
            result_dict = json.load(f)
        elif network_params_path.endswith('yaml'):
            result_dict = yaml.load(f)
    return result_dict

def _log_info_message(message: str, logger:  logging.Logger, skip_message: bool = False, tag_report: str = None) -> None:
    """
    It writes any log either on a logger or on stdout 

    Params:
    -------
        :message: str,
        :logger: logging.Logger,
        :skip_message: bool, default = False
    """

    if tag_report is not None:
        message = f"[{tag_report} - START]{message}[{tag_report} - END]"

    if logger is None:
        if skip_message is True: return
        print(message, file=sys.stdout)
    else:
        logger.info(message)
    pass

def main(cmd_line_params: dict, curr_date_str: str):

    # Setting seed values for enabling determins features, for:
    # - random built-in python module,
    # - numpy library,
    # - and, tensorflow library.
    random.seed(cmd_line_params.seed)
    np.random.seed(cmd_line_params.seed)
    tf.random.set_seed(cmd_line_params.seed)

    # Default name for output dir.
    base_dir: str = 'bioinfo_project' 

    # Read NN params from input file and
    # store them within dictionary-like object.
    network_params = read_neural_network_params(cmd_line_params) 
    
    # It defines and prepares the output file-system
    # where output results will be stored.
    print(f"----> Set up analysis environment.")
    logger, meta_info_project_dict = setup_analysis_environment(logger_name=str(__name__), base_dir=base_dir, params=cmd_line_params)
    
    logger.info(f"Running on date: {curr_date_str}")

    # Dictionary-like variable used to specify
    # how the data is loaded from source directory containing it.
    conf_load_dict: dict = {
        'sequence_type': cmd_line_params.sequence_type,
        'path': cmd_line_params.path_source_data,
        'columns_names': [
            'Sequences','Count','Unnamed: 0','Label','Translated_sequences','Protein_length', 'k_mer_sequences'
        ],
        'train_bins': [1,2,3],
        'val_bins': [4],
        'test_bins': [5],
    }

    # Dictionary-like variable used to specify
    # how the input data will be preprocessed, before executing one between
    # train phase or test phase, or both.
    conf_preprocess_dict: dict = {
        'padding': 'post',
        'maxlen': network_params['maxlen'],
        'onehot_flag': cmd_line_params.onehot_flag,
        'sequence_type': cmd_line_params.sequence_type
    }

    # Updating dictionary-object about NN params with some params values
    # coming from command line.
    network_params['batch_size'] = cmd_line_params.batch_size
    network_params['lr'] = cmd_line_params.lr
    network_params['sequence_type'] = cmd_line_params.sequence_type
    network_params['onehot_flag'] = cmd_line_params.sequence_type
    network_params['pretrained_model'] = cmd_line_params.pretrained_model
    network_params['onehot_flag'] = cmd_line_params.onehot_flag

    # This function is used to:
    # - either starts the training phases (holdout, validation or both),
    # - or to execute test phase (inference) onto test data or unknown new data.
    # - bat, both previous options can be executed one at a time and in the order specidied as above,
    #   so before is runned training phase and after test phase.
    run_pipeline(
        conf_load_dict=conf_load_dict,
        conf_preprocess_dict=conf_preprocess_dict,
        cmd_line_params=cmd_line_params,
        network_params=network_params,
        meta_info_project_dict=meta_info_project_dict,
        main_logger=logger
    )
    pass

if __name__ == "__main__":
    """Program entry point."""

    # Specifying some environment variables for
    # enabling determin feature while program is running.
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # Get command line params specified for running
    # a particular instance of this program.
    cmd_line_params, _, curr_date_str = get_parsed_params()

    # Running main function which contains the whole
    # logic of our program.
    main(cmd_line_params, curr_date_str)
    pass
