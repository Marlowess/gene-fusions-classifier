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

# =============================================================================================== #
# UTILITY FUNCTIONS                                                                               #
# =============================================================================================== #

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

# =============================================================================================== #
# COMPILE MODEL FUNCTION                                                                          #
# =============================================================================================== #

def _log_info_message(message: str, logger:  logging.Logger, skip_message: bool = False, tag_report: str = None) -> None:
    """
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
        print(message)
    else:
        logger.info(message)
    pass

# =============================================================================================== #
# MAIN FUNCTION                                                                                   #
# =============================================================================================== #

def main(cmd_line_params: dict, curr_date_str: str):

    tf.random.set_seed(cmd_line_params.seed)

    base_dir: str = 'bioinfo_project'
    status_analysis: str = "SUCCESS"   

    network_params = read_neural_network_params(cmd_line_params) 
    
    # It defines the output file-system
    print(f"----> Set up analysis environment.")
    logger, meta_info_project_dict = setup_analysis_environment(logger_name=str(__name__), base_dir=base_dir, params=cmd_line_params)
    
    logger.info(f"Running on date: {curr_date_str}")

    conf_load_dict: dict = {
        'sequence_type': cmd_line_params.sequence_type,
        'path': cmd_line_params.path_source_data,
        'columns_names': [
            'Sequences','Count','Unnamed: 0','Label','Translated_sequences','Protein_length'
        ],
        'train_bins': [1,2,3],
        'val_bins': [4],
        'test_bins': [5],
    }
    
    network_params['batch_size'] = cmd_line_params.batch_size
    network_params['lr'] = cmd_line_params.lr
    network_params['sequence_type'] = cmd_line_params.sequence_type
    network_params['onehot_flag'] = cmd_line_params.sequence_type
    network_params['pretrained_model'] = cmd_line_params.pretrained_model
    network_params['onehot_flag'] = cmd_line_params.onehot_flag

    conf_preprocess_dict: dict = {
        'padding': 'post',
        'maxlen': network_params['maxlen'],
        'onehot_flag': cmd_line_params.onehot_flag,
    }

    # This function starts the training phases (holdout, validation or both)
    run_pipeline(
        conf_load_dict=conf_load_dict,
        conf_preprocess_dict=conf_preprocess_dict,
        cmd_line_params=cmd_line_params,
        network_params=network_params,
        meta_info_project_dict=meta_info_project_dict,
        main_logger=logger
    )
    pass


# =============================================================================================== #
# ENTRY - POINT                                                                                   #
# =============================================================================================== #

if __name__ == "__main__":
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    cmd_line_params, _, curr_date_str = get_parsed_params()
    main(cmd_line_params, curr_date_str)
    pass
