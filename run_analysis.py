import datetime
import json
import yaml
import logging
import os
import sys
import time

from pprint import pprint

from utils.parse_args_util import get_parsed_params
from utils.pipeline_analysis_util import run_pipeline
from utils.setup_analysis_environment_util import setup_analysis_environment

# =============================================================================================== #
# UTILITY FUNCTIONS                                                                               #
# =============================================================================================== #

def read_neural_network_params(cmd_line_params):
    if cmd_line_params.network_parameters is not None:
        network_params_path = cmd_line_params.network_parameters
    else:
        raise Exception('[ERROR] Please define a valid parameters\' filename')        
    
    # Parameters read from file
    network_params = get_neural_network_params_from_file(network_params_path)

    # It it exists, weights of a pre-trained model are loaded
    network_params['pretrained_model'] = cmd_line_params.pretrained_model
    return network_params

def get_neural_network_params_from_file(network_params_path: str) -> dict:
    result_dict: dict = None

    with open(network_params_path, "r") as f:
        if network_params_path.endswith('json'):
            result_dict = json.load(f)
        elif network_params_path.endswith('yaml'):
            result_dict = yaml.load(f)
    return result_dict

# =============================================================================================== #
# MAIN FUNCTION                                                                                   #
# =============================================================================================== #

def main(cmd_line_params: dict):

    base_dir: str = 'bioinfo_project'        

    network_params = read_neural_network_params(cmd_line_params) 
    
    # It defines the output file-system
    print(f"----> Set up analysis environment.")
    logger, meta_info_project_dict = setup_analysis_environment(logger_name=__name__, base_dir=base_dir, params=cmd_line_params)
    pprint(cmd_line_params)
    logger.info("\n" + json.dumps(network_params, indent=4))

    conf_load_dict: dict = {
        'sequence_type': cmd_line_params.sequence_type,
        'path': './data/bins_translated',
        'columns_names': [
            'Sequences','Count','Unnamed: 0','Label','Translated_sequences','Protein_length'
        ],
        'train_bins': [1,2,3],
        'val_bins': [4],
        'test_bins': [5],
    }

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
    # Useless rigth now. Just ignore
    dict_images: dict = {
        'loss': {
            'title': 'Training With Validation Loss',
            'fig_name': 'train_val_loss',
            'fig_format': 'png',
            'savefig_flag': True
        },
        'acc': {
            'title': 'Training With Validation Accuracy',
            'fig_name': 'train_val_acc',
            'fig_format': 'png',
            'savefig_flag': True
        },
        'roc_curve': {
            'title': 'Roc Curve',
            'fig_name': 'roc_curve',
            'fig_format': 'png',
            'savefig_flag': True
        },
        'confusion_matrix': {
            'title': 'Confusion Matrix',
            'fig_name': 'confusion_matrix',
            'fig_format': 'png',
            'savefig_flag': True
        }
    }

    cmd_line_params, _ = get_parsed_params()
    main(cmd_line_params)
    pass