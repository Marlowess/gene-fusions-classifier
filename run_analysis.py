import datetime
import json
import yaml
import logging
import os
import sys
import time

from pprint import pprint

from utils_dir.parse_args_util import get_parsed_params
from utils_dir.pipeline_analysis_util import run_pipeline
from utils_dir.setup_analysis_environment_util import setup_analysis_environment

def get_network_params(network_params_path: str) -> dict:
    result_dict: dict = None

    with open(network_params_path, "r") as f:
        if network_params_path.endswith('json'):
            result_dict = json.load(f)
        elif network_params_path.endswith('yaml'):
            result_dict = yaml.load(f)
    return result_dict

def main(conf_load_dict: dict, conf_preprocess_dict: dict, cmd_line_params: dict):

    base_dir: str = 'bioinfo_project'

    # pprint(cmd_line_params)

    if cmd_line_params.network_parameters is not None:
        network_params_path = cmd_line_params.network_parameters
    else:
        network_params_path = 'parameters.json'

    network_params = get_network_params(network_params_path)
    network_params['pretrained_model'] = cmd_line_params.pretrained_model
    
    print(f"----> Set up analysis environment.")
    logger, meta_info_project_dict = setup_analysis_environment(logger_name=__name__, base_dir=base_dir, params=cmd_line_params)
    pprint(cmd_line_params)

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

    conf_load_dict: dict = {
        'sequence_type': 'dna',
        'path': './bins_translated',
        'columns_names': [
            'Sequences','Count','Unnamed: 0','Label','Translated_sequences','Protein_length'
        ],
        'train_bins': [1,2,3],
        'val_bins': [4],
        'test_bins': [5],
    }

    conf_preprocess_dict: dict = {
        'padding': 'post',
        'maxlen': 14000,
        'onehot_flag': False,
    }

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

    main(conf_load_dict, conf_preprocess_dict, cmd_line_params)
    pass