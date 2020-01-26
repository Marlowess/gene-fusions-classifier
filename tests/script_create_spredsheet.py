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
from utils.spredsheet_util import *

from openpyxl import Workbook

# =============================================================================================== #
# MAIN UTILS                                                                                      #
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

def run_test_decorator(a_test):
    def wrapper_function(a_dict):
        flag_test_passed: bool = True 
        try:
            message: str = f" [*] Running TEST for function {a_test.__name__}"
            print()
            print(f"{message}", '-' * len(message), sep='\n')

            result = a_test(a_dict)
        except Exception as err:
            print(f'ERROR: {str(err)}')
            flag_test_passed = False
            sys.exit(-1)
        finally:
            status_test: str = 'PASSED' if flag_test_passed is True else 'FAILED'
            message: str = f" [*] TEST on function {a_test.__name__} ended with STATUS = {status_test}"
            print()
            print(f"{message}", '-' * len(message), sep='\n')
        return result
    return wrapper_function

# =============================================================================================== #
# TESTS SECTION                                                                                   #
# =============================================================================================== #

@run_test_decorator
def test_write_history_2_spredsheet(spredsheet_info_dict: dict) -> bool:

    workbook: Workbook = spredsheet_info_dict['workbook']
    workbook_name: str = spredsheet_info_dict['workbook_name']

    path_2_history: str = spredsheet_info_dict['path_2_history']
    columns: list = spredsheet_info_dict['columns']

    add_history_2_spredsheet(
        path_2_history=path_2_history,
        columns=columns,
        workbook=workbook,
        sheet_name='history'
    )

    save_workbook(workbook, workbook_name)
    close_workbook(workbook)
    return True

@run_test_decorator
def test_write_input_params_2_spredsheet(spredsheet_info_dict: dict) -> bool:

    workbook: Workbook = spredsheet_info_dict['workbook']
    workbook_name: str = spredsheet_info_dict['workbook_name']

    cmd_args_dict :dict = spredsheet_info_dict['cmd_args_dict']
    network_input_params_dict: dict = spredsheet_info_dict['network_input_params_dict']

    add_input_params_2_spredsheet(
        cmd_args_dict=cmd_args_dict,
        network_input_params_dict=network_input_params_dict,
        workbook=workbook
    )

    return True

@run_test_decorator
def test_write_graph_model_2_spredsheet(spredsheet_info_dict: dict) -> bool:

    workbook: Workbook = spredsheet_info_dict['workbook']
    workbook_name: str = spredsheet_info_dict['workbook_name']

    image_path : str = spredsheet_info_dict['image_path']

    add_image_2_spredsheet(
        image_path,
        workbook=workbook
    )

    return True

# =============================================================================================== #
# MAIN FUNCTION                                                                                   #
# =============================================================================================== #

def main(cmd_line_params: dict):

    base_dir: str = 'bioinfo_project'
    network_params = read_neural_network_params(cmd_line_params)

    conf_load_dict: dict = {
        'sequence_type': cmd_line_params.sequence_type,
        'path': cmd_line_params.network_parameters,
        'columns_names': [
            'Sequences','Count','Unnamed: 0','Label','Translated_sequences','Protein_length'
        ],
        'train_bins': [1,2,3],
        'val_bins': [4],
        'test_bins': [5],
    }

    # -------------------------------------------- #
    # Here - Start spredsheet tests                #
    # -------------------------------------------- #

    workbook: Workbook = get_workbook()
    workbook_name: str = 'analysis.xlsx'

    # print(type(vars(cmd_line_params)))
    # print(type(network_params))
    # sys.exit(0)

    cmd_args_dict: dict = vars(cmd_line_params)
    spredsheet_info_dict: dict = {
        'workbook': workbook,
        'workbook_name': workbook_name,
        'cmd_args_dict': cmd_args_dict,
        'network_input_params_dict': network_params,
    }

    test_write_input_params_2_spredsheet(spredsheet_info_dict)
    # save_workbook(workbook, workbook_name)

    spredsheet_info_dict: dict = {
        'workbook': workbook,
        'workbook_name': workbook_name,
        'path_2_history': './tests/res_spredsheet_tests/history.csv',
        'columns': [
            'epoch,accuracy',
            'f1_m',
            'loss',
            'precision_m',
            'recall_m',
            'val_accuracy',
            'val_f1_m',
            'val_loss',
            'val_precision_m',
            'val_recall_m'
        ],
    }
    
    test_write_history_2_spredsheet(spredsheet_info_dict)
    

    spredsheet_info_dict: dict = {
        'workbook': workbook,
        'workbook_name': workbook_name,
        'image_path': './tests/res_spredsheet_tests/model_graph.png'
    }

    test_write_graph_model_2_spredsheet(spredsheet_info_dict)


    save_workbook(workbook, workbook_name)
    close_workbook(workbook)

    pass

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