import datetime
import logging
import os
import sys
import time

import tensorflow as tf

from utils_dir.load_dataset_util import load_dataset
from utils_dir.parse_args_util import get_parsed_params
from utils_dir.setup_analysis_environment_util import setup_analysis_environment


def main(conf_load_dict: dict, conf_preprocess_dict: dict, cmd_line_params: dict):

    base_dir: str = 'bioinfo_project'

    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_VAL = 64
    BUFFER_SIZE = 100
    embedding_dim = 256

    print(f"----> Set up analysis environment.")
    logger = setup_analysis_environment(logger_name=__name__, base_dir=base_dir, params=cmd_line_params)

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