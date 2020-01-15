from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import numpy as np
import pandas as pd

def load_dataset(conf_load_dict: dict) -> dict:

    sequence_type: str = conf_load_dict['sequence_type']
    path: str = conf_load_dict['path']

    columns_names: list = conf_load_dict['columns_names']

    train_bins_list: list = conf_load_dict['train_bins']
    val_bins_list: list = conf_load_dict['val_bins']
    test_bins_list: list = conf_load_dict['test_bins']

    x_train, y_train = _prepared_data(
        path=path,
        sequence_type=sequence_type,
        bins_list=train_bins_list,
        names=columns_names,
        message='Loading Training Bins...')
    
    x_val, y_val = _prepared_data(
        path=path,
        sequence_type=sequence_type,
        bins_list=val_bins_list,
        names=columns_names,
        message='Loading Validation Bins...')
    
    x_test, y_test = _prepared_data(
        path=path,
        sequence_type=sequence_type,
        bins_list=test_bins_list,
        names=columns_names,
        message='Loading Test Bins...')

    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test':x_test,
        'y_test': y_test
    }

def _prepared_data(path: str, sequence_type: str, bins_list: list, names: list, message: str) -> object:

    print(f" [*] {message}")
    df = _get_full_dataframe(path, bins_list, names)
    sequence_column = 'Sequences' if sequence_type == 'dna' else 'Translated_sequences'
    
    sequences = df[sequence_column].values
    labels = df['Label'].values
    return sequences, labels

def _get_full_dataframe(path: str, bins_list: list, names: list) -> object:
    result_df = None
    df_list = list()
    for _, bin_no in enumerate(bins_list):
        bin_path = os.path.join(path, f"bin_{bin_no}_translated.csv")
        print(f" > Adding bin: {bin_path}...", end='')
        tmp_df = pd.read_csv(bin_path, skiprows=1, names=names)
        df_list.append(tmp_df)
        print(f" Done.")
    result_df = pd.concat(df_list)
    return result_df