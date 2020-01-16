from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import numpy as np
import pandas as pd

def load_dataset(conf_load_dict: dict) -> dict:
    """Function `load_dataset` takes as input parameters a single argument which is just a Python dictionary,
    that is involved into describig how to load or fetch the requested data samples, for performing some kind of
    analysis, later.

    Params:
    -------
        :conf_load_dict: Python dictionary, containing some meta-info describing which kind of dataset to load.
    Return:
    -------
        A Python dictionary.
    """

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
    """Functon `_prepared_data` takes as input up tp 5 parametes, which are a path corresponding to the dataset location,
    an argument referring to the kind of biological sequence we are looking for, a list of bins to combine into a single
    pandas dataframe and then extract the pair (sequences, label) necessary for our later analysis, a list of names for referring
    to pandas df features, and finally a message python str used for verbose reasons detailng how is the state of such function.

    Params:
    -------
        :path: Python str, path used in order to reach and fetch the necessary bins of data samples.
        :sequence_type: Pyhton str, type of sequence, currently there are only two kind of supported biological sequences which are respectively 'dna' or 'protein'.
        :bins_list: Pyhton list with the index-like values for referring to the bins we aim at loading and stacking all together.
        :names: Python list, with the features or columns names of each bin file, since it is a csv-like file.
        :message: Python str used for verbose reasons detailng how is the state of this function.
    Return:
    -------
        :sequences, labels: pairs of biological sequences, either dna or protein, with their corresponding encoded label, either 0 for not-cancer and 1 for cancer.
    """

    print(f" [*] {message}")
    df = _get_full_dataframe(path, bins_list, names)
    sequence_column = 'Sequences' if sequence_type == 'dna' else 'Translated_sequences'
    
    sequences = df[sequence_column].values
    labels = df['Label'].values
    return sequences, labels

def _get_full_dataframe(path: str, bins_list: list, names: list) -> object:
    """Function `_get_full_dataframe` taking as input parameters a Python str that is the path argument
    which refers to the dataset location used for performing some kind of analysis later, a list of bins to load or fetch and
    then stack or combine into a single pandas df, and finally a list of columns names for both data samples' features and labels.
    
    Params:
    -------
        :path: Python str, path toward dataset location.
        :bins_list: Python list with bins's index to load.
        :names: Python list with bin's columns names.
    Return:
        :result_df: pandas df resulting from the concatenation of all loaded bins.
    """
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