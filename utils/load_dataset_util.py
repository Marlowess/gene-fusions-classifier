from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import logging

import numpy as np
import pandas as pd

import matplotlib as plt

def _log_info_message(message: str, logger:  logging.Logger, skip_message: bool = False) -> None:
    """
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

def load_dataset(conf_load_dict: dict, main_logger: logging.Logger = None, k_mer_flag: bool = False) -> dict:
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

    if k_mer_flag is False:
        x_train, y_train = _prepared_data(
            path=path,
            sequence_type=sequence_type,
            bins_list=train_bins_list,
            names=columns_names,
            message='Loading Training Bins...',
            logger=main_logger)
    
        x_val, y_val = _prepared_data(
            path=path,
            sequence_type=sequence_type,
            bins_list=val_bins_list,
            names=columns_names,
            message='Loading Validation Bins...',
            logger=main_logger)
    
        x_test, y_test = _prepared_data(
            path=path,
            sequence_type=sequence_type,
            bins_list=test_bins_list,
            names=columns_names,
            message='Loading Test Bins...',
            logger=main_logger)
    else:
        x_train, y_train, x_val, y_val, x_test, y_test = \
            _load_training_data(train_bins_list, val_bins_list, test_bins_list, sequence_type)

    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test':x_test,
        'y_test': y_test
    }

def _prepared_data(path: str, sequence_type: str, bins_list: list, names: list, message: str, logger = None) -> object:
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

    _log_info_message(f" [*] {message}", logger)
    df = _get_full_dataframe(path, bins_list, names, logger)
    sequence_column = 'Sequences' if sequence_type == 'dna' else 'Translated_sequences'
    
    sequences = df[sequence_column].values
    labels = df['Label'].values
    
    # len_sequences_list = list(map(lambda xi: len(xi), sequences))
    # tmp_df = pd.DataFrame(len_sequences_list)
    # print(tmp_df.describe())
    
    return sequences, labels

def _get_full_dataframe(path: str, bins_list: list, names: list, logger: logging.Logger = None, verbose: int = 0) -> object:
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
        if verbose != 0:
            print(f" > Adding bin: {bin_path}...", end='')
        tmp_df = pd.read_csv(bin_path, skiprows=1, names=names)
        df_list.append(tmp_df)
        if verbose != 0:
            print(f" Done.")
        _log_info_message(f" > Added bin: {bin_path}, Done.", logger, skip_message=True)
    result_df = pd.concat(df_list)
    return result_df

def _load_training_data(train_bins_list: list, val_bins_list: list, test_bins_list: list, sequence_type: str, path: str):
    """
    Load all the training-data or validation-data for the amino-acid data-set.
    Returns the sequence and oncogenic, non-oncogenic class-labels.

    Inputs:
        - train: (optional) List of bins to use as validation set
        - validation: (optional) List of bins to use as validation set
    """
    X_train, y_train, X_val, y_val, X_test, y_test = None, None, None, None, None, None
    if (len(train_bins_list) >= 1):
        X_train, y_train = _load_bins(train_bins_list, sequence_type,path)
    if (len(val_bins_list) >= 1):
        X_val, y_val = _load_bins(val_bins_list, sequence_type, path) 
    if (len(test_bins_list) >= 1):
        X_test, y_test = _load_bins(test_bins_list, sequence_type, path)
    return X_train, y_train, X_val, y_val, X_test, y_test

def _load_bins(bins, sequence_type, path):
    """
    Load bins from file and concatenate them
    :param bins: bin numbers to read from file
    :return: concatenated data read from file

    """
    bin_dfs = [pd.read_csv(path + 'bin_' + str(i) +
        '_translated k3.csv') for i in bins]
    
    if (sequence_type == 'dna'):
        Xs = [bin_dfs[i]['k_mer_sequences'] for i in range(len(bin_dfs))]
    elif (sequence_type == 'protein'):
        Xs = [bin_dfs[i]['Translated_sequences'] for i in range(len(bin_dfs))]
    else:
        raise ValueError("sequence_type must be 'protein' or 'dna'")
    ys = [bin_dfs[i]['Label'] for i in range(len(bin_dfs))]
    Xs = pd.concat(Xs, axis=0)
    ys = pd.concat(ys, axis=0)
    X = Xs.values
    y = ys.values
        
    return X, y