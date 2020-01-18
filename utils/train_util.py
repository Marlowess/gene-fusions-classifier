import argparse
import datetime
import logging
import os
import sys
import time

from models.ModelFactory import ModelFactory
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

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

def _get_callbacks_list(history_filename: str) -> list:
    callbacks_list: list = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    keras.callbacks.ModelCheckpoint(
                        filepath='my_model.h5',
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=0
                    ),
                    keras.callbacks.CSVLogger(history_filename),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=5,
                        monitor='val_loss',
                        factor=0.75,
                        verbose=1,
                        min_lr=5e-6)
    ]
    return callbacks_list

def _holdout(
    x_train,
    y_train,
    x_val,
    y_val,
    conf_load_dict: dict,
    cmd_line_params,
    network_params: dict,
    meta_info_project_dict: dict,
    tokenizer: Tokenizer,
    logger: logging.Logger,
    message: str = 'Performing first phase (holdout)...') -> object:
    
    # Some logs recorded.
    _log_info_message(f" [*] {message}", logger)

    train_bins, val_bins = conf_load_dict['train_bins'], conf_load_dict['val_bins']
    _log_info_message("Training on bins: {}, validation on {}".format(train_bins, val_bins), logger)

    vocabulary_len: int = len(tokenizer.index_word) + 1
    network_params['vocabulary_len'] = vocabulary_len

    # Get Callbacks.
    base_dir: str = meta_info_project_dict['base_dir']
    history_filename: str = os.path.join(base_dir, 'history.csv')
    network_params['result_base_dir'] = meta_info_project_dict['val_result_path']    

    # TODO: callbacks are defined for each model --> remove them from all dictionaries
    callbacks_list = _get_callbacks_list(history_filename)

    # Get Model from ModelFactory Static class.
    network_model_name: str = cmd_line_params.load_network
    model = ModelFactory.getModelByName(network_model_name, network_params)

    # Build model.
    _log_info_message(f"> build model (holdout).", logger)
    
    # It compiles the model and print its summary (architecture's structure)
    model.build(logger)

    # It plots on a file the model's structure
    model.plot_model()

    sys.exit(-1)

    # Train model.
    _log_info_message(f"> train model (holdout)...", logger)

    # TODO: the wrapper fit method does not receive the callbacks' list, remove it!
    model.fit(
        X_tr=x_train,
        y_tr=y_train,
        epochs=cmd_line_params.num_epochs,
        callbacks_list=callbacks_list,
        validation_data=(x_val, y_val)
        )
    _log_info_message(f"> train model (holdout): Done.", logger)

    # Eval model.
    _log_info_message(f"> eval model (holdout).", logger)

    # scores contains [loss, accuracy, f1_score, precision, recall]
    results_dict = model.evaluate(x_val, y_val)
    res_string = ", ".join(f'{k}:{v}' for k,v in results_dict.items())
    _log_info_message("{}".format(results_str), logger)
    _log_info_message(f" [*] {message} Done.", logger)
    return model

def _train():
    pass