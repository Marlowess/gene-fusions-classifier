import argparse
import datetime
import logging
import os
import sys
import time


def _init_logger(log_dir: str) -> logging.Logger:
    """
    Initialize main logger
    :param log_dir: directory where to save log file
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    # file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'main.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    # add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info('logger initialization done!')
    return logger

def setup_analysis_environment(logger_name: logging.Logger, base_dir: str, params: dict) -> object:
    print(" [*] Creating environment for saving current analysis results...")

    result_dict : dict = dict()

    result_dict['base_dir'] = base_dir

    results_dir = os.path.join(base_dir, params.subdir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    logger = _init_logger(results_dir)

    logger.info(params)

    if params.validation is True:
        try:
            val_result_path: str = os.path.join(results_dir, f"results_holdout_validation")
            os.makedirs(val_result_path)
            result_dict['val_result_path'] = val_result_path
        except:
            pass
    
    if params.train is True:
        try:
            train_result_path: str = os.path.join(results_dir, f"results_train")
            os.makedirs(train_result_path)
            result_dict['train_result_path'] = val_result_path
        except:
            pass
    return logger, result_dict