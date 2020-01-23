import argparse
import datetime
import logging
import os
import sys
import time


def _init_logger(log_dir: str, filename: str = 'main.log', flag_test: bool = False) -> logging.Logger:
    """
    Initialize main logger
    :param log_dir: directory where to save log file
    """

    if flag_test is True:
        filename = f"test_{filename}"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    formatter = logging.Formatter('%(message)s')
    # file handler
    file_handler = logging.FileHandler(os.path.join(log_dir,  f"{filename}"))
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

def setup_analysis_environment(logger_name: logging.Logger, base_dir: str, params: dict, filename: str = 'main.log', flag_test: bool = False) -> object:
    print(" [*] Creating environment for saving current analysis results...")

    result_dict : dict = dict()

    result_dict['base_dir'] = base_dir

    if flag_test is True or params.experimental_mode is True:
        flag_test = True
        results_dir = os.path.join(base_dir, f"{params.output_dir}_test")
    else:
        results_dir = os.path.join(base_dir, params.output_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    logger = _init_logger(results_dir, filename=filename, flag_test=flag_test)

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
            result_dict['train_result_path'] = train_result_path
        except:
            pass
    return logger, result_dict