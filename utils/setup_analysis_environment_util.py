import argparse
import datetime
import logging
import os
import sys
import time


def _init_logger(log_dir: str, filename: str = 'main.log') -> logging.Logger:
    """
    Initialize main logger
    :param log_dir: directory where to save log file
    """

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

def setup_analysis_environment(logger_name: str, base_dir: str, params: dict, filename: str = 'main.log') -> object:
    """
    It prepares folder for saving log file and execution outcome depending on whether validation, train or test is performed.
    
    Params:
    -------
        :logger_name: logger name.
        :base_dir: str, base output directory folder path.
        :params: dictionary, command line parameters.
        :filename:, str, log file name.
    """
    print(" [*] Creating environment for saving current analysis results...")

    result_dict : dict = dict()

    result_dict['base_dir'] = base_dir

    results_dir = os.path.join(base_dir, params.output_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    logger = _init_logger(results_dir, filename=filename)

    logger.info(params)

    # if params.validation is true setups or prepares an output
    # directory where validation results will be stored.
    if params.validation is True:
        try:
            val_result_path: str = os.path.join(results_dir, f"results_holdout_validation")
            os.makedirs(val_result_path)
            result_dict['val_result_path'] = val_result_path
        except:
            pass
    
    # if params.train is true setups or prepares an output
    # directory where training results will be stored.
    if params.train is True:
        try:
            train_result_path: str = os.path.join(results_dir, f"results_train")
            os.makedirs(train_result_path)
            result_dict['train_result_path'] = train_result_path
        except:
            pass

    # if params.test is true setups or prepares an output
    # directory where test results will be stored.
    if params.test is True:
        try:
            test_result_path: str = os.path.join(results_dir, f"results_test")
            os.makedirs(test_result_path)
            result_dict['test_result_path'] = test_result_path
        except:
            pass

    return logger, result_dict