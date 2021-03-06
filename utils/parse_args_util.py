import argparse
import datetime
import os
import sys
import time


def _get_custom_parser(output_dir: str) -> object:

    """
    It is used to obtain na object with the values provided from command line, and set the default value
    of the output directory providing it as input params to the function itself.

    Params:
    -------
        :output_dir: default value for the output dir.\n
    Returns:
    --------
        :object: with the values provided from command line.\n
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1234, help='Value assigned to global seed for initializing tensorflow module entirely.', type=int)

    parser.add_argument('--output_dir', default=output_dir, help='Directory path for storing all results gained from running this analysis tool.')
    parser.add_argument('--path_source_data', default='./data/bins_translated', help='Path that specifies the location of source datasets.', type=str)
    parser.add_argument('--sequence_type', default='protein', choices=['dna', 'protein', 'kmers'], help='Type of sequence to process in the model: "dna","protein" or "kmers"', type=str)

    parser.add_argument('--train', default=False, help='Train model on whole training data and save it', action='store_true')
    parser.add_argument('--validation', default=False, help='Perform Holdout-Validation for hyperparameter selection', action='store_true')
    parser.add_argument('--test', default=False, help='Test saved model, in the specified subdir, on test bin', action='store_true')

    parser.add_argument('--lr', default=1e-3, help='Learning rate coefficient',type=float)
    parser.add_argument('--num_epochs', default=50, help='Number of epochs before halting the training',type=int)
    parser.add_argument('--batch_size', default=10, help='Number of sample for each training step',type=int)

    parser.add_argument('--network_parameters', default=None, help='File with neural network parameters, either json or yaml format', type=str)
    parser.add_argument('--load_network', default=None, help='Architecture\'s name. According to this a different model is loaded', type=str)

    parser.add_argument('--onehot_flag', default=False, help='If true, it encodes data by using one-hot encodin, otherwise embedding representation is used', action='store_true')    
    parser.add_argument('--early_stopping_on_loss', default=False, help='If true, it perform early stopping during train based on train loss of holdout phase ', action='store_true')    
    parser.add_argument('--early_stopping_epoch', default=None, help='Number of epochs after holdout train stops for early stopping', type=int)
    parser.add_argument('--pretrained_model', help='Path where to find the weights of a pre-trained model', type=str, default=None)
 
    params = parser.parse_args()

    return params, parser

def get_parsed_params() -> dict:
    """
    It is used to get the dictionary containing the command line values passed
    as input parameters to the program

        
    Returns:
    --------
        :dict: dictionary containing command line values passed
    as input parameters to the program.\n
    """

    output_dir: str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')
    output_dir_date: str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d') 
    output_dir_time: str = datetime.datetime.strftime(datetime.datetime.now(), '%H_%M_%S') 

    params, parser = _get_custom_parser(output_dir)

    if params.output_dir != output_dir:
        params.output_dir = \
            os.path.join(f"{params.output_dir}", f"{output_dir_date}", f"train_{output_dir_time}")

    curr_date: str = output_dir
    return params, parser, curr_date