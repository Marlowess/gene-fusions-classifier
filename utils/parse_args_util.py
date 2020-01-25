import argparse
import datetime
import os
import sys
import time


def _get_custom_parser(subdir: str) -> object:

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--subdir', default=subdir, help='Checkpoints directory')
    parser.add_argument('--validation', default=False, help='Perform Holdout-Validation for hyperparameter selection', action='store_true')
    parser.add_argument('--train', default=False, help='Train model on whole training data and save it', action='store_true')
    parser.add_argument('--test', default=False, help='Test saved model, in the specified subdir, on test bin', action='store_true')
    parser.add_argument('--batch_size', default=10, help='Number of sample for each training step',type=int)
    parser.add_argument('--num_epochs', default=50, help='Number of epochs before halting the training',type=int)
    parser.add_argument('--lr', default=1e-3, help='Learning rate coefficient',type=float)
    parser.add_argument('--sequence_type', choices=['dna', 'protein'], help='Type of sequence to process in the model: "dna" or "protein"', type=str)
    parser.add_argument('--pretrained_model', help='Path where to find the weights of a pre-trained model', type=str, default=None)
    parser.add_argument('--network_parameters', default=None, help='File with neural network parameters, either json or yaml format', type=str)
    parser.add_argument('--load_network', default=None, help='Architecture\'s name. According to this a different model is loaded', type=str)
    parser.add_argument('--onehot_flag', default=False, help='If true, it encodes data by using one-hot encodin, otherwise embedding representation is used', action='store_true')
    parser.add_argument('--steps', default=None, help='Number of steps of training', type=int)
    parser.add_argument('--early_stopping_epoch', default=None, help='Number of epochs after holdout train stops for early stopping', type=int)
 
    params = parser.parse_args()

    return params, parser

def get_parsed_params() -> dict:

    subdir: str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')

    params, parser = _get_custom_parser(subdir)
    return params, parser