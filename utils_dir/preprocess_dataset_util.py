import tensorflow as tf

import logging

import numpy as np
from pprint import pprint

def _log_info_message(message: str, logger:  logging.Logger, skip_message: bool = False) -> None:
    if logger is None:
        if skip_message is True: return
        print(message)
    else:
        logger.info(message)
    pass

def _labels_text2num(labels_list: list) -> np.array:
  """Function `_labels_text2num` takes a plain Python list as input,
  and retrieves a `np.array` object, mapping the values within the list from characters to non-negative integers.

  Params:
  -------
    :labels_list: non empty Python list.
  Return:
  -------
    :np.array: a numpy iterable array of non-negative integers.
  """

  char2int = lambda x: 0 if x == 'N' else 1
  mapped_labels: list = list(map(char2int, labels_list))
  return np.array(mapped_labels, dtype=np.int32)

def _tokenize(data_samples, conf_tok_dict: dict, data_tokenizer = None) -> object:
    """Function `_tokenize` receives as input parameters two objects, which
    are an iterable made of data samples containing the biological sequences,
    and a dictionary describing through which modes performing tokenization task.

    Params:
    -------
      :data_samples: iterable object, containing biological sequences.
      :conf_tok_dict: Python dict, with the options describing how to perform tokenization task.
    Return:
    -------
      :tensor: result provided by tokenization task, if specified this result might be onehot-encoded too.
      :data_tokenizer: `tf.keras.preprocessing.text.Tokenizer` object with the specification describing how tokenization task has been performed.
    """


    padding: str = conf_tok_dict['padding']
    maxlen: int = conf_tok_dict['maxlen']
    onehot_flag: bool = conf_tok_dict['onehot_flag']
    if data_tokenizer is None:
      data_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=True, char_level=True)
      data_tokenizer.fit_on_texts(data_samples)
    
    tensor = data_tokenizer.texts_to_sequences(data_samples)
    # print(data_tokenizer.index_word)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=maxlen, padding=padding)

    if onehot_flag is True:
      num_classes: int = conf_tok_dict['num_classes']
      tensor = tf.keras.utils.to_categorical(tensor, num_classes=num_classes)

    return tensor, data_tokenizer

def preprocess_data(data: dict, conf_tok_dict: dict, main_logger: logging.Logger) -> object:
  """Function `preprocess_data` takes as inputs two parameters which are two Python dictionary,
  which are the former a dictionary of samples and labels for train, val, and test sets, while the latter a dictionary
  with the specification describing how to carry out the preprocessing step.

  Params:
  -------
    :data: a Python dictionary within which there are as keys the train, val and test set of features samples, and as values the related iterable objects,
    for which there are also the corresponding train, val, test labels iterable objects referring to the related keys.
    :conf_tok_dict: a Python dictionary containing the specification that describes how to perform preprocessing phase
  Return:
    x_train, y_train, x_val, y_val, x_test, y_test
  """
  
  _log_info_message(f" [*] Preprocessing data...", main_logger)
  # pprint(conf_tok_dict)
  
  x_train, tokenizer = _tokenize(data['x_train'], conf_tok_dict)
  y_train = _labels_text2num(data['y_train'])

  x_val, _ = _tokenize(data['x_val'], conf_tok_dict, tokenizer)
  y_val = _labels_text2num(data['y_val'])

  x_test, _ = _tokenize(data['x_test'], conf_tok_dict, tokenizer)
  y_test = _labels_text2num(data['y_test'])
  
  return x_train, y_train, x_val, y_val, x_test, y_test, tokenizer