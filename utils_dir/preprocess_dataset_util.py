import tensorflow as tf

import numpy as np
from pprint import pprint

def _labels_text2num(labels_list: list) -> np.array:
  char2int = lambda x: 0 if x == 'N' else 1
  mapped_labels: list = list(map(char2int, labels_list))
  return np.array(mapped_labels)

def _tokenize(data_samples, conf_tok_dict: dict) -> object:

    padding: str = conf_tok_dict['padding']
    maxlen: int = conf_tok_dict['maxlen']
    onehot_flag: bool = conf_tok_dict['onehot_flag']

    data_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=True)
    
    data_tokenizer.fit_on_texts(data_samples)
    
    tensor = data_tokenizer.texts_to_sequences(data_samples)
    
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=maxlen, padding=padding)

    if onehot_flag is True:
      num_classes: int = conf_tok_dict['num_classes']
      tensor = tf.keras.utils.to_categorical(tensor, num_classes=num_classes)

    return tensor, data_tokenizer

def preprocess_data(data, conf_tok_dict: dict) -> object:
  
  print(f" [*] Preprocessing data...")
  # pprint(conf_tok_dict)
  
  x_train, _ = _tokenize(data['x_train'], conf_tok_dict)
  y_train = _labels_text2num(data['y_train'])

  x_val, _ = _tokenize(data['x_val'], conf_tok_dict)
  y_val = _labels_text2num(data['y_val'])

  x_test, _ = _tokenize(data['x_test'], conf_tok_dict)
  y_test = _labels_text2num(data['y_test'])
  
  return x_train, y_train, x_val, y_val, x_test, y_test