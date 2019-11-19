__author__ = "Stefano Brilli"
__copyright__ = "Copyright 2019, Bioinformatics course's project"
__credits__ = ["Stefano Brilli"]
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = "Stefano Brilli"
__email__ = "s249914@studenti.polito.it"
__status__ = "Production"

import tensorflow as tf
import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


############################
#   PREPROCESSING METHODS  #
############################


def encode_messages(messages, vocab_to_int):
    """
    Encode messages
    :param messages: list of list of strings. List of message tokens
    :param vocab_to_int: mapping of vocab to idx
    :return: list of ints. Lists of encoded messages
    """
    messages_encoded = []
    for message in messages:
        messages_encoded.append([vocab_to_int[word] for word in message.split()])

    return np.array(messages_encoded)


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict maps a vocab word to and integeter
             The second maps an integer back to to the vocab word
    """
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab, 1)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


def zero_pad_messages(messages, seq_len):
    """
    Zero Pad input messages
    :param messages: Input list of encoded messages
    :param seq_len: Input int, maximum sequence input length
    :return: numpy array.  The encoded labels
    """
    messages_padded = np.zeros((len(messages), seq_len), dtype=int)
    for i, row in enumerate(messages):
        messages_padded[i, -len(row):] = np.array(row)[:seq_len]

    return np.array(messages_padded)


def pad_features(reviews_int, seq_length):
    """
    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    """
    features = np.zeros((len(reviews_int), seq_length), dtype=int)

    for i, review in enumerate(reviews_int):
        review_len = len(review)

        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length - review_len))
            new = zeroes + review

        # elif review_len > seq_length:
        else:
            new = review[0:seq_length]

        features[i, :] = np.array(new)

    return features


def train_val_test_split(messages, labels, split_frac, random_seed=None):
    """
    Zero Pad input messages
    :param random_seed: Seed for shuffling data
    :param messages: Input list of encoded messages
    :param labels: Input list of encoded labels
    :param split_frac: Input float, training split percentage
    :return: tuple of arrays train_x, val_x, test_x, train_y, val_y, test_y
    """
    # make sure that number of messages and labels allign
    assert len(messages) == len(labels)
    # random shuffle data
    if random_seed:
        np.random.seed(random_seed)
    shuf_idx = np.random.permutation(len(messages))
    messages_shuf = np.array(messages)[shuf_idx]
    labels_shuf = np.array(labels)[shuf_idx]

    # make splits
    split_idx = int(len(messages_shuf) * split_frac)
    train_x, val_x = messages_shuf[:split_idx], messages_shuf[split_idx:]
    train_y, val_y = labels_shuf[:split_idx], labels_shuf[split_idx:]

    test_idx = int(len(val_x) * 0.5)
    val_x, test_x = val_x[:test_idx], val_x[test_idx:]
    val_y, test_y = val_y[:test_idx], val_y[test_idx:]

    return train_x, val_x, test_x, train_y, val_y, test_y


######################
#     I/O METHODS    #
######################

# Function for writing the log file
def print_and_write(log_file, data):
    """
    Prints data in log_file
    :param log_file:
    :param data:
    :return:
    """
    print(data)
    log_file.write(data)


def print_model_infos(log_file, params):
    """
    Prints the model's architecture in log_file
    :param log_file:
    :param params:
    :return:
    """
    for key, value in params.items():
        data = '{} : {}\n'.format(key, value)
        print(data)
        log_file.write(data)


def read_data(dataset_name):
    """
    Reads dataset from dataset_name (path)
    :param dataset_name:
    :return:
    """
    data = pd.read_csv(dataset_name)

    # Perform a shuffle
    # data.sample(frac=1, random_state=19)

    features = data.protein.values
    labels = data.label.map({'C': 1, 'N': 0}).values

    # Define the list of all words and create a dictionary
    full_lexicon = " ".join(features).split()
    vocab_to_int, int_to_vocab = create_lookup_tables(full_lexicon)
    # print("Vocabulary size: {}".format(len(vocab_to_int)))
    # print(vocab_to_int)

    # Encode features and labels
    # No need to encode labels, since they has been encoded during the data loading
    proteins = encode_messages(features, vocab_to_int)

    # proteins_lens = Counter([len(x) for x in proteins])
    # print("Zero-length proteins: {}".format(proteins_lens[0]))
    # print("Maximum protein length: {}".format(max(proteins_lens)))
    # print("Average protein length: {}".format(np.mean([len(x) for x in features])))

    # Then I have to pad proteins, before applying one-hot encoding
    # proteins = zero_pad_messages(proteins, seq_len=max(proteins_lens))

    return proteins, labels, vocab_to_int, int_to_vocab


def load_parameters(file_path="parameters.json"):
    """
    It reads parameters from a JSON file and convertes them to a Python dictionary
    :param file_path:
    :return: a dictionary containing parameters
    """

    with open(file_path, "r") as fp:
        return json.load(fp)


def print_params_to_json(params, path):
    """
    This methods print params to a file
    :param params:
    :param path:
    :return:
    """
    data = json.dumps(params)

    with open(path, "w+") as log:
        log.write(data)


def print_train_infos(history, title):
    """
    This method plots informations on losses for train and validation set during the training
    :param history:
    :return:
    """
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.title(title)
    plt.show()


######################
#   MODEL'S METHODS  #
######################


def initialize_model(architecture):
    """
    Loads architecture's infos from the json file and builds a tf.keras model
    :param architecture: json containing infos about the model
    :return:
    """

    global optimizer
    model = tf.keras.Sequential(name=architecture['model_name'])

    for layer in architecture['layers']:
        model.add(eval(layer['layer']))

    # Compile the model and return
    loss = architecture['loss']
    lr = architecture['learning_rate']
    clip_norm = architecture['clip_norm']

    if architecture['optimizer'] is 'ADAM':
        optimizer = tf.keras.optimizers.Adam(lr=lr, clipnorm=clip_norm)
    else:
        optimizer = tf.keras.optimizers.RMSprop(lr=lr, clipnorm=clip_norm)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=architecture['metrics'])

    return model


def get_callbacks_from_dict(callbacks, phase):
    """
    Reads informations on the callbacks from JSON and put them into a list
    :param callbacks:
    :return:
    """
    callbacks_list = []

    for callback in callbacks:
        if callback['enable'] is False:
            continue
        else:
            callbacks_list.append(callback)
    return callbacks_list


def train_cv(x_train,
             y_train,
             architecture,
             log_file,
             n_splits,
             seed,
             n_epochs,
             batch_size,
             callbacks):
    """
    Train the network by using cross-validation
    :param callbacks:
    :param batch_size:
    :param n_epochs:
    :param seed:
    :param shuffle:
    :param n_splits:
    :param log_file:
    :param architecture:
    :param x_train:
    :param y_train:
    :return:
    """

    # Train by using cross-validation
    kfold = StratifiedKFold(n_splits=n_splits,
                            shuffle=True,
                            random_state=seed)
    cvscores = []

    print_and_write(log_file, "\n*** CROSS VALIDATION PHASE ***\n")

    counter = 1
    for train, test in kfold.split(x_train, y_train):
        # Reset the model
        model_cv = initialize_model(architecture)

        # Fit the model
        history = model_cv.fit(x_train[train], y_train[train], epochs=n_epochs,
                               batch_size=batch_size, verbose=1,
                               validation_data=(x_train[test], y_train[test]),
                               callbacks=callbacks)

        # Evaluate the model
        scores = model_cv.evaluate(x_train[test], y_train[test], verbose=1)
        # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print_and_write(log_file, "Fold %d " % counter)
        print_and_write(log_file, "%s: %.2f%%\n" % (model_cv.metrics_names[1], scores[1] * 100))
        print_model_infos(log_file, history.history)
        print_and_write(log_file, "\n")
        cvscores.append(scores[1] * 100)
        print_train_infos(history, "Fold %d " % counter)
        print_and_write(log_file, "\n")
        counter += 1

    # print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print_and_write(log_file, "\n\nMean accuracy: %.2f%% (+/- %.2f%%)\n\n" % (np.mean(cvscores), np.std(cvscores)))
    print_and_write(log_file, "------------------------------------------")

    return np.mean(cvscores)


# Method used for fit the model on the entire training set
def train(x_train, y_train, model, log_file, seed, n_epochs, batch_size, callbacks):
    """
    This method trains a method for the specified number of epochs
    :param y_val:
    :param x_val:
    :param x_train:
    :param y_train:
    :param model:
    :param log_file:
    :param n_epochs:
    :param batch_size:
    :param callbacks:
    :return:
    """
    print_and_write(log_file, "\n*** TRAINING PHASE ***\n")

    history = model.fit(x_train, y_train, epochs=n_epochs,
                        batch_size=batch_size, verbose=1,
                        validation_split=0.25,
                        callbacks=callbacks)

    print_model_infos(log_file, history.history)
    acc = "%s: %.2f%%\n" % (model.metrics_names[1], history.history['acc'][len(history.history['acc']) - 1] * 100)
    print_and_write(log_file, acc)
    print_and_write(log_file, "------------------------------------------")
    print_train_infos(history, 'Training phase')
