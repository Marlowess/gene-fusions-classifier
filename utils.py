from tensorflow.keras.utils import Sequence
import tensorflow as tf
from keras.utils import to_categorical
import numpy as np

import json

class BioSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, tokenizer, shuffle=True, seed=0):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.tokenizer = tokenizer
 
        idxs = np.arange(self.x.shape[0])
        if (shuffle):
            if (seed is not None):
                np.random.seed(seed)
            np.random.shuffle(idxs)
            self.x = self.x[idxs]
            self.y = self.y[idxs]

        # self.lock = threading.Lock()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        # with self.lock:
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_x = self.tokenizer.texts_to_sequences(batch_x)
        batch_x = tf.keras.preprocessing.sequence.pad_sequences(batch_x)
        batch_y = label_text_to_num(batch_y)
        return batch_x, batch_y
    
def gen(data, tokenizer):
    """
    Generator for preprocessing training data at batch level. Here we translate text in sequences, labels in numbers,
    and we apply padding to sequences with lenght less the the longest one in the batch.
    :param data: class Dataset with loaded training bins
    """
    # while(True):
    for X, y in data:
        X = tokenizer.texts_to_sequences(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(X)
        y = label_text_to_num(y)
        yield X, y

def preprocess_data(X, y, tokenizer, maxlen, hot_encoded=False, num_classes=None):
   """
    preprocess validation data converting text into number sequences and char labels in binary ones
    :param X: unpreprocessed features data
    :param y: unpreprocessed label of validation data
    :param tokenizer: tokenizer already fitted with all the training set
    :return: preprocessed validation data
    """
   X = tokenizer.texts_to_sequences(X)
   X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=maxlen, padding='post')
   if(hot_encoded):
    X = to_categorical(X, num_classes=21)
   y = label_text_to_num(y)

   return X, y

def label_text_to_num(y):
    """
    Function that converts array of chars in corresponding binary label
    :param y: array of labels
    :return: array of binary labels
    """
    f = lambda x: 0 if x == 'N' else 1
    return np.array([f(y[i]) for i in range(len(y))])

def plot_history(dir, history, name='loss.png'):
    """
    Function that plots loss history and if present validation
    :param dir: directory where to save plot
    :history: history containing loss/losses data/s
    """
    loss = history.history['loss']
    # val_loss = history.history['val_loss']
    val_loss = history.history.get('val_loss', None)
    epochs = range(0, len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    if (val_loss is not None):
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
    else:
        plt.title('Training loss')
    plt.legend()
    plt.savefig(os.path.join(dir, name))

def load_parameters(file_path="parameters.json"):
    """
    It reads parameters from a JSON file and convertes them to a Python dictionary
    :param file_path:
    :return: a dictionary containing parameters
    """

    with open(file_path, "r") as fp:
        return json.load(fp)