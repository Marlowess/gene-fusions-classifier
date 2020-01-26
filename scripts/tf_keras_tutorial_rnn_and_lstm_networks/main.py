#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

from models import *

import os

if __name__ == "__main__":

    base_dir_images: str = './figures/'
    try:
        os.makedirs(base_dir_images)
    except:
        pass

    get_example_gru_model_sequential(return_sequences_flag = True, plot_model_flag=True, dir_path=base_dir_images)
    get_example_lstm_model_sequential(plot_model_flag=True, dir_path=base_dir_images)
    get_bidirectional_lstm_model_sequential(plot_model_flag=True, dir_path=base_dir_images)

    model = get_mnist_classifier_made_from_lstm_model(
        allow_cudnn_kernel=True,
        plot_model_flag=True,
        dir_path=base_dir_images,
        filename='example_mnist_classifier_lstm_model_sequential.png')
    slow_model = get_mnist_classifier_made_from_lstm_model(
        allow_cudnn_kernel=False,
        plot_model_flag=True,
        dir_path=base_dir_images,
        filename='example_mnist_classifier_lstm_model_sequential_slow.png')

    x_train, y_train, x_test, y_test, sample, sample_label = load_mnist_dataset()
    units, input_dim, output_size = 64, 28, 10
    with tf.device('CPU:0'):
        cpu_model = build_model(units, input_dim, output_size, allow_cudnn_kernel=True)
        cpu_model.set_weights(model.get_weights())
        result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
        print('Predicted result is: %s, target result is: %s' % (result.numpy(), sample_label))
        plt.imshow(sample, cmap=plt.get_cmap('gray'))
        pass


    pass