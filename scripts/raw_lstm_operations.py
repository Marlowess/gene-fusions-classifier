#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import sys
import os

os.system('clear')

# Variables for demo.

num_row = 30 # h
num_col = 15 # d
num_samples = 10 # n

# Weights Matrices random initiated.
W_i = np.random.rand(num_row, num_row)
U_i = np.random.rand(num_row, num_col)

W_f = np.random.rand(num_row, num_row)
U_f = np.random.rand(num_row, num_col)

W_o = np.random.rand(num_row, num_row)
U_o = np.random.rand(num_row, num_col)

W_g = np.random.rand(num_row, num_row)
U_g = np.random.rand(num_row, num_col)


# Hidden states and bias initiated with arrays of zeros.
C_prev = np.zeros((1, num_row), dtype=np.float)

b_i = np.zeros((1, num_row), dtype=np.float)
b_f = np.zeros((1, num_row), dtype=np.float)
b_o = np.zeros((1, num_row), dtype=np.float)
b_c = np.zeros((1, num_row), dtype=np.float)

def cell_lstm(x_t, h_t):

    global C_prev

    sigmoid_local = tf.keras.activations.sigmoid
    tanh_local = tf.keras.activations.tanh

    x_t_local = np.transpose(x_t)
    h_t_local = np.transpose(h_t)

    i = sigmoid_local(np.dot(U_i, x_t_local) + np.dot(W_i, h_t_local) + np.transpose(b_i))
    f = sigmoid_local(np.dot(U_f, x_t_local) + np.dot(W_f, h_t_local) + np.transpose(b_f))
    o = sigmoid_local(np.dot(U_o, x_t_local) + np.dot(W_o, h_t_local) + np.transpose(b_o))
    
    c_prev_local = np.transpose(C_prev)
    c = np.multiply(f, c_prev_local) + np.multiply(i, tanh_local(np.dot(U_g, x_t_local) + np.dot(W_g, h_t_local) + np.transpose(b_c)))

    h = np.multiply(o, tanh_local(c))
    h_t = np.transpose(h)

    C_prev = np.transpose(c)
    
    return h_t

h_t = np.zeros((1, num_row), dtype=np.float)
samples = np.asarray(list(map(lambda _: np.random.rand(1, num_col), range(num_samples))))

for ii, x_t in enumerate(samples):
    h_t = cell_lstm(x_t, h_t)
    print(f" [*] ({ii})")
    print(f"{h_t}\r")

# x_t = np.random.rand(1, num_col)
# h_t = np.random.rand(1, num_col)

# h_t = cell_lstm(x_t, h_t)
# print(h_t.shape)
# print(h_t)


# x_t = np.random.rand(1, num_col)
# h_t = cell_lstm(x_t, h_t)
# print(h_t.shape)
# print(h_t)