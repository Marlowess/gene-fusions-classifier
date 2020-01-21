#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

# ------------------------------- #
# Links                           #
# ------------------------------- #
# - https://www.tensorflow.org/guide/data#time_series_windowing
# - https://www.tensorflow.org/api_docs/python/tf/data/Dataset

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf

np.set_printoptions(precision=4)

dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
print(" [*] Dataset object created:")
print(dataset)

print(" [*] Elements within Dataset Object:")
for elem in dataset:
  print(elem.numpy())

print(" [*] Gets a piece of data within Dataset Object passing through next():")
it = iter(dataset)
print(next(it).numpy())

print(" [*] Sum of element within Dataset Object:")
print(dataset.reduce(0, lambda state, value: state + value).numpy())

print(" [*] Dataset Object element spec:")
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
print(dataset1.element_spec)

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

print(" [*] Dataset Object element spec:")
print(dataset2.element_spec)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(" [*] Dataset Object element spec:")
print(dataset3.element_spec)


# Dataset containing a sparse tensor.
dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

print(dataset4.element_spec)

# Use value_type to see the type of value represented by the element spec
print(dataset4.element_spec.value_type)

dataset1 = tf.data.Dataset.from_tensor_slices(
    tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))

print(dataset1)

for z in dataset1:
  print(z.numpy())

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

print(dataset2)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

print(dataset3)

for a, (b,c) in dataset3:
  print('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))


train, test = tf.keras.datasets.fashion_mnist.load_data()

images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
print(dataset)

def count(stop):
  i = 0
  while i<stop:
    yield i
    i += 1

for n in count(5):
  print(n)

ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )

for count_batch in ds_counter.repeat().batch(10).take(10):
  print(count_batch.numpy())

def gen_series():
  i = 0
  while True:
    size = np.random.randint(0, 10)
    yield i, np.random.normal(size=(size,))
    i += 1

for i, series in gen_series():
  print(i, ":", str(series))
  if i > 5:
    break

ds_series = tf.data.Dataset.from_generator(
    gen_series, 
    output_types=(tf.int32, tf.float32), 
    output_shapes=((), (None,)))

print(ds_series)

shapes = (tf.TensorShape(()), tf.TensorShape([None,]))
ds_series_batch = ds_series.shuffle(20).padded_batch(10, shapes)

ids, sequence_batch = next(iter(ds_series_batch))
print(ids.numpy())
print()
print(sequence_batch.numpy())

flowers = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
