import tensorflow as tf
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def reconstruct(data):

  data += tf.random_normal(tf.shape(data))

  conv1 = tf.layers.conv2d(data, 32, 4, 2, activation=tf.nn.relu, padding="SAME")
  conv2 = tf.layers.conv2d(conv1, 16, 4, 2, activation=tf.nn.relu, padding="SAME")
  conv3 = tf.layers.conv2d(conv2, 8, 4, 2, activation=tf.nn.relu, padding="SAME")

  # 32 x 32 x 1 -> 16 x 16 x 32
  # 16 x 16 x 32 -> 8 x 8 x 16
  # 8 x 8 x 16 -> 4 x 4 x 8

  conv4 = tf.layers.conv2d_transpose(conv3, 16, 4, 2, activation=tf.nn.relu, padding="SAME")
  conv5 = tf.layers.conv2d_transpose(conv4, 32, 4, 2, activation=tf.nn.relu, padding="SAME")
  final = tf.layers.conv2d_transpose(conv5, 1, 4, 2, activation=tf.nn.relu, padding="SAME")

  return final

def resize_images(images):

  # Just it case it isn't in this form yet, reshape the tensor.
  images = images.reshape((-1, 28, 28, 1))

  # Initialize a tensor full of zeroes to hold the correct resized tensor
  resized_images = np.zeros((images.shape[0], 32, 32, 1))

  # For each image in the batch we have,
  for i in range(images.shape[0]):

    # find the correct slot in the resultant batch, and store the resized image there.
    resized_images[i, ..., 0] = transform.resize(images[i, ..., 0], (32, 32))

  return resized_images

inputs = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
rec_images = reconstruct(inputs)
loss = tf.reduce_mean(tf.square(rec_images - inputs))
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 300
epochs = 7
num_batches = mnist.train.num_examples // batch_size
