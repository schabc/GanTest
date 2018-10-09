from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time
import functools
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

# Main TFGAN library.
tfgan = tf.contrib.gan

# TFGAN MNIST examples from `tensorflow/models`.
from mnist import data_provider
from mnist import util

# TF-Slim data provider.
from datasets import download_and_convert_mnist

# Shortcuts for later.
queues = tf.contrib.slim.queues
layers = tf.contrib.layers
ds = tf.contrib.distributions
framework = tf.contrib.framework

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)
  

def visualize_training_generator(train_step_num, start_time, data_np):
    """Visualize generator outputs during training.
    
    Args:
        train_step_num: The training step number. A python integer.
        start_time: Time when training started. The output of `time.time()`. A
            python float.
        data: Data to plot. A numpy array, most likely from an evaluated TensorFlow
            tensor.
    """
    print('Training step: %i' % train_step_num)
    time_since_start = (time.time() - start_time) / 60.0
    print('Time since start: %f m' % time_since_start)
    print('Steps per min: %f' % (train_step_num / time_since_start))
    plt.axis('off')
    plt.imshow(np.squeeze(data_np), cmap='gray')
    plt.show()

def visualize_digits(tensor_to_visualize):
    """Visualize an image once. Used to visualize generator before training.
    
    Args:
        tensor_to_visualize: An image tensor to visualize. A python Tensor.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with queues.QueueRunners(sess):
            images_np = sess.run(tensor_to_visualize)
    plt.axis('off')
    plt.imshow(np.squeeze(images_np), cmap='gray')

def evaluate_tfgan_loss(gan_loss, name=None):
    """Evaluate GAN losses. Used to check that the graph is correct.
    
    Args:
        gan_loss: A GANLoss tuple.
        name: Optional. If present, append to debug output.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with queues.QueueRunners(sess):
            gen_loss_np = sess.run(gan_loss.generator_loss)
            dis_loss_np = sess.run(gan_loss.discriminator_loss)
    if name:
        print('%s generator loss: %f' % (name, gen_loss_np))
        print('%s discriminator loss: %f'% (name, dis_loss_np))
    else:
        print('Generator loss: %f' % gen_loss_np)
        print('Discriminator loss: %f'% dis_loss_np)

MNIST_DATA_DIR = '/tmp/mnist-data'

if not tf.gfile.Exists(MNIST_DATA_DIR):
    tf.gfile.MakeDirs(MNIST_DATA_DIR)

download_and_convert_mnist.run(MNIST_DATA_DIR)

tf.reset_default_graph()

def _get_train_input_fn(batch_size, noise_dims):
    def train_input_fn():
        with tf.device('/cpu:0'):
            real_images, _, _ = data_provider.provide_data(
                'train', batch_size, MNIST_DATA_DIR)
        noise = tf.random_normal([batch_size, noise_dims])
        return noise, real_images
    return train_input_fn


def _get_predict_input_fn(batch_size, noise_dims):
    def predict_input_fn():
        noise = tf.random_normal([batch_size, noise_dims])
        return noise
    return predict_input_fn


BATCH_SIZE = 32
NOISE_DIMS = 64
NUM_STEPS = 2000

# Initialize GANEstimator with options and hyperparameters.
gan_estimator = tfgan.estimator.GANEstimator(
    generator_fn=generator_fn,
    discriminator_fn=discriminator_fn,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    generator_optimizer=tf.train.AdamOptimizer(0.001, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5),
    add_summaries=tfgan.estimator.SummaryType.IMAGES)

# Train estimator.
train_input_fn = _get_train_input_fn(BATCH_SIZE, NOISE_DIMS)
start_time = time.time()
gan_estimator.train(train_input_fn, max_steps=NUM_STEPS)
time_since_start = (time.time() - start_time) / 60.0
print('Time since start: %f m' % time_since_start)
print('Steps per min: %f' % (NUM_STEPS / time_since_start))

def _get_next(iterable):
    try:
        return iterable.next()  # Python 2.x.x
    except AttributeError:
        return iterable.__next__()  # Python 3.x.x

# Run inference.
predict_input_fn = _get_predict_input_fn(36, NOISE_DIMS)
prediction_iterable = gan_estimator.predict(
    predict_input_fn, hooks=[tf.train.StopAtStepHook(last_step=1)])
predictions = [_get_next(prediction_iterable) for _ in xrange(36)]

try: # Close the predict session.
    _get_next(prediction_iterable)
except StopIteration:
    pass

# Nicely tile output and visualize.
image_rows = [np.concatenate(predictions[i:i+6], axis=0) for i in
              range(0, 36, 6)]
tiled_images = np.concatenate(image_rows, axis=1)

# Visualize.
plt.axis('off')
plt.imshow(np.squeeze(tiled_images), cmap='gray')
