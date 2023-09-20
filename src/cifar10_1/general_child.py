from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf
import keras
from src.cifar10_1.models import Model
from src.cifar10_1.image_ops import batch_norm

from src.utils import count_model_params
from src.utils import get_train_ops
from src.common_ops import create_weight
from tensorflow.keras.layers import Input


# class GeneralChild(Model):
class GeneralChild(tf.keras.Model):
  def __init__(self,
               images,
               labels,
               data_format="NCHW",
               batch_size=128,
               keep_prob=1.0,
               cutout_size=None,
               whole_channels=False,
               fixed_arc=None,
               out_filters_scale=1,
               num_layers=2,
               num_branches=6,
               out_filters=24,
               name="child",
               *args,
               **kwargs
              ):

    super(GeneralChild, self).__init__(
      name=name)

    self.out_filters = out_filters * out_filters_scale
    self.num_layers = num_layers

    self.num_branches = num_branches
    self.fixed_arc = fixed_arc
    self.out_filters_scale = out_filters_scale

    pool_distance = self.num_layers // 3
    self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]

    self.images = images
    self.labels = labels
    self.data_format = data_format
    self.batch_size = batch_size
    self.keep_prob = keep_prob
    # self.input_images = Input(tensor=input_images)
    # self.input_labels = Input(tensor=input_labels)

  def _factorized_reduction(self, x, out_filters, stride, is_training):
    """Reduces the shape of x without information loss due to striding."""
    assert out_filters % 2 == 0, (
        "Need even number of filters when using this factorized reduction.")
    if stride == 1:
      # with tf.variable_scope("path_conv"):
      inp_c = x.shape[1]
      print("inp_c hereeee", inp_c)
      w = create_weight("w", [1, 1, inp_c, out_filters])
      x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                        data_format=self.data_format)
      x = batch_norm(x, is_training, data_format=self.data_format)
      return x

  def _model(self, input_images, is_training):
      '''
      Args:
        images: dict with keys {"train", "valid", "test"}.
        is_training: bool tensor.
      '''
      layers = []
      # self.data_format == "NCHW" else 'channels_last'
      out_filters = self.out_filters
      conv2d = tf.keras.layers.Conv2D(filters=out_filters, kernel_size=3, strides=1, padding="same",
                                      data_format='channels_last')(input_images)
      x = tf.keras.layers.BatchNormalization()(conv2d)
      x = tf.keras.layers.ReLU()(x)
      layers.append(x)

      start_idx = 0
      for layer_id in range(self.num_layers):
          if self.fixed_arc is None:
              x = self._enas_layer(layer_id, layers, start_idx, out_filters, is_training)
          else:
              x = self._fixed_layer(layer_id, layers, start_idx, out_filters, is_training)
          layers.append(x)
          
          start_idx += 1 + layer_id
          print(layers[-1])

      x = tf.keras.layers.GlobalAveragePooling2D(data_format=('channels_first' if self.data_format == "NCHW" else 'channels_last'))(x)

      if is_training:
          x = tf.keras.layers.Dropout(1 - self.keep_prob)(x)
    
      dense_layer = tf.keras.layers.Dense(10)
      x = dense_layer(x)
      return x

  def _enas_layer(self, layer_id, prev_layers, start_idx, out_filters, is_training):
      """
      Args:
        layer_id: current layer
        prev_layers: cache of previous layers. for skip connections
        start_idx: where to start looking at.
        is_training: for batch_norm
      """

      inputs = prev_layers[-1]
      inp_c = inputs.shape[1]
      inp_h = inputs.shape[2]
      inp_w = inputs.shape[3]

      count = self.sample_arc[start_idx]

      if count == 0:
          y = self._conv_branch(inputs, 3, is_training, out_filters, out_filters, start_idx=0)
      elif count == 1:
          y = self._empty_layer(inputs, is_training, out_filters, "empty", start_idx=0)
      elif count == 6:
          y = self._pool_branch(inputs, is_training, out_filters, "max", start_idx=0)
      else:
          if self.data_format == "NHWC":
              y = tf.zeros([self.batch_size, inp_h, inp_w, out_filters])
          else: # "NCHW"
              y = tf.zeros([self.batch_size, out_filters, inp_h, inp_w])
      
      out = y
      
      return out


  def _fixed_layer(
      self, layer_id, prev_layers, start_idx, out_filters, is_training):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
      is_training: for batch_norm
    """

    inputs = prev_layers[-1]
    # if self.whole_channels:
    self.data_format == "NCHW"
    inp_c = inputs.get_shape()[1].value
    actual_data_format = "channels_first"

    count = self.sample_arc[start_idx]
    if count == 0:#Empty Layer
      out = tf.identity(inputs)
    elif count in [1,2,3,4]: # depthwise separable layers
      filter_sizes = {2: 3, 4: 5}
      filter_size = filter_sizes.get(count)
      # Conv 1x1
      conv2d = tf.keras.layers.Conv2D(filters=out_filters, kernel_size=1, strides=1, padding="same",
                                      data_format='channels_last')(inputs)
      x = tf.keras.layers.BatchNormalization()(conv2d)
      x = tf.keras.layers.ReLU()(x)

      depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same",
                                                  depthwise_initializer='glorot_uniform',
                                                  data_format='channels_last')(inputs)
      x = tf.keras.layers.BatchNormalization()(depthwise_conv)
      out = tf.keras.layers.ReLU()(x)
    elif count == 5:
      # Conv 1x1
      conv2d = tf.keras.layers.Conv2D(filters=self.out_filters, kernel_size=1, strides=1, padding="same",
                                      data_format='channels_last')(inputs)
      x = tf.keras.layers.BatchNormalization()(conv2d)
      x = tf.keras.layers.ReLU()(x)
      # Pooling  
      out = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='SAME',
                                        data_format='channels_last')(x)

    else:
      raise ValueError("Unknown operation number '{0}'".format(count))

    return out

  def _conv_branch(self, inputs, filter_size, is_training, count, out_filters,
                  ch_mul=1, start_idx=None, separable=False):
      """
      Args:
        start_idx: where to start taking the output channels. if None, assuming
          fixed_arc mode
        count: how many output_channels to take.
      """

      if self.data_format == "NHWC":
          inp_c = inputs.shape[3]
      elif self.data_format == "NCHW":
          inp_c = inputs.shape[1]

      # Define data_format for keras layers
      actual_data_format = 'channels_last' if self.data_format == "NHWC" else 'channels_first'

      # Conv 1x1
      x = tf.keras.layers.Conv2D(filters=out_filters, kernel_size=1, strides=1, padding="same",
                                data_format='channels_last')(inputs)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.ReLU()(x)

      # Depthwise Convolution
      x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same",
                                          depthwise_initializer='glorot_uniform',
                                          data_format=actual_data_format)(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.ReLU()(x)

      return x


  def _pool_branch(self, inputs, is_training, count, avg_or_max, start_idx=None):
      """
      Args:
        start_idx: where to start taking the output channels. if None, assuming
          fixed_arc mode
        count: how many output_channels to take.
      """

      # Check if the architecture is fixed
      if start_idx is None:
          assert self.fixed_arc is not None, "you screwed up!"

      # Get the input channels
      if self.data_format == "NHWC":
          inp_c = inputs.shape[3]
      elif self.data_format == "NCHW":
          inp_c = inputs.shape[1]

      # Define data_format for keras layers
      actual_data_format = 'channels_last' if self.data_format == "NHWC" else 'channels_first'

      # Conv 1x1
      x = tf.keras.layers.Conv2D(filters=self.out_filters, kernel_size=1, strides=1, padding="same",
                                data_format='channels_last')(inputs)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.ReLU()(x)

      # Apply MaxPooling
      if avg_or_max == "max":
          x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='SAME',
                                          data_format=actual_data_format)(x)

      # Take specific output channels if start_idx is provided
      if start_idx is not None:
          if self.data_format == "NHWC":
              x = x[:, :, :, start_idx: start_idx+count]
          elif self.data_format == "NCHW":
              x = x[:, start_idx: start_idx+count, :, :]

      return x


  def _empty_layer(self, inputs, is_training, count, empty, start_idx=None):
      """
      EMPTY LAYER: Return a Tensor with the same shape and contents as input. 
      """

      # Check if architecture is fixed
      if start_idx is None:
          assert self.fixed_arc is not None, "you screwed up!"

      # Apply 1x1 MaxPooling which does nothing
      x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding="SAME",
                                      data_format='channels_last')(inputs)

      # Take specific output channels if start_idx is provided
      if start_idx is not None:
          if self.data_format == "NHWC":
              x = x[:, :, :, start_idx: start_idx+count]
          elif self.data_format == "NCHW":
              x = x[:, start_idx: start_idx+count, :, :]

      return x



  # override
  def _build_train(self):
    print("-" * 80)
    print("Build train graph")
    self.x_train = self.images['train']
    self.y_train = self.labels['train']
    # Accessing the input tensors
    batch_size = 128
    self.x_train = tf.data.Dataset.from_tensor_slices((self.images['train'], self.labels['train']))
    for images_batch, labels_batch in self.x_train.batch(batch_size):
      logits = self._model(images_batch, is_training=True)
      # compute loss, gradients, and update model weights...


    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels_batch)
    self.loss = tf.reduce_mean(log_probs)

    self.train_preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    self.train_acc = tf.reduce_sum(tf.cast(tf.equal(self.train_preds, self.y_train), tf.int32))

    # Get the number of trainable variables (parameters) for this model.
    model_trainable_variables = self._model.trainable_variables
    self.num_vars = np.sum([np.prod(v.get_shape().as_list()) for v in model_trainable_variables])
    print("Model has {} params".format(self.num_vars))


    self.global_step = tf.Variable(
      0, dtype=tf.int32, trainable=False, name="global_step")
    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.global_step,
      clip_mode=self.clip_mode,
      grad_bound=self.grad_bound,
      l2_reg=self.l2_reg,
      lr_init=self.lr_init,
      lr_dec_start=self.lr_dec_start,
      lr_dec_every=self.lr_dec_every,
      lr_dec_rate=self.lr_dec_rate,
      lr_cosine=self.lr_cosine,
      lr_max=self.lr_max,
      lr_min=self.lr_min,
      lr_T_0=self.lr_T_0,
      lr_T_mul=self.lr_T_mul,
      num_train_batches=self.num_train_batches,
      optim_algo=self.optim_algo,
      sync_replicas=self.sync_replicas,
      num_aggregate=self.num_aggregate,
      num_replicas=self.num_replicas)

  # override
  def _build_valid(self):
      if self.x_valid is not None:
          print("-" * 80)
          print("Build valid graph")
          logits = self._model(self.x_valid, is_training=False)
          self.valid_preds = tf.argmax(logits, axis=1, output_type=tf.int32)
          self.valid_acc = tf.reduce_sum(tf.cast(tf.equal(self.valid_preds, self.y_valid), tf.int32))

  def _build_test(self):
      print("-" * 80)
      print("Build test graph")
      logits = self._model(self.x_test, is_training=False)
      self.test_preds = tf.argmax(logits, axis=1, output_type=tf.int32)
      self.test_acc = tf.reduce_sum(tf.cast(tf.equal(self.test_preds, self.y_test), tf.int32))


  # override
  def build_valid_rl(self, shuffle=False):
      print("-" * 80)
      print("Build valid graph on shuffled data")
      
      # Shuffle and batch data using tf.data
      data = tf.data.Dataset.from_tensor_slices((self.images["valid_original"], self.labels["valid_original"]))
      
      if shuffle:
          data = data.shuffle(buffer_size=25000, seed=self.seed)
      
      data = data.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

      # Preprocess function
      def _pre_process(x, y):
          x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
          x = tf.image.random_crop(x, [32, 32, 3], seed=self.seed)
          x = tf.image.random_flip_left_right(x, seed=self.seed)
          if self.data_format == "NCHW":
              x = tf.transpose(x, [2, 0, 1])
          return x, y

      if shuffle:
          data = data.map(_pre_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      
      x_valid_shuffle, y_valid_shuffle = next(iter(data))
      
      logits = self._model(x_valid_shuffle, is_training=False)
      valid_shuffle_preds = tf.argmax(logits, axis=1, output_type=tf.int32)
      self.valid_shuffle_acc = tf.reduce_sum(tf.cast(tf.equal(valid_shuffle_preds, y_valid_shuffle), tf.int32))


  def connect_controller(self, controller_model):
      if self.fixed_arc is None:
          self.sample_arc = controller_model.sample_arc
          # self.sampled_fr = controller_model.sampled_fr
      else:
          fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
          self.sample_arc = fixed_arc

      self._build_train()
      self._build_valid()
      self._build_test()


