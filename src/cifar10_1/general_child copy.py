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


# class GeneralChild(Model):
class GeneralChild(Model):
  def __init__(self,
               images,
               labels,
               cutout_size=None,
               whole_channels=False,
               fixed_arc=None,
               out_filters_scale=1,
               num_layers=2,
               num_branches=6,
               out_filters=24,
               keep_prob=1.0,
               batch_size=32,
               clip_mode=None,
               grad_bound=None,
               l2_reg=1e-4,
               lr_init=0.1,
               lr_dec_start=0,
               lr_dec_every=10000,
               lr_dec_rate=0.1,
               lr_cosine=False,
               lr_max=None,
               lr_min=None,
               lr_T_0=None,
               lr_T_mul=None,
               optim_algo=None,
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               data_format="NCHW",
               name="child",
               *args,
               **kwargs
              ):
    """
    """

    super(self.__class__, self).__init__(
      images,
      labels,
      cutout_size=cutout_size,
      batch_size=batch_size,
      clip_mode=clip_mode,
      grad_bound=grad_bound,
      l2_reg=l2_reg,
      lr_init=lr_init,
      lr_dec_start=lr_dec_start,
      lr_dec_every=lr_dec_every,
      lr_dec_rate=lr_dec_rate,
      keep_prob=keep_prob,
      optim_algo=optim_algo,
      sync_replicas=sync_replicas,
      num_aggregate=num_aggregate,
      num_replicas=num_replicas,
      data_format=data_format,
      name=name)

    self.whole_channels = whole_channels
    self.lr_cosine = lr_cosine
    self.lr_max = lr_max
    self.lr_min = lr_min
    self.lr_T_0 = lr_T_0
    self.lr_T_mul = lr_T_mul
    self.out_filters = out_filters * out_filters_scale
    self.num_layers = num_layers

    self.num_branches = num_branches
    self.fixed_arc = fixed_arc
    self.out_filters_scale = out_filters_scale

    pool_distance = self.num_layers // 3
    self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]

    self.images = images
    self.labels = labels


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

  def _model(self, images, is_training, reuse=False):
    '''
    Args:
      images: dict with keys {"train", "valid", "test"}.
      is_training: bool tensor.
    '''
    layers = []

    out_filters = self.out_filters
    conv2d = tf.keras.layers.Conv2D(filters=out_filters, kernel_size=3, strides=1, padding="same",
                                      data_format=('channels_first' if self.data_format == "NCHW" else 'channels_last'))(images)
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
      x = tf.nn.dropout(x, self.keep_prob)
  
    self.data_format == "NCHW"
    dense_layer = tf.keras.layers.Dense(10)
    x = dense_layer(x)
    return x

  def _enas_layer(self, layer_id, prev_layers, start_idx, out_filters, is_training):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
      is_training: for batch_norm
    """

    inputs = prev_layers[-1]
    self.data_format == "NCHW"
    inp_c = inputs.shape[1]
    inp_h = inputs.shape[2]
    inp_w = inputs.shape[3]

    count = self.sample_arc[start_idx]
    branches = {}
    y = self._conv_branch(inputs, 3, is_training, out_filters, out_filters,start_idx=0)
    branches[tf.equal(count, 0)] = lambda: y
    # Empty branch ###
    y = self._empty_layer(inputs, is_training, out_filters, "empty",start_idx=0)
    branches[tf.equal(count, 1)] = lambda: y
    y = self._pool_branch(inputs, is_training, out_filters, "max",start_idx=0)
    branches[tf.equal(count, 6)] = lambda: y

    out = tf.case(branches, default=lambda: tf.constant(0, tf.float32, shape=[self.batch_size, out_filters, inp_h, inp_w]), exclusive=True)

    if self.data_format == "NHWC":
      out.set_shape([None, inp_h, inp_w, out_filters])
    elif self.data_format == "NCHW":
      out.set_shape([None, out_filters, inp_h, inp_w])

    if layer_id > 0:
      # if self.whole_channels:
      skip_start = start_idx + 1
      skip = self.sample_arc[skip_start: skip_start + layer_id]
      # with tf.variable_scope("skip"):
      res_layers = []
      for i in range(layer_id):
        res_layers.append(tf.cond(tf.equal(skip[i], 1),
                                  lambda: prev_layers[i],
                                  lambda: tf.zeros_like(prev_layers[i])))
      res_layers.append(out)
      out = tf.add_n(res_layers)
      batch_norm_layer = tf.keras.layers.BatchNormalization(axis=1)
      out = batch_norm_layer(out, training=is_training)
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
                                      data_format=('channels_first' if self.data_format == "NCHW" else 'channels_last'))(inputs)
      x = tf.keras.layers.BatchNormalization()(conv2d)
      x = tf.keras.layers.ReLU()(x)

      depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same",
                                                  depthwise_initializer='glorot_uniform',
                                                  data_format=('channels_first' if self.data_format == "NCHW" else 'channels_last'))(inputs)
      x = tf.keras.layers.BatchNormalization()(depthwise_conv)
      out = tf.keras.layers.ReLU()(x)
    elif count == 5:
      # Conv 1x1
      conv2d = tf.keras.layers.Conv2D(filters=self.out_filters, kernel_size=1, strides=1, padding="same",
                                      data_format=('channels_first' if self.data_format == "NCHW" else 'channels_last'))(inputs)
      x = tf.keras.layers.BatchNormalization()(conv2d)
      x = tf.keras.layers.ReLU()(x)
      # Pooling  
      out = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='SAME',
                                        data_format=('channels_first' if actual_data_format == 'NCHW' else 'channels_last'))(x)

    else:
      raise ValueError("Unknown operation number '{0}'".format(count))

    if layer_id > 0:
      skip_start = start_idx + 1
      skip = self.sample_arc[skip_start: skip_start + layer_id]
      total_skip_channels = np.sum(skip) + 1
      res_layers = []
      for i in range(layer_id):
        if skip[i] == 1:
          res_layers.append(prev_layers[i])
      prev = res_layers + [out]

      if self.data_format == "NHWC":
        prev = tf.concat(prev, axis=3)
      elif self.data_format == "NCHW":
        prev = tf.concat(prev, axis=1)
      out = prev
      conv2d = tf.keras.layers.Conv2D(filters=self.out_filters, kernel_size=1, strides=1, padding="same",
                                      data_format=('channels_first' if self.data_format == "NCHW" else 'channels_last'))(inputs)
      x = tf.keras.layers.BatchNormalization()(conv2d)
      x = tf.keras.layers.ReLU()(x)

    return out

  def _conv_branch(self, inputs, filter_size, is_training, count, out_filters,
                   ch_mul=1, start_idx=None, separable=False):
    """
    Args:
      start_idx: where to start taking the output channels. if None, assuming
        fixed_arc mode
      count: how many output_channels to take.
    """

    self.data_format == "NCHW"
    actual_data_format = "channels_first"
    if start_idx is None:
      assert self.fixed_arc is not None, "you screwed up!"
    if self.data_format == "NHWC":
      inp_c = inputs.get_shape()[3].value
    elif self.data_format == "NCHW":
      inp_c = inputs.get_shape()[1].value
    # Conv 1x1
    conv2d = tf.keras.layers.Conv2D(filters=out_filters, kernel_size=1, strides=1, padding="same",
                                    data_format=('channels_first' if self.data_format == "NCHW" else 'channels_last'))(inputs)
    x = tf.keras.layers.BatchNormalization()(conv2d)
    x = tf.keras.layers.ReLU()(x)
    #if start_idx is None:
    depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same",
                                                 depthwise_initializer='glorot_uniform',
                                                 data_format=('channels_first' if self.data_format == "NCHW" else 'channels_last'))(inputs)
    x = tf.keras.layers.BatchNormalization()(depthwise_conv)
    x = tf.keras.layers.ReLU()(x)
    #else:
      
    return x

  def _pool_branch(self, inputs, is_training, count, avg_or_max, start_idx=None):
    """
    Args:
      start_idx: where to start taking the output channels. if None, assuming
        fixed_arc mode
      count: how many output_channels to take.
    """

    if start_idx is None:
      assert self.fixed_arc is not None, "you screwed up!"

    if self.data_format == "NHWC":
      inp_c = inputs.get_shape()[3].value
    elif self.data_format == "NCHW":
      inp_c = inputs.get_shape()[1].value
    self.data_format == "NCHW"
    actual_data_format = "channels_first"
    avg_or_max == "max"

    # Conv 1x1
    conv2d = tf.keras.layers.Conv2D(filters=self.out_filters, kernel_size=1, strides=1, padding="same",
                                    data_format=('channels_first' if self.data_format == "NCHW" else 'channels_last'))(inputs)
    x = tf.keras.layers.BatchNormalization()(conv2d)
    x = tf.keras.layers.ReLU()(x)
    # Pooling  
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='SAME',
                                      data_format=('channels_first' if actual_data_format == 'NCHW' else 'channels_last'))(x)

    if start_idx is not None:
      if self.data_format == "NHWC":
        x = x[:, :, :, start_idx : start_idx+count]
      elif self.data_format == "NCHW":
        x = x[:, start_idx : start_idx+count, :, :]

    return x

  def _empty_layer(self, inputs, is_training, count, empty, start_idx=None):
    """
    EMPTY LAYER: Return a Tensor with the same shape and contents as input. 
    """
    if start_idx is None:
      assert self.fixed_arc is not None, "you screwed up!"
    if self.data_format == "NHWC":
      inp_c = inputs.get_shape()[3].value
    elif self.data_format == "NCHW":
      inp_c = inputs.get_shape()[1].value
    # with tf.variable_scope("empty"):
      #x = tf.identity(inputs)    
    # x = tf.layers.max_pooling2d(inputs, [1, 1], [1, 1], "SAME")
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding="SAME")(inputs)


    if start_idx is not None:
      if self.data_format == "NHWC":
        x = x[:, :, :, start_idx : start_idx+count]
      elif self.data_format == "NCHW":
        x = x[:, start_idx : start_idx+count, :, :]
    return x


  # override
  def _build_train(self):
    print("-" * 80)
    print("Build train graph")
    self.x_train = self.images['train']
    self.y_train = self.labels['train']
    self.y_train = tf.squeeze(self.y_train, axis=-1) # remove dim 1
    self.y_train = tf.cast(self.y_train, tf.int32)

    logits = self._model(self.x_train, is_training=True)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_train)
    self.loss = tf.reduce_mean(log_probs)

    self.train_preds = tf.argmax(logits, axis=1)
    self.train_preds = tf.cast(self.train_preds, tf.int32)
    self.train_acc = tf.equal(self.train_preds, self.y_train)
    self.train_acc = tf.cast(self.train_acc, tf.int32)
    self.train_acc = tf.reduce_sum(self.train_acc)

    tf_variables = [var for var in tf.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
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
      logits = self._model(self.x_valid, False, reuse=True)
      self.valid_preds = tf.argmax(logits, axis=1)
      self.valid_preds = tf.to_int32(self.valid_preds)
      self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
      self.valid_acc = tf.to_int32(self.valid_acc)
      self.valid_acc = tf.reduce_sum(self.valid_acc)

  # override
  def _build_test(self):
    print("-" * 80)
    print("Build test graph")
    logits = self._model(self.x_test, False, reuse=True)
    self.test_preds = tf.argmax(logits, axis=1)
    self.test_preds = tf.to_int32(self.test_preds)
    self.test_acc = tf.equal(self.test_preds, self.y_test)
    self.test_acc = tf.to_int32(self.test_acc)
    self.test_acc = tf.reduce_sum(self.test_acc)

  # override
  def build_valid_rl(self, shuffle=False):
    print("-" * 80)
    print("Build valid graph on shuffled data")
    with tf.device("/cpu:0"):
      # shuffled valid data: for choosing validation model
      if not shuffle and self.data_format == "NCHW":
        self.images["valid_original"] = np.transpose(
          self.images["valid_original"], [0, 3, 1, 2])
      x_valid_shuffle, y_valid_shuffle = tf.train.shuffle_batch(
        [self.images["valid_original"], self.labels["valid_original"]],
        batch_size=self.batch_size,
        capacity=25000,
        enqueue_many=True,
        min_after_dequeue=0,
        num_threads=16,
        seed=self.seed,
        allow_smaller_final_batch=True,
      )

      def _pre_process(x):
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
        x = tf.random_crop(x, [32, 32, 3], seed=self.seed)
        x = tf.image.random_flip_left_right(x, seed=self.seed)
        if self.data_format == "NCHW":
          x = tf.transpose(x, [2, 0, 1])

        return x

      if shuffle:
        x_valid_shuffle = tf.map_fn(
          _pre_process, x_valid_shuffle, back_prop=False)

    logits = self._model(x_valid_shuffle, False, reuse=True)
    valid_shuffle_preds = tf.argmax(logits, axis=1)
    valid_shuffle_preds = tf.to_int32(valid_shuffle_preds)
    self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, y_valid_shuffle)
    self.valid_shuffle_acc = tf.to_int32(self.valid_shuffle_acc)
    self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)

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

