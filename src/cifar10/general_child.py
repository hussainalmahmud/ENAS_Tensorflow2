from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

from src.cifar10.models import Model
from src.cifar10.image_ops import conv
from src.cifar10.image_ops import fully_connected
from src.cifar10.image_ops import batch_norm
from src.cifar10.image_ops import batch_norm_with_mask
from src.cifar10.image_ops import relu
from src.cifar10.image_ops import relu6
from src.cifar10.image_ops import max_pool
from src.cifar10.image_ops import global_avg_pool

from src.utils import count_model_params
from src.utils import get_train_ops
from src.common_ops import create_weight
from src.cifar10 import fr_globals
from src.utils import DEFINE_string
flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_string("child_fr_fixed", None, "provide fr for fixed arc")

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
    print("child_fr_fixed chillddddd", FLAGS.child_fr_fixed)

    # change FLAGS.child_fr_fixed to a list 
    self.child_fr_fixed = FLAGS.child_fr_fixed
    if FLAGS.child_fr_fixed is not None:
      self.child_fr_fixed = list(map(int, FLAGS.child_fr_fixed.split()))


  def _get_C(self, x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if self.data_format == "NHWC":
      return x.get_shape()[3].value
    elif self.data_format == "NCHW":
      return x.get_shape()[1].value
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))

  def _get_HW(self, x): 
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    return x.get_shape()[2].value

  def _get_strides(self, stride):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if self.data_format == "NHWC":
      return [1, stride, stride, 1]
    elif self.data_format == "NCHW":
      return [1, 1, stride, stride]
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))

  def _factorized_reduction(self, x, out_filters, stride, is_training):
    """Reduces the shape of x without information loss due to striding."""
    assert out_filters % 2 == 0, (
        "Need even number of filters when using this factorized reduction.")
    if stride == 1:
      with tf.variable_scope("path_conv"):
        inp_c = self._get_C(x)
        print("inp_c hereeee", inp_c)
        w = create_weight("w", [1, 1, inp_c, out_filters])
        x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                         data_format=self.data_format)
        x = batch_norm(x, is_training, data_format=self.data_format)
        return x

    stride_spec = self._get_strides(stride)
    # Skip path 1
    path1 = tf.nn.avg_pool(
        x, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
    with tf.variable_scope("path1_conv"):
      inp_c = self._get_C(path1)
      w = create_weight("w", [1, 1, inp_c, out_filters // 2])
      path1 = tf.nn.conv2d(path1, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
  
    # Skip path 2
    # First pad with 0"s on the right and bottom, then shift the filter to
    # include those 0"s that were added.
    if self.data_format == "NHWC":
      pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
      path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
      concat_axis = 3
    else:
      pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
      path2 = tf.pad(x, pad_arr)[:, :, 1:, 1:]
      concat_axis = 1
  
    path2 = tf.nn.avg_pool(
        path2, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
    with tf.variable_scope("path2_conv"):
      inp_c = self._get_C(path2)
      w = create_weight("w", [1, 1, inp_c, out_filters // 2])
      path2 = tf.nn.conv2d(path2, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
  
    # Concat and apply BN
    final_path = tf.concat(values=[path1, path2], axis=concat_axis)
    final_path = batch_norm(final_path, is_training,
                            data_format=self.data_format)
    #print("final_path is hereeeee", final_path)
    return final_path


  def _get_C(self, x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if self.data_format == "NHWC":
      return x.get_shape()[3].value
    elif self.data_format == "NCHW":
      return x.get_shape()[1].value
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))

  def _model(self, images, is_training, reuse=False):
    '''
    Args:
      images: dict with keys {"train", "valid", "test"}.
      is_training: bool tensor.
    '''
    with tf.variable_scope(self.name, reuse=reuse):
      layers = []

      out_filters = self.out_filters
      with tf.variable_scope("stem_conv"):
        w = create_weight("w", [3, 3, 3, out_filters])
        x = tf.nn.conv2d(images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        x = batch_norm(x, is_training, data_format=self.data_format)
        layers.append(x)

      # if self.whole_channels:
      start_idx = 0
      for layer_id in range(self.num_layers):
        with tf.variable_scope("layer_{0}".format(layer_id)):
          if self.fixed_arc is None:
            x = self._enas_layer(layer_id, layers, start_idx, out_filters, is_training)
          else:
            x = self._fixed_layer(layer_id, layers, start_idx, out_filters, is_training)
          layers.append(x)
          if self.fixed_arc is None:
            if layer_id in fr_globals.fr_pos:
              with tf.variable_scope("pool_at_{0}".format(layer_id)):
                pooled_layers = []
                for i, layer in enumerate(layers):
                  with tf.variable_scope("from_{0}".format(i)):
                    x = self._factorized_reduction(
                      layer, out_filters, 2, is_training)
                  pooled_layers.append(x)
                layers = pooled_layers
          elif self.fixed_arc is not None:
            if layer_id in self.child_fr_fixed:
              with tf.variable_scope("pool_at_{0}".format(layer_id)):
                pooled_layers = []
                for i, layer in enumerate(layers):
                  with tf.variable_scope("from_{0}".format(i)):
                    x = self._factorized_reduction(
                      layer, out_filters, 2, is_training)
                  pooled_layers.append(x)
                layers = pooled_layers
        start_idx += 1 + layer_id
        print(layers[-1])

      x = global_avg_pool(x, data_format=self.data_format)
      if is_training:
        x = tf.nn.dropout(x, self.keep_prob)
      with tf.variable_scope("fc"):
        if self.data_format == "NWHC":
          inp_c = x.get_shape()[3].value
        elif self.data_format == "NCHW":
          inp_c = x.get_shape()[1].value
        else:
          raise ValueError("Unknown data_format {0}".format(self.data_format))
        w = create_weight("w", [inp_c, 10])
        x = tf.matmul(x, w)
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
    # if self.whole_channels:
    if self.data_format == "NHWC":
      inp_h = inputs.get_shape()[1].value
      inp_w = inputs.get_shape()[2].value
      inp_c = inputs.get_shape()[3].value
    elif self.data_format == "NCHW":
      inp_c = inputs.get_shape()[1].value
      inp_h = inputs.get_shape()[2].value
      inp_w = inputs.get_shape()[3].value

    count = self.sample_arc[start_idx]
    branches = {}
    # Empty branch ###
    with tf.variable_scope("branch_0"):
      y = self._conv_branch(inputs, 3, is_training, out_filters, out_filters,
                            start_idx=0)
      branches[tf.equal(count, 0)] = lambda: y

    with tf.variable_scope("branch_1"):
      y = self._empty_layer(inputs, is_training, out_filters, "empty",
                              start_idx=0)
      branches[tf.equal(count, 1)] = lambda: y
    with tf.variable_scope("branch_2"):
      y = self._conv_branch(inputs, 3, is_training, out_filters, out_filters,
                            start_idx=0, separable=True)
      branches[tf.equal(count, 2)] = lambda: y
    with tf.variable_scope("branch_3"):
      y = self._conv_branch(inputs, 5, is_training, out_filters, out_filters,
                            start_idx=0)
      branches[tf.equal(count, 3)] = lambda: y
    with tf.variable_scope("branch_4"):
      y = self._conv_branch(inputs, 5, is_training, out_filters, out_filters,
                            start_idx=0, separable=True)
      branches[tf.equal(count, 4)] = lambda: y
    with tf.variable_scope("branch_5"):
      y = self._pool_branch(inputs, is_training, out_filters, "avg",
                            start_idx=0)
    branches[tf.equal(count, 5)] = lambda: y
    with tf.variable_scope("branch_6"):
      y = self._pool_branch(inputs, is_training, out_filters, "max",
                            start_idx=0)
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
      with tf.variable_scope("skip"):
        res_layers = []
        for i in range(layer_id):
          res_layers.append(tf.cond(tf.equal(skip[i], 1),
                                    lambda: prev_layers[i],
                                    lambda: tf.zeros_like(prev_layers[i])))
        res_layers.append(out)
        out = tf.add_n(res_layers)
        out = batch_norm(out, is_training, data_format=self.data_format)

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
    if self.data_format == "NHWC":
      inp_c = inputs.get_shape()[3].value
      actual_data_format = "channels_last"
    elif self.data_format == "NCHW":
      inp_c = inputs.get_shape()[1].value
      actual_data_format = "channels_first"

      count = self.sample_arc[start_idx]
      print("count hereeee", count)
      if count == 0:#Empty Layer
        out = tf.identity(inputs)
      #out = tf.layers.max_pooling2d(inputs, [1, 1], [1, 1], "SAME", data_format=actual_data_format)
      elif count in [1, 3]: # convlutional layers
        filter_sizes = {1: 3, 3: 5}
        filter_size = filter_sizes.get(count)
        print("filter_size", filter_size)
        with tf.variable_scope("conv_1x1"): 
          w = create_weight("w", [1, 1, inp_c, out_filters])
          out = relu6(inputs)
          out = tf.nn.conv2d(out, w, [1, 1, 1, 1], "SAME",
                              data_format=self.data_format)
          out = batch_norm(out, is_training, data_format=self.data_format)
        with tf.variable_scope("conv_{0}x{0}".format(filter_size)):
          w = create_weight("w", [filter_size, filter_size, out_filters, out_filters])
          out = relu6(out)
          out = tf.nn.conv2d(out, w, [1, 1, 1, 1], "SAME",data_format=self.data_format)
          out = batch_norm(out, is_training, data_format=self.data_format)
      elif count in [2,4]: # depthwise separable layers
        filter_sizes = {2: 3, 4: 5}
        filter_size = filter_sizes.get(count)
        with tf.variable_scope("conv_1x1"): 
          w = create_weight("w", [1, 1, inp_c, out_filters])
          out = relu6(inputs)
          out = tf.nn.conv2d(out, w, [1, 1, 1, 1], "SAME",
                              data_format=self.data_format)
        with tf.variable_scope("conv_{0}x{0}".format(filter_size)):
          ch_mul=1
          w_depth = create_weight("w_depth", [filter_size, filter_size, out_filters, ch_mul])
          w_point = create_weight("w_point", [1, 1, out_filters * ch_mul, count])
          x = tf.nn.separable_conv2d(out, w_depth, w_point, strides=[1, 1, 1, 1],
                                      padding="SAME", data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
      elif count == 5:        
        with tf.variable_scope("pool"):
          out = tf.layers.average_pooling2d(inputs, [3, 3], [1, 1], "SAME", data_format=actual_data_format)
      elif count == 6:
        with tf.variable_scope("pool"):
          out = tf.layers.max_pooling2d(inputs, [3, 3], [1, 1], "SAME", data_format=actual_data_format)

      else:
        raise ValueError("Unknown operation number '{0}'".format(count))

    if layer_id > 0:
      # if self.whole_channels:
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
      with tf.variable_scope("skip"):
        w = create_weight(
          "w", [1, 1, total_skip_channels * out_filters, out_filters])
        out = tf.nn.relu(out)
        out = tf.nn.conv2d(
          out, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        out = batch_norm(out, is_training, data_format=self.data_format)

    return out

  def _conv_branch(self, inputs, filter_size, is_training, count, out_filters,
                   ch_mul=1, start_idx=None, separable=False):
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

    with tf.variable_scope("inp_conv_1"):
      w = create_weight("w", [1, 1, inp_c, out_filters])
      x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
      x = batch_norm(x, is_training, data_format=self.data_format)
      x = tf.nn.relu(x)

    with tf.variable_scope("out_conv_{}".format(filter_size)):
      if start_idx is None:
        if separable:
          w_depth = create_weight(
            "w_depth", [self.filter_size, self.filter_size, out_filters, ch_mul])
          w_point = create_weight("w_point", [1, 1, out_filters * ch_mul, count])
          x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                     padding="SAME", data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
        else:
          w = create_weight("w", [filter_size, filter_size, inp_c, count])
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
      else:
        if separable:
          w_depth = create_weight("w_depth", [filter_size, filter_size, out_filters, ch_mul])
          w_point = create_weight("w_point", [out_filters, out_filters * ch_mul])
          w_point = w_point[start_idx:start_idx+count, :]
          w_point = tf.transpose(w_point, [1, 0])
          w_point = tf.reshape(w_point, [1, 1, out_filters * ch_mul, count])

          x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                     padding="SAME", data_format=self.data_format)
          mask = tf.range(0, out_filters, dtype=tf.int32)
          mask = tf.logical_and(start_idx <= mask, mask < start_idx + count)
          x = batch_norm_with_mask(
            x, is_training, mask, out_filters, data_format=self.data_format)
        else:
          w = create_weight("w", [filter_size, filter_size, out_filters, out_filters])
          w = tf.transpose(w, [3, 0, 1, 2])
          w = w[start_idx:start_idx+count, :, :, :]
          w = tf.transpose(w, [1, 2, 3, 0])
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
          mask = tf.range(0, out_filters, dtype=tf.int32)
          mask = tf.logical_and(start_idx <= mask, mask < start_idx + count)
          x = batch_norm_with_mask(
            x, is_training, mask, out_filters, data_format=self.data_format)
      x = tf.nn.relu(x)
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

    with tf.variable_scope("conv_1"):
      w = create_weight("w", [1, 1, inp_c, self.out_filters])
      x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
      x = batch_norm(x, is_training, data_format=self.data_format)
      x = tf.nn.relu(x)

    with tf.variable_scope("pool"):
      if self.data_format == "NHWC":
        actual_data_format = "channels_last"
      elif self.data_format == "NCHW":
        actual_data_format = "channels_first"

      if avg_or_max == "avg":
        x = tf.layers.average_pooling2d(
          x, [3, 3], [1, 1], "SAME", data_format=actual_data_format)
      elif avg_or_max == "max":
        x = tf.layers.max_pooling2d(
          x, [3, 3], [1, 1], "SAME", data_format=actual_data_format)
      else:
        raise ValueError("Unknown pool {}".format(avg_or_max))

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
    with tf.variable_scope("empty"):
      #x = tf.identity(inputs)    
      x = tf.layers.max_pooling2d(inputs, [1, 1], [1, 1], "SAME")

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
    logits = self._model(self.x_train, is_training=True)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_train)
    self.loss = tf.reduce_mean(log_probs)

    self.train_preds = tf.argmax(logits, axis=1)
    self.train_preds = tf.to_int32(self.train_preds)
    self.train_acc = tf.equal(self.train_preds, self.y_train)
    self.train_acc = tf.to_int32(self.train_acc)
    self.train_acc = tf.reduce_sum(self.train_acc)

    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
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

