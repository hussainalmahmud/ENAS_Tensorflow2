# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from collections import OrderedDict


def accuracy_metrics(y_true, logits):
    return {'enas_acc': accuracy(y_true, logits)}

def accuracy(y_true, logits):
    # y_true: shape=(batch_size) or (batch_size,1), type=integer
    # logits: shape=(batch_size, num_of_classes), type=float
    # returns float
    batch_size = y_true.shape[0]
    y_true = tf.squeeze(y_true)
    y_pred = tf.math.argmax(logits, axis=1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    equal = tf.cast(y_pred == y_true, tf.int32)
    return tf.math.reduce_sum(equal).numpy() / batch_size

## more utils for trainer

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class AverageMeterGroup:
    """
    Average meter group for multiple average meters.
    """

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data):
        """
        Update the meter group with a dict of metrics.
        Non-exist average meters will be automatically created.
        """
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        """
        Return a summary string of group data.
        """
        return "  ".join(v.summary() for v in self.meters.values())


class AverageMeter:
    """
    Computes and stores the average and current value.

    Parameters
    ----------
    name : str
        Name to display.
    fmt : str
        Format string to print the values.
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        Reset the meter.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update with value and weight.

        Parameters
        ----------
        val : float or int
            The new value to be accounted in.
        n : int
            The weight of the new value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)
    

# class fill_zero_grads:
#     def __init__(self, grads, weights):
#         self.grads = grads
#         self.weights = weights

#     def __iter__(self):
#         for grad, weight in zip(self.grads, self.weights):
#             if grad is None:
#                 yield tf.zeros_like(weight)
#             else:
#                 yield grad

def fill_zero_grads(grads, weights):
    """
    Fill zero gradients for weights that are not trainable.
    """
    filled_grads = []
    for grad, weight in zip(grads, weights):
        if grad is None:
            filled_grads.append(tf.zeros_like(weight))
        else:
            filled_grads.append(grad)
    return filled_grads