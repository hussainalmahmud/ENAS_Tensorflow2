import tensorflow as tf
from general_controller import GeneralController
import pytest
from src.general_controller import GeneralController



def test_build_trainer():
    # Test case 1: CIFAR-10 dataset, pe_size = 16
    gc = GeneralController('cifar10', 16)
    child_model = tf.keras.models.Sequential()
    gc.build_trainer(child_model)
    assert gc.max_cycle == 135395000.0
    assert gc.min_cycle == 0.0
    assert gc.cycle_norm == 1.0

    # Test case 2: CIFAR-10 dataset, pe_size = 9
    gc = GeneralController('cifar10', 9)
    child_model = tf.keras.models.Sequential()
    gc.build_trainer(child_model)
    assert gc.max_cycle == 182414200.0
    assert gc.min_cycle == 0.0
    assert gc.cycle_norm == 1.0

    # Test case 3: CIFAR-10 dataset, pe_size = 8
    gc = GeneralController('cifar10', 8)
    child_model = tf.keras.models.Sequential()
    gc.build_trainer(child_model)
    assert gc.max_cycle == 182414200.0
    assert gc.min_cycle == 0.0
    assert gc.cycle_norm == 1.0

    # Test case 4: CIFAR-10 dataset, pe_size = 4
    gc = GeneralController('cifar10', 4)
    child_model = tf.keras.models.Sequential()
    gc.build_trainer(child_model)
    assert gc.max_cycle == 358232060.0
    assert gc.min_cycle == 0.0
    assert gc.cycle_norm == 1.0

    # Test case 5: non-CIFAR-10 dataset
    gc = GeneralController('other', 16)
    child_model = tf.keras.models.Sequential()
    gc.build_trainer(child_model)
    assert gc.max_cycle == 324971026.0
    assert gc.min_cycle == 303490.0
    assert gc.cycle_norm == 0.0

test_build_trainer()