from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import time

import numpy as np
import tensorflow as tf
# tf.config.run_functions_eagerly(True)
tf.compat.v1.disable_eager_execution()


from src import utils
from src.utils import Logger
from src.cifar10_1.data_utils import read_data
from src.cifar10_1.general_controller import GeneralController
from src.cifar10_1.general_child import GeneralChild
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)	

import argparse

parser = argparse.ArgumentParser(description="ENAS-CGRA-TF2")

# General
parser.add_argument("--reset_output_dir", action="store_true", help="Delete output_dir if exists.")
# parser.add_argument("--data_path", type=str, default="", help="Path to data directory.")
parser.add_argument("--output_dir", type=str, default="output", help="Path to output directory.")
parser.add_argument("--data_format", type=str, default="NHWC", choices=["NHWC", "NCWH"], help="'NHWC' or 'NCWH'")
parser.add_argument("--search_for", type=str, default=None, choices=["macro", "micro"], help="Must be [macro|micro]")

# Training
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
parser.add_argument("--num_epochs", type=int, default=300, help="Number of training epochs.")
parser.add_argument("--child_lr_dec_every", type=int, default=100, help="Learning rate decay frequency.")
parser.add_argument("--child_num_layers", type=int, default=5, help="Number of layers in child model.")
parser.add_argument("--child_num_cell_layers", type=int, default=5, help="Number of cells in child model.")
parser.add_argument("--child_filter_size", type=int, default=5, help="Filter size in child model.")
parser.add_argument("--child_out_filters", type=int, default=48, help="Number of output filters in child model.")
parser.add_argument("--child_out_filters_scale", type=int, default=1, help="Output filter scale in child model.")
parser.add_argument("--child_num_branches", type=int, default=4, help="Number of branches in child model.")
parser.add_argument("--child_num_aggregate", type=int, default=None, help="Number of aggregates in child model.")
parser.add_argument("--child_num_replicas", type=int, default=1, help="Number of replicas in child model.")
parser.add_argument("--child_block_size", type=int, default=3, help="Block size in child model.")
parser.add_argument("--child_lr_T_0", type=int, default=None, help="T_0 for lr schedule in child model.")
parser.add_argument("--child_lr_T_mul", type=int, default=None, help="T_mul for lr schedule in child model.")
parser.add_argument("--child_cutout_size", type=int, default=None, help="CutOut size in child model.")
parser.add_argument("--child_grad_bound", type=float, default=5.0, help="Gradient clipping in child model.")
parser.add_argument("--child_lr", type=float, default=0.1, help="Learning rate in child model.")
parser.add_argument("--child_lr_dec_rate", type=float, default=0.1, help="Learning rate decay rate in child model.")
parser.add_argument("--child_keep_prob", type=float, default=0.5, help="Keep probability in child model.")
parser.add_argument("--child_drop_path_keep_prob", type=float, default=1.0, help="Minimum drop_path_keep_prob in child model.")
parser.add_argument("--child_l2_reg", type=float, default=1e-4, help="L2 regularization in child model.")
parser.add_argument("--child_lr_max", type=float, default=None, help="Maximum learning rate in child model.")
parser.add_argument("--child_lr_min", type=float, default=None, help="Minimum learning rate in child model.")
parser.add_argument("--child_skip_pattern", type=str, default=None, choices=["dense", None], help="Must be ['dense', None].")
parser.add_argument("--child_fixed_arc", type=str, default=None, help="Fixed architecture in child model.")
parser.add_argument("--child_use_aux_heads", action="store_true", help="Use auxiliary heads in child model.")
parser.add_argument("--child_sync_replicas", action="store_true", help="To sync or not to sync.")
parser.add_argument("--child_lr_cosine", action="store_true", help="Use cosine lr schedule in child model.")
parser.add_argument('--controller_use_critic', type=bool, default=False, help='Whether to use critic in the controller.')

# Controller
parser.add_argument("--controller_lr", type=float, default=1e-3, help="Learning rate in controller model.")
parser.add_argument("--controller_lr_dec_rate", type=float, default=1.0, help="Learning rate decay rate in controller model.")
parser.add_argument("--controller_keep_prob", type=float, default=0.5, help="Keep probability in controller model.")
parser.add_argument("--controller_l2_reg", type=float, default=0.0, help="L2 regularization in controller model.")
parser.add_argument("--controller_bl_dec", type=float, default=0.99, help="Bl dec in controller model.")
parser.add_argument("--controller_tanh_constant", type=float, default=None, help="Tanh constant in controller model.")
parser.add_argument("--controller_op_tanh_reduce", type=float, default=1.0, help="Op tanh reduce in controller model.")
parser.add_argument("--controller_temperature", type=float, default=None, help="Temperature in controller model.")
parser.add_argument("--controller_entropy_weight", type=float, default=None, help="Entropy weight in controller model.")
parser.add_argument("--controller_skip_target", type=float, default=0.8, help="Skip target in controller model.")
parser.add_argument("--controller_skip_weight", type=float, default=0.0, help="Skip weight in controller model.")
parser.add_argument("--controller_num_aggregate", type=int, default=1, help="Number of aggregates in controller model.")
parser.add_argument("--controller_num_replicas", type=int, default=1, help="Number of replicas in controller model.")
parser.add_argument("--controller_train_steps", type=int, default=50, help="Training steps in controller model.")
parser.add_argument("--controller_forwards_limit", type=int, default=2, help="Forwards limit in controller model.")
parser.add_argument("--controller_train_every", type=int, default=2, help="Train the controller after this number of epochs.")
parser.add_argument("--controller_search_whole_channels", action="store_true", help="Search whole channels in controller model.")
parser.add_argument("--controller_sync_replicas", action="store_true", help="To sync or not to sync.")
parser.add_argument("--controller_training", action="store_true", help="Controller training in controller model.")
parser.add_argument("--controller_PE_Size", type=int, default=16, help="PE Size in controller model.")
parser.add_argument("--controller_Alpha_value", type=float, default=0, help="Alpha value in controller model.")
parser.add_argument("--controller_dataset", type=str, default=None, help="[Cifar10 or Cifar10]")

# Logging and Evaluation
parser.add_argument("--log_every", type=int, default=50, help="How many steps to log.")
parser.add_argument("--eval_every_epochs", type=int, default=1, help="How many epochs to eval.")

args = parser.parse_args()

# Now you can access the arguments using args.argument_name



def get_ops(images, labels):
    """
    Args:
        images: dict with keys {"train", "valid", "test"}.
        labels: dict with keys {"train", "valid", "test"}.
        args: Parsed command-line arguments.
    """

    assert args.search_for is not None, "Please specify --search_for"

    print("images: ",images.keys())
    print("labels: ",labels.keys())

    input_images = tf.keras.Input(shape=(32, 32, 3), name="input_images")
    input_labels = tf.keras.Input(shape=(1,), name="input_labels")

    ControllerClass = GeneralController
    ChildClass = GeneralChild

    child_model = ChildClass(
        images,
        labels,
        input_images,
        input_labels,
        use_aux_heads=args.child_use_aux_heads,
        cutout_size=args.child_cutout_size,
        whole_channels=args.controller_search_whole_channels,
        num_layers=args.child_num_layers,
        num_cells=args.child_num_cell_layers,
        num_branches=args.child_num_branches,
        fixed_arc=args.child_fixed_arc,
        out_filters_scale=args.child_out_filters_scale,
        out_filters=args.child_out_filters,
        keep_prob=args.child_keep_prob,
        drop_path_keep_prob=args.child_drop_path_keep_prob,
        num_epochs=args.num_epochs,
        l2_reg=args.child_l2_reg,
        # data_format=args.data_format,
        # batch_size=args.batch_size,
        clip_mode="norm",
        grad_bound=args.child_grad_bound,
        lr_init=args.child_lr,
        lr_dec_every=args.child_lr_dec_every,
        lr_dec_rate=args.child_lr_dec_rate,
        lr_cosine=args.child_lr_cosine,
        lr_max=args.child_lr_max,
        lr_min=args.child_lr_min,
        lr_T_0=args.child_lr_T_0,
        lr_T_mul=args.child_lr_T_mul,
        optim_algo="momentum",
        sync_replicas=args.child_sync_replicas,
        num_aggregate=args.child_num_aggregate,
        num_replicas=args.child_num_replicas,
        pe_size=args.controller_PE_Size,
        alpha_value=args.controller_Alpha_value,
        dataset=args.controller_dataset,
    )

    if args.child_fixed_arc is None:
        controller_model = ControllerClass(
            search_for=args.search_for,
            search_whole_channels=args.controller_search_whole_channels,
            skip_target=args.controller_skip_target,
            skip_weight=args.controller_skip_weight,
            num_cells=args.child_num_cell_layers,
            num_layers=args.child_num_layers,
            num_branches=args.child_num_branches,
            out_filters=args.child_out_filters,
            lstm_size=64,
            lstm_num_layers=1,
            lstm_keep_prob=1.0,
            tanh_constant=args.controller_tanh_constant,
            op_tanh_reduce=args.controller_op_tanh_reduce,
            temperature=args.controller_temperature,
            lr_init=args.controller_lr,
            lr_dec_start=0,
            lr_dec_every=1000000,  # never decrease learning rate
            l2_reg=args.controller_l2_reg,
            entropy_weight=args.controller_entropy_weight,
            bl_dec=args.controller_bl_dec,
            use_critic=args.controller_use_critic,
            optim_algo="adam",
            sync_replicas=args.controller_sync_replicas,
            num_aggregate=args.controller_num_aggregate,
            num_replicas=args.controller_num_replicas,
            pe_size=args.controller_PE_Size,
            alpha_value=args.controller_Alpha_value,
            dataset=args.controller_dataset,
        )

        child_model.connect_controller(controller_model)
        controller_model.build_trainer(child_model)

        controller_ops = {
            "train_step": controller_model.train_step,
            "loss": controller_model.loss,
            "train_op": controller_model.train_op,
            "lr": controller_model.lr,
            "grad_norm": controller_model.grad_norm,
            "valid_acc": controller_model.valid_acc,
            "optimizer": controller_model.optimizer,
            "baseline": controller_model.baseline,
            "entropy": controller_model.sample_entropy,
            "sample_arc": controller_model.sample_arc,
            "skip_rate": controller_model.skip_rate,
            # Added cycles, reward, new_reward, and cyc_norm
            "cycles": controller_model.cycles,
            "reward": controller_model.reward,
            "new_reward": controller_model.new_reward,
            "cycle_norm": controller_model.cycle_norm,
            "skip_num": controller_model.skip_num,
        }
    else:
        assert not args.controller_training, (
            "--child_fixed_arc is given, cannot train controller")
        child_model.connect_controller(None)
        controller_ops = None

    child_ops = {
        "global_step": child_model.global_step,
        "loss": child_model.loss,
        "train_op": child_model.train_op,
        "lr": child_model.lr,
        "grad_norm": child_model.grad_norm,
        "train_acc": child_model.train_acc,
        "optimizer": child_model.optimizer,
        "num_train_batches": child_model.num_train_batches,

    }

    ops = {
        "child": child_ops,
        "controller": controller_ops,
        "eval_every": child_model.num_train_batches * args.eval_every_epochs,
        "eval_func": child_model.eval_once,
        "num_train_batches": child_model.num_train_batches,
    }

    return ops

def train():
    # Check if a fixed architecture for the child model is specified.
    if args.child_fixed_arc is None:
        images, labels = read_data(args.controller_dataset)
    else:
        images, labels = read_data(args.controller_dataset, num_valids=5000)
    input_images = tf.keras.Input(shape=(32, 32, 3), name="input_images")
    input_labels = tf.keras.Input(shape=(1,), name="input_labels")
    # Create a new graph, and make it the default.
    g = tf.Graph()
    # Set the new graph as the default graph.
    with g.as_default():
        # Retrieve operations for both the child model and the controller.
        ops = get_ops(images, labels)
        child_ops = ops["child"]
        controller_ops = ops["controller"]

        # # Set up a mechanism to save the model periodically.
        # checkpoint = tf.train.Checkpoint(model=saver)
        # manager = tf.train.CheckpointManager(
        #     checkpoint, args.output_dir, max_to_keep=2)

        print("-" * 80)
        print("Starting training")

        start_time = time.time()
        while True:
            # Operations to run in this training step for the child model.
            run_ops = [
                child_ops["loss"],
                child_ops["lr"],
                child_ops["grad_norm"],
                child_ops["train_acc"],
                child_ops["train_op"],
            ]
            # Execute the specified operations and Fetch the current training step.
            loss, lr, gn, tr_acc, _ = child_ops["train_step"](*run_ops,
                                                              input_images,  # Pass images directly
                                                             input_labels ) # Pass labels directly
            global_step = child_ops["global_step"].numpy()

            # Adjust step count if using synchronized replicas.
            if args.child_sync_replicas:
                actual_step = global_step * args.num_aggregate
            else:
                actual_step = global_step
            # Calculate the current epoch.
            epoch = actual_step // ops["num_train_batches"]
            curr_time = time.time()
            # Log details every specified number of steps.
            if global_step % args.log_every == 0:
                log_string = ""
                log_string += "epoch={:<6d}".format(epoch)
                log_string += "ch_step={:<6d}".format(global_step)
                log_string += " loss={:<8.6f}".format(loss)
                log_string += " lr={:<8.4f}".format(lr)
                log_string += " |g|={:<8.4f}".format(gn)
                log_string += " tr_acc={:<3d}/{:>3d}".format(
                    tr_acc, args.batch_size)
                log_string += " mins={:<10.2f}".format(
                    float(curr_time - start_time) / 60)
                print(log_string)

            # Periodically evaluate the model.
            if actual_step % ops["eval_every"] == 0:
                # Evaluate the child model on validation and test datasets.
                if (args.controller_training and
                        epoch % args.controller_train_every == 0):
                    print("Epoch {}: Training controller".format(epoch))
                    for ct_step in range(args.controller_train_steps *
                                         args.controller_num_aggregate):
                        run_ops = [
                            controller_ops["loss"],
                            controller_ops["reward"],
                            controller_ops["new_reward"],
                            controller_ops["cycle_norm"],
                            controller_ops["entropy"],
                            controller_ops["lr"],
                            controller_ops["grad_norm"],
                            controller_ops["valid_acc"],
                            controller_ops["baseline"],
                            controller_ops["skip_num"],
                            controller_ops["train_op"],
                        ]
                        loss, reward, new_reward, cycle_norm, entropy, lr, gn, val_acc, bl, skip_num, _ = controller_ops[
                            "train_step"](*run_ops)
                        controller_step = controller_ops["train_step"].numpy()

                        if ct_step % args.log_every == 0:
                            curr_time = time.time()
                            log_string = ""
                            log_string += "ctrl_step={:<6d}".format(controller_step)
                            log_string += " loss={:<7.3f}".format(loss)
                            log_string += " acc={:<6.4f}".format(val_acc)
                            log_string += " reward={:<7.3f}".format(reward)
                            log_string += " n_reward={:<7.3f}".format(new_reward)
                            log_string += " c_norm={:<7.3f}".format(cycle_norm)
                            log_string += " ent={:<5.2f}".format(entropy)
                            log_string += " lr={:<6.4f}".format(lr)
                            log_string += " |g|={:<8.4f}".format(gn)
                            log_string += " bl={:<5.2f}".format(bl)
                            log_string += " mins={:<.2f}".format(
                                float(curr_time - start_time) / 60)
                            print(log_string)

                    print("Here are 10 architectures")
                    for _ in range(10):
                        arc, acc, cycles, new_reward, skip = controller_ops["sample_arc"]()
                        if args.search_for == "micro":
                            normal_arc, reduce_arc = arc
                            print(np.reshape(normal_arc, [-1]))
                            print(np.reshape(reduce_arc, [-1]))
                        else:
                            start = 0
                            for layer_id in range(args.child_num_layers):
                                if args.controller_search_whole_channels:
                                    end = start + 1 + layer_id
                                else:
                                    end = start + 2 * args.child_num_branches + layer_id
                                print(np.reshape(arc[start: end], [-1]))
                                start = end
                        print("-" * 80)
                        print("arc")
                        print(arc)
                        print("cycle")
                        print(cycles)
                        print("-" * 80)
                        print("Sum cycle= {}".format(np.sum(cycles)))
                        print("val_acc= {:<6.4f}".format(acc))
                        print("Reward= {:<6.4f}".format(new_reward))
                        print("-" * 80)
                print("Epoch {}: Eval".format(epoch))
                if args.child_fixed_arc is None:
                    ops["eval_func"](controller_ops["eval_func"], "valid")
                ops["eval_func"](controller_ops["eval_func"], "test")

            if epoch >= args.num_epochs:
                break


if __name__ == "__main__":
    print("-" * 80)
    if not os.path.isdir(args.output_dir):
        print("Path {} does not exist. Creating.".format(args.output_dir))
        os.makedirs(args.output_dir)
    elif args.reset_output_dir:
        print("Path {} exists. Remove and remake.".format(args.output_dir))
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    print("-" * 80)
    log_file = os.path.join(args.output_dir, "stdout")
    print("Logging to {}".format(log_file))
    sys.stdout = Logger(log_file)

    utils.print_user_flags()
    train()  # Pass the args to the train function

