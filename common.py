import tensorflow as tf
import os
import random
import logging
import numpy as np

from time import strftime, localtime


def setup_seed(SEED):
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)


def environment_preset(configs):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(configs["gpu"])

    seed = configs["seed"]
    logging.info("Using seed: {}".format(seed))
    setup_seed(seed)


def setup_folder(args):
    root = args["store_root"]
    if not os.path.isdir(root):
        print "Root folder of store is created: {}".format(root)
        os.mkdir(root)

    folder_name = args["name"] + strftime("-%m.%d-%H.%M.%S", localtime())
    full_path = os.path.join(root, folder_name)
    if os.path.isdir(full_path):
        raise ValueError("Folder with name `{}` already exists.".format(full_path))
    os.mkdir(full_path)
    return full_path


def get_proto_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def get_proto_config_with_occupy(ratio):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = ratio
    return config


def get_block_by_id(block_id):
    prefix = "residual_{}".format(block_id)
    return get_parameters_with_prefix(prefix)


def get_parameters_with_prefix(prefix):
    variables = tf.contrib.framework.get_variables_to_restore()
    return [v for v in variables if v.name.startswith(prefix)]


def get_gpu_config(ratio):
    if ratio is None:
        gpu_config = get_proto_config()
        logging.info("Auto-growth GPU memory.")
    else:
        gpu_config = get_proto_config_with_occupy(ratio)
        logging.info("{:.1f}% GPU memory occupied.".format(ratio * 100))

    return gpu_config


def summary_block(usage, block_size, title):
    logging.info("<Usage>::Block Usage of [{}]".format(title))
    block_usage = np.array(usage, dtype=np.float)

    ratio = np.sum(block_usage, axis=0) / len(block_usage)
    logging.info("\tUsage: " + " ".join(["{:.2f}".format(x) for x in ratio]))

    row_sum = np.sum(block_usage, axis=1)
    min_usage = np.min(row_sum)
    max_usage = np.max(row_sum)
    per_sample_usage = row_sum.squeeze()
    mean_usage = np.mean(per_sample_usage)
    std_usage = np.std(per_sample_usage)

    logging.info(
        "\t{} / [{} -> {}], Mean: {:.3f}, Std: {:.3f}".format(block_size, min_usage, max_usage, mean_usage, std_usage)
    )

    unique_policies = np.unique(block_usage, axis=0)
    lp = len(unique_policies)
    la = 2 ** block_size
    logging.info("\tUnique Policies: {}, Upper Bound: {}, Ratio: {:.3f}%".format(lp, la, 100.0 * lp / la))
