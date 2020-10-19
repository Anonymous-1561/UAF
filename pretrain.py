import argparse
import logging
import os
import time

import tensorflow as tf

from common import environment_preset, setup_folder, get_gpu_config

from data.loader_pretrain import DataLoaderPretrain

from models.nextitnet_plain import NextItNet

from utils import logger
from utils.metrics import SRSMetric
from utils.tools import dict_to_str


def update_model_args(item_size):
    configs_cpy = configs.copy()
    configs_cpy.update(
        {
            "using_negative_sampling": False,
            # "using_negative_sampling": True,
            # "negative_sampling_ratio": 0.2,
            "item_size": item_size,
            "dilations": configs_cpy["block_shape"] * configs_cpy["n_blocks"],
        }
    )
    return configs_cpy


def train():
    batch_size = configs["batch_size"]
    log_meter = configs["log_every"]
    total_steps = int(train_set.shape[0] / batch_size)

    for batch_step in range(total_steps):
        f, t = batch_step * batch_size, (batch_step + 1) * batch_size
        batch = train_set[f:t, :]
        _, loss = sess.run(
            [optimizer, model.loss_train],
            feed_dict={model.input_train: batch},
        )
        if (batch_step + 1) % log_meter == 0:
            logging.info("\t<{:5d}/{:5d}> Loss: {:.4f}".format(batch_step + 1, total_steps, loss))


def evaluate():
    batch_size = configs["batch_size"]
    total_steps = int(test_set.shape[0] / batch_size)

    meter = SRSMetric(k_list=[5, 20])
    meter.setup_and_clean()

    for batch_step in range(total_steps):
        f, t = batch_step * batch_size, (batch_step + 1) * batch_size
        test_batch = test_set[f:t, :]

        pred_probs = sess.run(model.probs_test, feed_dict={model.input_test: test_batch})

        meter.submit(pred_probs, test_batch[:, -1:])

    meter.calc()
    meter.output_to_logger()

    return meter.mrr[5]


def start():
    logging.info("Item Size: {}".format(configs["item_size"]))
    logging.info("Training Set Size: {}".format(len(train_set)))
    logging.info("Test Set Size: {}".format(len(test_set)))

    if not configs["resume"]:
        init = tf.global_variables_initializer()
        sess.run(init)
        start_at = 0
    else:
        resume_op = tf.train.Saver()
        logging.info("Loading from checkpoint `{}`...".format(configs["resume_path"]))
        resume_op.restore(sess, configs["resume_path"])
        start_at = configs["resume_at"]
        logging.info(">>>>> Resume from checkpoint, start at epoch {}".format(start_at))

    saver = tf.train.Saver(max_to_keep=1)

    model_save_path = configs["store_path"]
    total_iter = configs["iter"]

    best_mrr_at5 = 0.0

    for idx in range(start_at, total_iter):
        logging.info("-" * 30)
        logging.info("Iter: {} / {}".format(idx + 1, total_iter))
        tic = time.time()
        # --------------------------------------------
        train()
        mrr_at5 = evaluate()
        if mrr_at5 > best_mrr_at5:
            logging.info(">>>>> Saving... better MRR@5: {:.4f} <<<<< ".format(mrr_at5))
            save_path = os.path.join(model_save_path, "model.tfkpt")
            saver.save(sess, save_path)
            best_mrr_at5 = mrr_at5
        # --------------------------------------------
        toc = time.time()
        logging.info("Iter: {} / {} finish. Time: {:.2f} min".format(idx + 1, total_iter, (toc - tic) / 60))


def parse_arg(args_inline=None, **patch):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str)
    parser.add_argument("--resume_at", type=int, default=0)

    parser.add_argument("--n_blocks", required=True, type=int)
    parser.add_argument("--block_shape", type=str, default="1,4")
    parser.add_argument("--channel", type=int, default=256)
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--min_freq", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--iter", type=int, default=20)

    parser.add_argument("--store_root", type=str, default="store", help="/path/to/store")
    parser.add_argument("--console_output", action="store_true", help="Print logger info to console")
    parser.add_argument("--data_folder", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    parser.add_argument("--split_ratio", type=float, default=0.2, help="0.2 means 80% training and 20% testing")
    parser.add_argument("--log_every", type=int, default=500, help="Print log info every (x) iterations")
    parser.add_argument("--occupy", type=float, help="Occupy {x}-ratio of GPU memory at beginning")

    if args_inline is not None:
        data = vars(parser.parse_args(args_inline))
    else:
        data = vars(parser.parse_args())

    blocks = data["block_shape"]
    data["block_shape"] = [int(t) for t in blocks.split(",")]

    data.update(patch)
    return data


if __name__ == "__main__":
    # DEBUG setup
    pack = {
        # "console_output": True,
        # "log_every": 20,
    }
    configs = parse_arg(**pack)

    # 0. setup folder and logger
    path = setup_folder(configs)
    configs.update({"store_path": path})
    logger.setup_logger(configs)

    # 1. setup env, seed and gpu
    environment_preset(configs)
    gpu_config = get_gpu_config(configs["occupy"])
    sess = tf.Session(config=gpu_config)

    # 2. prepare dataset
    data_loader = DataLoaderPretrain(configs, configs["min_freq"])
    train_set, test_set = data_loader.split()
    data_loader.save_dict()
    logging.info("item dict saved to itemdict.pkl")

    # 3. update model configs
    configs = update_model_args(data_loader.item_nums)
    logging.info(dict_to_str(configs, "Configurations"))

    # 4. create model
    model = NextItNet(configs)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(configs["lr"]).minimize(model.loss_train)

    # 5. launch the rocket
    start()

    sess.close()
