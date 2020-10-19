import argparse
import logging
import os
import time

import tensorflow as tf
import numpy as np

from data.loader_fintune import DataLoaderFinetunePlain as DataLoader
from common import environment_preset, setup_folder, get_gpu_config, summary_block

from models.policy_net import PolicyNet, PolicyNetRandom, PolicyNetGru
from models.nextitnet_policy import NextItNetPolicy
from models.target_embedding import TargetEmbeddingBPR

from utils import logger
from utils.metrics import SRSMetric
from utils.tools import dict_to_str, random_neg, random_negs


def update_model_args(dataloader):
    configs_cpy = configs.copy()
    configs_cpy.update(
        {
            "using_negative_sampling": False,
            # "using_negative_sampling": True,
            # "negative_sampling_ratio": 0.2,
            "item_size": dataloader.source_nums,
            "target_item_size": dataloader.target_nums,
            "dilations": configs_cpy["block_shape"] * configs_cpy["n_blocks"],
        }
    )
    return configs_cpy


def train():
    batch_size = configs["batch_size"]
    log_meter = configs["log_every"]
    total_steps = int(train_set.shape[0] / batch_size)

    train_usage_sample = []

    first, last = data_loader.first_target, data_loader.last_target
    for batch_step in range(total_steps):
        f, t = batch_step * batch_size, (batch_step + 1) * batch_size
        item_batch = train_set[f:t, :]  # [B, L+1]

        context_batch = item_batch[:, :-1]  # [B, L]
        pos_target = item_batch[:, -1:]  # [B, 1]
        neg_target = np.array([[random_neg(first, last, s[0])] for s in pos_target])  # [B, 1]

        _, loss_out, action = sess.run(
            [train_op, target_model.train_loss, policy_model.actions_train],
            feed_dict={
                policy_model.input: context_batch,
                source_model.input_source_train: context_batch,
                target_model.input_train_pos: pos_target,
                target_model.input_train_neg: neg_target,
            },
        )
        train_usage_sample.extend(np.array(action).tolist())
        if (batch_step + 1) % log_meter == 0:
            logging.info("\t<{:5d}/{:5d}> Loss: {:.4f}".format(batch_step + 1, total_steps, loss_out))

    if configs["method"] == "hard":
        summary_block(train_usage_sample, len(configs["dilations"]), "Train")


def evaluate():
    batch_size = configs["batch_size"]
    total_steps = int(test_set.shape[0] / batch_size)

    meter = SRSMetric(k_list=[5, 20])
    meter.setup_and_clean()

    n_neg = configs["n_neg"]

    test_usage_sample = []
    for batch_step in range(total_steps):
        f, t = batch_step * batch_size, (batch_step + 1) * batch_size
        batch = test_set[f:t, :]  # [B, L+1]

        context = batch[:, :-1]
        pos_target = batch[:, -1:]
        neg_target = [random_negs(l=1, r=data_loader.target_nums, size=n_neg, pos=s[0]) for s in pos_target]
        target = np.concatenate([neg_target, pos_target], 1)  # [n_neg*neg+pos]

        test_probs, action = sess.run(
            [target_model.test_probs, policy_model.actions_test],
            feed_dict={
                policy_model.input: context,
                source_model.input_source_test: context,
                target_model.input_test: target,
            },
        )  # [B, #neg+1]
        ground_truth = [[n_neg]] * batch_size
        meter.submit(test_probs, ground_truth)

        test_usage_sample.extend(np.array(action).tolist())

    if configs["method"] == "hard":
        summary_block(test_usage_sample, len(configs["dilations"]), "Test")

    meter.calc()
    meter.output_to_logger()

    return meter.mrr[5]


def start():
    logging.info("Item Size: {}".format(configs["item_size"]))
    logging.info("Training Set Size: {}".format(len(train_set)))
    logging.info("Test Set Size: {}".format(len(test_set)))

    model_save_path = configs["store_path"]
    total_iter = configs["iter"]

    saver = tf.train.Saver(max_to_keep=1)
    best_mrr_at5 = 0.0

    for idx in range(0, total_iter):
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


def get_parameters():
    # policy vars
    # target vars
    # source vars:
    #  - frozen blocks vars
    #  - active blocks vars
    #  - others
    # pretrained vars: frozen blocks vars + others

    _variables = tf.contrib.framework.get_variables_to_restore()

    _policy = [v for v in _variables if v.name.startswith("policy")]
    _target = [v for v in _variables if v.name.startswith("target")]

    _source = set(_variables) - set(_policy) - set(_target)
    _source_frozen = [v for v in _source if not v.name.startswith("finetune") and v.name.startswith("residual")]
    _source_active = [v for v in _source if v.name.startswith("finetune")]
    _source_other = list(set(_source) - set(_source_frozen) - set(_source_active))

    _pretrained = _source_frozen + _source_other

    return _policy, _target, _source_frozen, _source_active, _source_other, _pretrained


def load_parameters():
    pre_model_folder = configs["pre"]
    logging.info("Restoring parameters from `{}`...".format(pre_model_folder))

    policy_v, _, source_frozen_v, _, _, pretrained_v = get_parameters()

    restore_op = tf.train.Saver(var_list=pretrained_v)
    pre_model_path = os.path.join(pre_model_folder, "model.tfkpt")
    restore_op.restore(sess, pre_model_path)

    # 2. copy block parameters to trainable blocks
    graph = tf.get_default_graph()
    op_list = []
    for v in source_frozen_v:
        ft_node = graph.get_tensor_by_name("finetune_" + v.name)
        op_list.append(tf.assign(ft_node, v))
    sess.run(op_list)

    # 3. init target model
    sess.run(tf.variables_initializer([target_model.target_item_embedding]))

    # 4. init policy model
    sess.run(tf.variables_initializer(policy_v))
    if configs["type"] != "random":
        sess.run(tf.assign(policy_model.item_embedding, source_model.item_embedding))

    logging.info("Restore and rebuild finish.")


def parse_arg(args_inline=None, **patch):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--pre", type=str, help="/path/to/model.tfkpt")

    # [conv, gru] * [hard, soft], random
    parser.add_argument("--type", choices=["conv", "gru", "random"], required=True, help="policy type")
    parser.add_argument("--method", choices=["hard", "soft"], help="policy forward type")

    parser.add_argument("--n_blocks", required=True, type=int)
    parser.add_argument("--block_shape", type=str, default="1,4")
    parser.add_argument("--channel", type=int, default=256)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--temp", type=int, default=10)

    parser.add_argument("--n_neg", type=int, default=99, help="sample `n` neg-elements when testing")
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
    pack = {
        # "console_output": True,
        # "log_every": 30,
    }
    configs = parse_arg(**pack)

    # setup folder and logger
    path = setup_folder(configs)
    configs.update({"store_path": path})
    logger.setup_logger(configs)

    # setup env, seed and gpu
    environment_preset(configs)
    gpu_config = get_gpu_config(configs["occupy"])
    sess = tf.Session(config=gpu_config)

    # prepare dataset
    data_loader = DataLoader(configs, threshold=configs["min_freq"])
    data_loader.build()
    train_set, test_set = data_loader.split()

    # update model configs
    configs = update_model_args(data_loader)
    logging.info(dict_to_str(configs, "Configurations"))

    # create models
    with tf.variable_scope("policy"):
        policy_type = configs["type"]
        if policy_type == "conv":
            logging.info("Policy type: << conv >>")
            policy_model = PolicyNet(configs)
        elif policy_type == "gru":
            logging.info("Policy type: << gru >>")
            policy_model = PolicyNetGru(configs)
        elif policy_type == "random":
            logging.info("Policy type: << random >>")
            policy_model = PolicyNetRandom(configs)
        else:
            raise ValueError("Unexpected policy type: `{}`".format(policy_type))

    with tf.variable_scope(tf.get_variable_scope()):
        source_model = NextItNetPolicy(configs)
        source_model.build_train_graph(policy_model.actions_train)
        source_model.build_test_graph(policy_model.actions_test)

    with tf.variable_scope("target"):
        target_model = TargetEmbeddingBPR(configs)
        target_model.build_train_graph(source_model.hidden_train)
        target_model.build_test_graph(source_model.hidden_test)

    if configs["pre"] is not None:
        logging.info("Restoring parameters from pretrain model...")
        load_parameters()
        logging.info("Restore and rebuild finish.")
    else:
        logging.info("Init parameters from scratch.")
        init = tf.global_variables_initializer()
        sess.run(init)

    # create optimizer
    v_policy, v_target, v_source_frozen, v_source_active, v_source_other, v_pretrained = get_parameters()
    active_vars = v_policy + v_target + v_source_active + v_source_other

    optimizer = tf.train.AdamOptimizer(configs["lr"], name="adam")
    train_op = optimizer.minimize(target_model.train_loss, var_list=active_vars)

    sess.run(tf.variables_initializer(optimizer.variables()))

    start()
    sess.close()
