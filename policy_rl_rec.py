import argparse
import logging
import os
import time

import tensorflow as tf
import numpy as np

from common import environment_preset, setup_folder, get_gpu_config, summary_block

from data.loader_fintune import DataLoaderFinetunePlain as DataLoader

from models.policy_net_rl import PolicyNetRL, PolicyNetGruRL
from models.nextitnet_policy import NextItNetPolicy
from models.target_embedding import TargetEmbeddingBPR

from utils import logger
from utils.metrics import SRSMetric, sample_top_k
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


def reward_fn(probs, n_neg, action, gamma, k=5):
    # probs: [B, #item]
    # action: [B, #block]
    top_rank = sample_top_k(probs, k)  # [B, k]
    hr_list = []
    for pred_top_k in top_rank:
        if n_neg in pred_top_k:
            hr_list.append(1)
        else:
            hr_list.append(0)
    hits = np.array(hr_list)

    action_num = np.shape(action)[1]
    reward = np.where(hits > 0, 1 - gamma * np.square(1 - np.sum(action, axis=-1) / action_num), -gamma)
    return reward


def train_rl_on():
    batch_size = configs["batch_size"]
    log_meter = configs["log_every"]
    reward_k = configs["reward_k"]
    n_neg = configs["n_neg"]
    gamma = configs["gamma"]

    action_nums = len(configs["dilations"])
    total_steps = int(train_set.shape[0] / batch_size)
    first, last = data_loader.first_target, data_loader.last_target

    train_usage_sample = []
    for batch_step in range(total_steps):
        f, t = batch_step * batch_size, (batch_step + 1) * batch_size
        item_batch = train_set[f:t, :]  # [B, L+1]

        context_batch = item_batch[:, :-1]  # [B, L]

        pos_target = item_batch[:, -1:]  # [B, 1]
        neg_target_train = np.array([[random_neg(first, last, s[0])] for s in pos_target])

        neg_target_test = [random_negs(l=1, r=data_loader.target_nums, size=n_neg, pos=s[0]) for s in pos_target]
        target = np.concatenate([neg_target_test, pos_target], 1)  # [n_neg*NEG+POS]

        # [B, n_neg + 1], [B, #Blocks]
        [soft_probs, soft_action] = sess.run(
            [target_model.test_probs, policy_model.test_action],
            feed_dict={
                source_model.input_source_test: context_batch,
                policy_model.input: context_batch,
                policy_model.method: np.array(0),
                policy_model.sample_action: np.ones((batch_size, action_nums)),
                target_model.input_test: target,
            },
        )
        # [B, n_neg + 1], [B, #Blocks]
        [hard_probs, hard_action] = sess.run(
            [target_model.test_probs, policy_model.test_action],
            feed_dict={
                source_model.input_source_test: context_batch,
                policy_model.input: context_batch,
                policy_model.method: np.array(1),
                policy_model.sample_action: np.ones((batch_size, action_nums)),
                target_model.input_test: target,
            },
        )

        reward_soft = reward_fn(soft_probs, n_neg, soft_action, gamma, k=reward_k)
        reward_hard = reward_fn(hard_probs, n_neg, hard_action, gamma, k=reward_k)
        reward_train = reward_soft - reward_hard

        _, _, action, loss, rl_loss = sess.run(
            [train_rl, train_finetune, policy_model.train_action, target_model.train_loss, policy_model.rl_loss],
            feed_dict={
                source_model.input_source_train: context_batch,
                policy_model.input: context_batch,
                policy_model.method: np.array(-1),
                policy_model.sample_action: soft_action,
                policy_model.reward: reward_train,
                target_model.input_train_pos: pos_target,
                target_model.input_train_neg: neg_target_train,
            },
        )

        train_usage_sample.extend(np.array(action).tolist())
        if (batch_step + 1) % log_meter == 0:
            logging.info(
                "\t<{:5d}/{:5d}> Loss: {:.4f}, RL-Loss: {:+.4f}, Reward-Avg: {:+.4f}".format(
                    batch_step + 1, total_steps, loss, rl_loss, np.mean(reward_train)
                )
            )

    summary_block(train_usage_sample, len(configs["dilations"]), "Train")


def train_rl_off():
    batch_size = configs["batch_size"]
    log_meter = configs["log_every"]

    total_steps = int(train_set.shape[0] / batch_size)

    action_nums = len(configs["dilations"])
    first, last = data_loader.first_target, data_loader.last_target

    train_usage_sample = []
    for batch_step in range(total_steps):
        f, t = batch_step * batch_size, (batch_step + 1) * batch_size
        item_batch = train_set[f:t, :]  # [B, L+1]

        context_batch = item_batch[:, :-1]  # [B, L]

        pos_target = item_batch[:, -1:]  # [B, 1]
        neg_target = np.array([[random_neg(first, last, s[0])] for s in pos_target])

        hard_action = sess.run(
            policy_model.test_action,
            feed_dict={
                policy_model.input: context_batch,
                policy_model.method: np.array(1),
                policy_model.sample_action: np.ones((batch_size, action_nums)),
            },
        )

        _, action, loss = sess.run(
            [train_finetune, policy_model.train_action, target_model.train_loss],
            feed_dict={
                source_model.input_source_train: context_batch,
                policy_model.input: context_batch,
                policy_model.method: np.array(-1),
                policy_model.sample_action: hard_action,
                target_model.input_train_pos: pos_target,
                target_model.input_train_neg: neg_target,
            },
        )

        train_usage_sample.extend(np.array(action).tolist())
        if (batch_step + 1) % log_meter == 0:
            logging.info("\t<{:5d}/{:5d}> Loss: {:.4f}".format(batch_step + 1, total_steps, loss))

    summary_block(train_usage_sample, action_nums, "Train")


def evaluate():
    batch_size = configs["batch_size"]
    n_neg = configs["n_neg"]

    total_steps = int(test_set.shape[0] / batch_size)
    action_nums = len(configs["dilations"])

    meter = SRSMetric(k_list=[5, 20])
    meter.setup_and_clean()

    test_usage_sample = []
    for batch_step in range(total_steps):
        f, t = batch_step * batch_size, (batch_step + 1) * batch_size
        batch = test_set[f:t, :]  # [B, L+1]

        context = batch[:, :-1]
        pos_target = batch[:, -1:]
        neg_target = [random_negs(l=1, r=data_loader.target_nums, size=n_neg, pos=s[0]) for s in pos_target]
        target = np.concatenate([neg_target, pos_target], 1)  # [n_neg*neg+pos]

        test_probs, action = sess.run(
            [target_model.test_probs, policy_model.test_action],
            feed_dict={
                source_model.input_source_test: context,
                policy_model.input: context,
                policy_model.method: np.array(1),
                policy_model.sample_action: np.ones((batch_size, action_nums)),
                target_model.input_test: target,
            },
        )
        ground_truth = [[n_neg]] * batch_size
        meter.submit(test_probs, ground_truth)

        test_usage_sample.extend(np.array(action).tolist())

    summary_block(test_usage_sample, len(configs["dilations"]), "Test")

    meter.calc()
    meter.output_to_logger()

    return meter.mrr[5]


def start():
    logging.info("Source Domain Item Size: {}".format(data_loader.source_nums))
    logging.info("Target Domain Item Size: {}".format(data_loader.target_nums))

    logging.info("Training Set Size: {}".format(len(train_set)))
    logging.info("Test Set Size: {}".format(len(test_set)))

    model_save_path = configs["store_path"]
    total_iter = configs["iter"]

    saver = tf.train.Saver(max_to_keep=1)
    best_mrr_at5 = 0.0

    for idx in range(0, total_iter):
        logging.info("-" * 30)
        tic = time.time()
        # --------------------------------------------
        if idx >= configs["rl_start_iter"]:
            logging.info("[RL - ON]  Iter: {} / {}".format(idx + 1, total_iter))
            train_rl_on()
        else:
            logging.info("[RL - OFF] Iter: {} / {}".format(idx + 1, total_iter))
            train_rl_off()

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
    sess.run(tf.assign(policy_model.item_embedding, source_model.item_embedding))

    logging.info("Restore and rebuild finish.")


def parse_arg(args_inline=None, **patch):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--pre", type=str, help="/path/to/model.tfkpt")

    parser.add_argument("--type", choices=["conv", "gru"], required=True, help="policy type")

    parser.add_argument("--gamma", type=float, default=1.0, help="RL training gamma.")
    parser.add_argument("--reward_k", type=int, default=5, help="Top-K to reward.")
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
    parser.add_argument("--rl_start_iter", type=int, default=5)

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
        # "log_every": 50,
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
            policy_model = PolicyNetRL(configs)
        elif policy_type == "gru":
            logging.info("Policy type: << gru >>")
            policy_model = PolicyNetGruRL(configs)
        else:
            raise ValueError("Unexpected policy type: `{}`".format(policy_type))

    with tf.variable_scope(tf.get_variable_scope()):
        source_model = NextItNetPolicy(configs)
        source_model.build_train_graph(policy_model.train_action)
        source_model.build_test_graph(policy_model.test_action)
    with tf.variable_scope("target"):
        target_model = TargetEmbeddingBPR(configs)
        target_model.build_train_graph(source_model.hidden_train)
        target_model.build_test_graph(source_model.hidden_test)
    load_parameters()

    # create optimizer

    v_policy, v_target, _, v_source_active, v_source_other, _ = get_parameters()
    all_active_vars = v_policy + v_target + v_source_active + v_source_other

    optimizer_finetune = tf.train.AdamOptimizer(configs["lr"], name="adam_finetune")
    optimizer_rl = tf.train.AdamOptimizer(configs["lr"], name="adam_rl")

    train_finetune = optimizer_finetune.minimize(target_model.train_loss, var_list=all_active_vars)
    train_rl = optimizer_finetune.minimize(policy_model.rl_loss, var_list=all_active_vars)

    sess.run(tf.variables_initializer(optimizer_finetune.variables() + optimizer_rl.variables()))

    start()
    sess.close()
