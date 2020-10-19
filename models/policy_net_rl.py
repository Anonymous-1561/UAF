import tensorflow as tf

from modules.gumbel_softmax import gumbel_softmax_v2
from modules import blocks


class PolicyNetRL:
    def __init__(self, args):
        self.temp = args["temp"]
        self.kernel_size = args["kernel_size"]
        self.item_size = args["item_size"]
        self.channels = args["channel"]
        self.blocks = args["block_shape"]
        self.action_num = len(args["dilations"])

        self.item_embedding = tf.get_variable(
            "item_embedding",
            shape=[self.item_size, self.channels],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )

        self.softmax_w = tf.get_variable(
            "softmax_w",
            shape=[self.action_num * 2, self.channels],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0.0, 0.01),
        )
        self.output_bias = tf.get_variable(
            "softmax_b",
            shape=[self.action_num * 2],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0.0, 0.01),
        )

        self.input = tf.placeholder("int32", [None, None], name="item_seq_input")  # [B, L]
        self.method = tf.placeholder("int32", name="chosen_method")
        self.sample_action = tf.placeholder("float32", [None, None], name="sample_action")
        self.reward = tf.placeholder("float32", [None], name="reward_input")

        self.test_action = self.build_policy(self.input, train=False)
        self.train_action, self.rl_loss = self.build_policy(self.input, train=True)

    def build_policy(self, context, train):
        hidden = self._expand_model_graph(context, train=train)  # # [B, L, C]

        hidden = tf.reduce_mean(hidden, axis=1)  # [B, C]
        logits = tf.matmul(hidden, self.softmax_w, transpose_b=True)  # [B, C] * [C, A*2]=[B,#block*2]
        logits = tf.nn.bias_add(logits, self.output_bias)  # [B,#block*2]
        logits = tf.reshape(logits, [-1, self.action_num, 2])  # [B, #block, 2]
        logits = tf.sigmoid(logits)

        hard_action, _ = gumbel_softmax_v2(logits, hard=True)
        soft_action, _ = gumbel_softmax_v2(logits, hard=False)
        given_action, _ = gumbel_softmax_v2(logits, given_action=self.sample_action)

        action_predict = tf.case(
            {
                tf.equal(self.method, 1): lambda: hard_action[:, :, 0],
                tf.equal(self.method, 0): lambda: soft_action[:, :, 0],
                tf.less(self.method, 0): lambda: given_action[:, :, 0],
            },
            name="condition_action_predict",
            exclusive=True,
        )

        if not train:
            return action_predict

        sample_action = tf.expand_dims(self.sample_action, -1)
        ont_hot_sample_action = tf.concat([sample_action, 1 - sample_action], axis=-1)
        entropy = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.reshape(ont_hot_sample_action, [-1, 2]),
            logits=tf.reshape(logits, [-1, 2]),
            reduction=tf.losses.Reduction.NONE,
        )
        entropy = tf.reshape(entropy, [-1, self.action_num])
        rl_loss = tf.reduce_mean(tf.reduce_sum(entropy * tf.expand_dims(self.reward, axis=-1), axis=-1))

        return action_predict, rl_loss

    def _expand_model_graph(self, context, train):
        hidden = tf.nn.embedding_lookup(self.item_embedding, context, name="context_embedding")

        for layer_id, dilation in enumerate(self.blocks):
            hidden = blocks.res_block(
                hidden, dilation, layer_id, self.channels, self.kernel_size, causal=True, train=train
            )

        return hidden


class PolicyNetGruRL:
    def __init__(self, args):
        self.temp = args["temp"]
        self.item_size = args["item_size"]
        self.channels = args["channel"]
        self.action_num = len(args["dilations"])

        self.gru = tf.keras.layers.GRU(self.channels)

        self.item_embedding = tf.get_variable(
            "item_embedding",
            shape=[self.item_size, self.channels],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )

        self.softmax_w = tf.get_variable(
            "softmax_w",
            shape=[self.action_num * 2, self.channels],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0.0, 0.01),
        )
        self.output_bias = tf.get_variable(
            "softmax_b",
            shape=[self.action_num * 2],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0.0, 0.01),
        )

        self.input = tf.placeholder("int32", [None, None], name="item_seq_input")  # [B, L]
        self.method = tf.placeholder("int32", name="chosen_method")
        self.sample_action = tf.placeholder("float32", [None, None], name="sample_action")
        self.reward = tf.placeholder("float32", [None], name="reward_input")

        self.test_action = self.build_policy(self.input, train=False)
        self.train_action, self.rl_loss = self.build_policy(self.input, train=True)

    def build_policy(self, context, train):
        hidden = tf.nn.embedding_lookup(self.item_embedding, context, name="context_embedding")  # [B, L, C]
        hidden = self.gru(hidden)  # [B, C]

        logits = tf.matmul(hidden, self.softmax_w, transpose_b=True)  # [B, C] * [C, A*2]=[B,#block*2]
        logits = tf.nn.bias_add(logits, self.output_bias)  # [B,#block*2]
        logits = tf.reshape(logits, [-1, self.action_num, 2])  # [B, #block, 2]
        logits = tf.sigmoid(logits)

        hard_action, _ = gumbel_softmax_v2(logits, hard=True)
        soft_action, _ = gumbel_softmax_v2(logits, hard=False)
        given_action, _ = gumbel_softmax_v2(logits, given_action=self.sample_action)

        action_predict = tf.case(
            {
                tf.equal(self.method, 1): lambda: hard_action[:, :, 0],
                tf.equal(self.method, 0): lambda: soft_action[:, :, 0],
                tf.less(self.method, 0): lambda: given_action[:, :, 0],
            },
            name="condition_action_predict",
            exclusive=True,
        )

        if not train:
            return action_predict

        sample_action = tf.expand_dims(self.sample_action, -1)
        ont_hot_sample_action = tf.concat([sample_action, 1 - sample_action], axis=-1)
        entropy = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.reshape(ont_hot_sample_action, [-1, 2]),
            logits=tf.reshape(logits, [-1, 2]),
            reduction=tf.losses.Reduction.NONE,
        )
        entropy = tf.reshape(entropy, [-1, self.action_num])
        rl_loss = tf.reduce_mean(tf.reduce_sum(entropy * tf.expand_dims(self.reward, axis=-1), axis=-1))

        return action_predict, rl_loss
