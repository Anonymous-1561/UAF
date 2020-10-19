import tensorflow as tf

from modules.gumbel_softmax import gumbel_softmax_, gumbel_softmax_v2
from modules import blocks


class PolicyNet:
    def __init__(self, args):
        self.temp = args["temp"]
        self.kernel_size = args["kernel_size"]
        self.item_size = args["item_size"]
        self.channels = args["channel"]
        self.blocks = args["block_shape"]
        self.action_num = len(args["dilations"])

        self.hard_policy = args["method"] == "hard"

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
        self.logits_train, self.actions_train = self.build_policy(self.input, train=True)
        self.logits_test, self.actions_test = self.build_policy(self.input, train=False)

    def build_policy(self, context, train):
        hidden = self._expand_model_graph(context, train=train)  # # [B, L, C]

        hidden = tf.reduce_mean(hidden, axis=1)  # [B, C]
        logits = tf.matmul(hidden, self.softmax_w, transpose_b=True)  # [B, C] * [C, A*2]=[B,#block*2]
        logits = tf.nn.bias_add(logits, self.output_bias)  # [B,#block*2]
        logits = tf.reshape(logits, [-1, self.action_num, 2])  # [B, #block, 2]

        if self.hard_policy:
            logits = tf.sigmoid(logits)
            action = gumbel_softmax_(logits, temperature=self.temp, hard=True)  # [B, #block, 2]
        else:  # soft policy
            action = tf.sigmoid(logits)  # [B, #block, 2]

        logits_check = tf.nn.softmax(logits)
        action_predict = action[:, :, 0]  # [B, #block]
        return logits_check, action_predict

    def _expand_model_graph(self, context, train):
        hidden = tf.nn.embedding_lookup(self.item_embedding, context, name="context_embedding")

        for layer_id, dilation in enumerate(self.blocks):
            hidden = blocks.res_block(
                hidden, dilation, layer_id, self.channels, self.kernel_size, causal=True, train=train
            )

        return hidden


class PolicyNetGru:
    def __init__(self, args):
        self.temp = args["temp"]
        self.item_size = args["item_size"]
        self.channels = args["channel"]
        self.action_num = len(args["dilations"])

        self.hard_policy = args["method"] == "hard"

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

        self.logits_train, self.actions_train = self.build_policy(self.input)
        self.logits_test, self.actions_test = self.build_policy(self.input)

    def build_policy(self, context):
        hidden = tf.nn.embedding_lookup(self.item_embedding, context, name="context_embedding")  # [B, L, C]
        hidden = self.gru(hidden)  # [B, C]

        logits = tf.matmul(hidden, self.softmax_w, transpose_b=True)  # [B, C] * [C, A*2]=[B,#block*2]
        logits = tf.nn.bias_add(logits, self.output_bias)  # [B,#block*2]
        logits = tf.reshape(logits, [-1, self.action_num, 2])  # [B, #block, 2]

        if self.hard_policy:
            logits = tf.sigmoid(logits)
            action = gumbel_softmax_(logits, temperature=self.temp, hard=True)  # [B, #block, 2]
        else:  # soft policy
            action = tf.sigmoid(logits)  # [B, #block, 2]

        logits_check = tf.nn.softmax(logits)
        action_predict = action[:, :, 0]  # [B, #block]
        return logits_check, action_predict


class PolicyNetRandom:
    def __init__(self, args, prob=0.5):
        self.prob = prob
        self.action_num = len(args["dilations"])

        self.input = tf.placeholder("int32", [None, None], name="item_seq_input")  # [B, SessionLen]

        self.actions_train = self.build_policy()
        self.actions_test = self.build_policy()

    def build_policy(self):
        batch_size = tf.cast(tf.shape(self.input)[0], dtype=tf.int32)
        num_blocks = self.action_num
        probs = tf.ones([batch_size, num_blocks], dtype=tf.float32) * self.prob
        action = tf.cast(tf.distributions.Bernoulli(probs=probs).sample(), tf.float32)
        return action
