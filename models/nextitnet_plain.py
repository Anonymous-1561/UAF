import tensorflow as tf

from modules import blocks
from modules.convs import conv1d


class NextItNet:
    def __init__(self, args):
        # self.args = args
        self.channels = args["channel"]
        self.dilations = args["dilations"]
        self.item_size = args["item_size"]
        self.kernel_size = args["kernel_size"]
        self.using_negative_sampling = args["using_negative_sampling"]
        if self.using_negative_sampling:
            self.negative_sampling_ratio = args["negative_sampling_ratio"]

        self.item_embedding = tf.get_variable(
            "item_embedding",
            [self.item_size, self.channels],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )
        if self.using_negative_sampling:
            self.softmax_w = tf.get_variable(
                "softmax_w",
                [self.item_size, self.channels],
                tf.float32,
                initializer=tf.random_normal_initializer(0.0, 0.01),
            )
            self.softmax_b = tf.get_variable(
                "softmax_b",
                [self.item_size],
                tf.float32,
                initializer=tf.constant_initializer(0.1),
            )

        # TRAIN
        self.input_train = tf.placeholder("int32", [None, None], name="input_train")  # [B, SessionLen]
        self.loss_train, self.hidden_train = self.build_train_graph()

        # TEST
        self.input_test = tf.placeholder("int32", [None, None], name="input_test")  # [B, SessionLen]
        self.loss_test, self.probs_test, self.hidden_test = self.build_test_graph()

    def build_train_graph(self):
        label_seq, dilate_input = self._expand_model_graph(self.input_train, train=True)

        if self.using_negative_sampling:
            logits_2d = tf.reshape(dilate_input, [-1, self.channels])

            label_flat = tf.reshape(label_seq, [-1, 1])
            num_sampled = int(self.negative_sampling_ratio * self.item_size)
            loss = tf.nn.sampled_softmax_loss(
                self.softmax_w,
                self.softmax_b,
                labels=label_flat,
                inputs=logits_2d,
                num_sampled=num_sampled,
                num_classes=self.item_size,
            )
        else:
            logits = conv1d(tf.nn.relu(dilate_input), output_channels=self.item_size, name="logits")
            logits_2d = tf.reshape(logits, [-1, self.item_size])
            label_flat = tf.reshape(label_seq, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2d)

        loss_train = tf.reduce_mean(loss)
        return loss_train, dilate_input

    def build_test_graph(self):
        tf.get_variable_scope().reuse_variables()

        # label_seq: [B, SessLen - 1]
        # dilate_input: [B, SessLen - 1, DilatedChannel]

        label_seq, dilate_input = self._expand_model_graph(self.input_test, train=False)
        loss_test, probs_test = self._get_test_result(dilate_input, label_seq)

        return loss_test, probs_test, dilate_input

    def _get_test_result(self, dilate_input, label_seq):
        if self.using_negative_sampling:
            logits_2d = tf.reshape(dilate_input[:, -1:, :], [-1, self.channels])
            logits_2d = tf.matmul(logits_2d, tf.transpose(self.softmax_w))
            logits_2d = tf.nn.bias_add(logits_2d, self.softmax_b)
        else:
            logits = conv1d(tf.nn.relu(dilate_input[:, -1:, :]), self.item_size, name="logits")
            logits_2d = tf.reshape(logits, [-1, self.item_size])

        label_flat = tf.reshape(label_seq[:, -1], [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2d)

        loss = tf.reduce_mean(loss)
        probs = tf.nn.softmax(logits_2d)
        return loss, probs

    def _expand_model_graph(self, item_seq_input, train=True):
        # item_seq_input: input tensor [B, SessLen]
        context_seq = item_seq_input[:, 0:-1]
        label_seq = item_seq_input[:, 1:]

        hidden = tf.nn.embedding_lookup(self.item_embedding, context_seq, name="context_embedding")
        for layer_id, dilation in enumerate(self.dilations):
            hidden = blocks.res_block(
                hidden, dilation, layer_id, self.channels, self.kernel_size, causal=True, train=train
            )

        return label_seq, hidden
