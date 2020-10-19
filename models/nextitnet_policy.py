import tensorflow as tf

from modules import blocks


class NextItNetPolicy:
    def __init__(self, args):
        self.channels = args["channel"]
        self.dilations = args["dilations"]
        self.item_size = args["item_size"]
        self.kernel_size = args["kernel_size"]
        self.batch_size = args["batch_size"]

        self.item_embedding = tf.get_variable(
            "item_embedding",
            [self.item_size, self.channels],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )

        # TRAIN
        self.input_source_train = tf.placeholder("int32", [self.batch_size, None], name="input_train")
        self.hidden_train = None

        # TEST
        self.input_source_test = tf.placeholder("int32", [self.batch_size, None], name="input_test")
        self.hidden_test = None

    def build_train_graph(self, policy_action):
        dilate_input = self._expand_model_graph(self.input_source_train, policy_action, train=True)
        self.hidden_train = dilate_input

    def build_test_graph(self, policy_action):
        tf.get_variable_scope().reuse_variables()
        dilate_input = self._expand_model_graph(self.input_source_test, policy_action, train=False)
        self.hidden_test = dilate_input

    def _expand_model_graph(self, source_seq, policy_action, train=True):
        hidden = tf.nn.embedding_lookup(self.item_embedding, source_seq, name="context_embedding")

        for layer_id, dilation in enumerate(self.dilations):
            action_mask = tf.reshape(policy_action[:, layer_id], [-1, 1, 1])
            active_block = blocks.res_block_freeze(
                hidden,
                dilation,
                layer_id,
                self.channels,
                self.kernel_size,
                causal=True,
                train=train,
                finetune=True,
            )
            freeze_block = blocks.res_block_freeze(
                hidden,
                dilation,
                layer_id,
                self.channels,
                self.kernel_size,
                causal=True,
                train=train,
                finetune=False,
            )
            hidden = action_mask * active_block + (1 - action_mask) * freeze_block

        return hidden
