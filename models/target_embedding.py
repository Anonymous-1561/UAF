import tensorflow as tf

from modules.convs import conv1d


class TargetEmbeddingBPR:
    def __init__(self, args):
        self.target_item_size = args["target_item_size"]
        self.channel = args["channel"]

        self.target_item_embedding = tf.get_variable(
            "target_item_embedding",
            [self.target_item_size, self.channel],
            initializer=tf.truncated_normal_initializer(stddev=0.01),
        )
        # Training
        self.input_train_pos = tf.placeholder("int32", [None, None], name="input_train_positive")  # [B, 1]
        self.input_train_neg = tf.placeholder("int32", [None, None], name="input_train_negative")  # [B, 1]
        self.train_loss = None

        # Testing
        self.input_test = tf.placeholder("int32", [None, None], name="input_test")  # [B, #neg+1]
        self.test_probs = None

    def build_train_graph(self, source_hidden):
        # source_hidden: [B, L, C]
        source_emb = tf.reduce_mean(source_hidden, 1)  # [B, C]

        pos_emb = tf.nn.embedding_lookup(self.target_item_embedding, self.input_train_pos, name="pos_emb")  # [B, 1, C]
        neg_emb = tf.nn.embedding_lookup(self.target_item_embedding, self.input_train_neg, name="neg_emb")  # [B, 1, C]

        pos_score = source_emb * tf.reshape(pos_emb, [-1, self.channel])  # [B, C]
        neg_score = source_emb * tf.reshape(neg_emb, [-1, self.channel])  # [B, C]

        pos_logits = tf.reduce_sum(pos_score, -1)  # [B]
        neg_logits = tf.reduce_sum(neg_score, -1)  # [B]

        self.train_loss = tf.reduce_mean(
            -tf.log(tf.sigmoid(pos_logits) + 1e-24) - tf.log(1 - tf.sigmoid(neg_logits) + 1e-24)
        )

    def build_test_graph(self, source_hidden):
        tf.get_variable_scope().reuse_variables()

        target_emb = tf.nn.embedding_lookup(
            self.target_item_embedding, self.input_test, name="target_emb"
        )  # [B, #neg+1, C]
        target_emb = tf.transpose(target_emb, [0, 2, 1])  # [B, C, #neg+1]

        source_emb = tf.reduce_mean(source_hidden, 1)  # [B, C]
        source_emb = tf.expand_dims(source_emb, 1)  # [B, 1, C]

        score_test = tf.matmul(source_emb, target_emb)  # [B, 1, C] * [B, C, #neg+1] = [B, 1, #neg+1]
        self.test_probs = tf.squeeze(score_test)  # [B, #neg+1]


class TargetEmbeddingClassifier:
    def __init__(self, args):
        self.target_item_size = args["target_item_size"]
        self.channel = args["channel"]

        self.target_item_embedding = tf.get_variable(
            "target_item_embedding",
            [self.target_item_size, self.channel],
            initializer=tf.truncated_normal_initializer(stddev=0.01),
        )
        # Training
        self.input_train_pos = tf.placeholder("int32", [None, None], name="input_train_positive")  # [B, 1]
        self.input_train_neg = tf.placeholder("int32", [None, None], name="input_train_negative")  # [B, 1]
        self.train_loss = None

        # Testing
        self.test_probs = None

    def build_train_graph(self, source_hidden):
        # source_hidden: [B, L, C]
        source_emb = tf.reduce_mean(source_hidden, 1)  # [B, C]

        pos_emb = tf.nn.embedding_lookup(self.target_item_embedding, self.input_train_pos, name="pos_emb")  # [B, 1, C]
        neg_emb = tf.nn.embedding_lookup(self.target_item_embedding, self.input_train_neg, name="neg_emb")  # [B, 1, C]

        pos_score = source_emb * tf.reshape(pos_emb, [-1, self.channel])  # [B, C]
        neg_score = source_emb * tf.reshape(neg_emb, [-1, self.channel])  # [B, C]

        pos_logits = tf.reduce_sum(pos_score, -1)  # [B]
        neg_logits = tf.reduce_sum(neg_score, -1)  # [B]

        # BPR Loss
        self.train_loss = -tf.reduce_mean(tf.log(tf.sigmoid(pos_logits - neg_logits))) + 1e-24

    def build_test_graph(self, source_hidden):
        tf.get_variable_scope().reuse_variables()
        # [B, L, C] => [B, C]
        source_item_embedding = tf.reduce_mean(source_hidden, 1)
        # [B, C] * [C, #TargetItem] => [B, #TargetItem]
        logits_2d = tf.matmul(source_item_embedding, tf.transpose(self.target_item_embedding))
        self.test_probs = logits_2d


class TargetEmbeddingSoftmax:
    def __init__(self, args):
        self.target_item_size = args["target_item_size"]
        self.channel = args["channel"]

        self.target_item_embedding = tf.get_variable(
            "target_item_embedding",
            [self.target_item_size, self.channel],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )  # [Ts, C]
        # Training
        self.target = tf.placeholder("int32", [None, None], name="target_ground_truth")  # [B, 1]
        self.train_loss = None  # after `build_train` function

        # Testing
        self.test_probs = None  # after `build_test` function

    def build_train_graph(self, source_hidden):
        # source_hidden: [B, L, C]
        logits = conv1d(tf.nn.relu(source_hidden[:, -1:, :]), output_channels=self.target_item_size, name="logits")
        logits_2d = tf.reshape(logits, [-1, self.target_item_size])

        label_flat = tf.reshape(self.target, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2d)

        self.train_loss = tf.reduce_mean(loss)

    def build_test_graph(self, source_hidden):
        tf.get_variable_scope().reuse_variables()

        logits = conv1d(tf.nn.relu(source_hidden[:, -1:, :]), self.target_item_size, name="logits")
        logits_2d = tf.reshape(logits, [-1, self.target_item_size])

        self.test_probs = tf.nn.softmax(logits_2d)
