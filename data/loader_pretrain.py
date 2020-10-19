import logging

import pandas as pd
import os
import numpy as np
from collections import Counter
import pickle


class DataLoaderPretrain:
    def __init__(self, args, threshold=None):
        self._split_ratio = args["split_ratio"]
        self._save_folder = args["store_path"]

        folder = args["data_folder"]
        file_name = args["data_file"]
        data_file = os.path.join(folder, file_name)

        data = pd.read_csv(data_file, sep=",", header=None)

        counter = Counter()
        for x in data:
            counter.update(data[x].value_counts().to_dict())

        if threshold is None:
            logging.info("No threshold")
            freq_list = [k for k, v in counter.most_common()]
            freq_arr = [v for k, v in counter.most_common()]
            self._freq_arr = freq_arr
        else:
            logging.info("Threshold: {}".format(threshold))
            freq_list = [k for k, v in counter.most_common() if v >= threshold]

        table = {ele: i + 1 for i, ele in enumerate(freq_list)}
        table.update({"<UNK>": 0})

        for x in data:
            data[x] = data[x].map(table)

        data = data.to_numpy(dtype=np.int)

        self._item = data
        self._item_dict = table

    def save_dict(self):
        save_path = os.path.join(self._save_folder, "itemdict.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self._item_dict, f, pickle.HIGHEST_PROTOCOL)

    def save_freq_arr(self, name):
        save_path = os.path.join(self._save_folder, "{}-freq.pkl".format(name))
        with open(save_path, "wb") as f:
            pickle.dump(self._freq_arr, f, pickle.HIGHEST_PROTOCOL)
        return save_path

    @property
    def item_nums(self):
        return len(self._item_dict)

    @property
    def session_length(self):
        return np.shape(self._item)[1]

    def split(self):
        data_size = len(self._item)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        self._item = self._item[shuffle_indices]

        test_idx = -1 * int(self._split_ratio * float(data_size))
        train_set, test_set = self._item[:test_idx], self._item[test_idx:]
        return train_set, test_set

    def generate_sub_sessions(self, data, pad_token):
        pad_idx = self._item_dict[pad_token]
        sub_seq_data = []

        sess_size = len(data[0])

        for i in range(len(data)):
            seq = data[i]
            for j in range(sess_size - 1):  # minimal size 2
                sub_seq_end = seq[: len(seq) - j]
                sub_seq_pad = [pad_idx] * j
                sub_seq_data.append(list(sub_seq_pad) + list(sub_seq_end))
        x_train = np.array(sub_seq_data)
        del sub_seq_data

        return x_train
