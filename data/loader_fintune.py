import pickle

import pandas as pd
import os
import numpy as np
from collections import Counter


class DataLoaderFinetunePlain:
    """
    This is dataloader for all datasets.
    Caution:
        1. use `threshold` to reduce size of item to avoid OOM
        2. set `target_use_unk=False` in classification tasks
        3. - set `args[pre]` if you have `itemdict.pkl` in that folder
           - or leave it blank to build item dict from scratch
    """

    def __init__(self, args, threshold=None, target_use_unk=True):
        self._split_ratio = args["split_ratio"]
        self._folder = args["data_folder"]
        self._file_name = args["data_file"]
        self._target_use_unk = target_use_unk

        if "pre" in args and args["pre"] is not None:
            self._from_pre = True
            self._pretrain_folder = args["pre"]
        else:
            self._from_pre = False

        self.UNK = "<UNK>"
        self.source_threshold = threshold

        self.source = None
        self.source_dict = None

        self.target = None
        self.target_dict = None

        self.samples = None

    def build(self):
        data_file = os.path.join(self._folder, self._file_name)
        full_data = list(open(data_file, "r").readlines())

        sep = ",,"
        source_data, target_data = [], []
        for s in full_data:
            s0, s1 = s.split(sep)
            if s1.endswith("\n"):
                s1 = s1[:-1]
            source_data.append(s0.split(","))
            target_data.append(s1.split(","))

        if self._from_pre:
            self.source_dict = self.load_dict()
            self.source = self.do_transform(self.source_dict, source_data)
        else:
            self.source, self.source_dict = self.build_mapping_dict(
                source_data, use_unk=True, threshold=self.source_threshold
            )

        self.target, self.target_dict = self.build_mapping_dict(target_data, use_unk=self._target_use_unk)

        self.samples = []
        for line in range(len(self.source)):
            source_line = self.source[line]
            target_line = self.target[line]
            for target in target_line:
                # using UNK => ignore 0
                # otherwise => all, in cls tasks
                if target != 0 or not self._target_use_unk:
                    unit = np.array(source_line)
                    unit = np.append(unit, target)
                    self.samples.append(unit)

        self.samples = np.array(self.samples)

    def build_mapping_dict(self, list_data, use_unk, threshold=None, return_counter=False):
        data = pd.DataFrame(list_data)
        data = data.fillna(self.UNK)

        counter = Counter()
        for x in data:
            counter.update(data[x].value_counts().to_dict())
        del counter[self.UNK]

        if threshold is None:
            freq_list = [k for k, v in counter.most_common()]
        else:
            freq_list = [k for k, v in counter.most_common() if v >= threshold]

        if use_unk:
            table = {ele: i + 1 for i, ele in enumerate(freq_list)}  # [0] is reserved for <UNK>
            table.update({self.UNK: 0})
        else:
            table = {ele: i for i, ele in enumerate(freq_list)}  # no UNK

        data = self.do_transform(table, data)

        if not return_counter:
            return data, table
        else:
            return data, table, counter

    def load_dict(self):
        save_path = os.path.join(self._pretrain_folder, "itemdict.pkl")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def do_transform(table, data):
        _data = pd.DataFrame(data)
        for x in _data:
            _data[x] = _data[x].map(table)
        _data = _data.fillna(0)  # if item not in table, convert to `<UNK>`
        _data = _data.to_numpy(dtype=np.int)
        return _data

    def split(self):
        data_size = len(self.samples)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        self.samples = self.samples[shuffle_indices]

        test_idx = -1 * int(self._split_ratio * float(data_size))
        train_set, test_set = self.samples[:test_idx], self.samples[test_idx:]
        return train_set, test_set

    @property
    def source_nums(self):
        return len(self.source_dict)

    @property
    def target_nums(self):
        return len(self.target_dict)

    @property
    def first_target(self):
        if self._target_use_unk:
            return 1
        else:
            return 0

    @property
    def last_target(self):
        if self._target_use_unk:
            return len(self.target_dict)
        else:
            return len(self.target_dict) - 1

    @property
    def source_data_shape(self):
        return np.shape(self.source)

    @property
    def target_data_shape(self):
        return np.shape(self.target)

    @property
    def samples_data_shape(self):
        return np.shape(self.samples)

    @property
    def samples_session_len(self):
        return len(self.samples[0])
