import logging

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from utils.tools import tabular_pretty_print


def sample_top_k_old(a, top_k):
    idx = np.argsort(a)[:, ::-1]
    idx = idx[:, :top_k]
    return idx


def sample_top_k(a, top_k):
    idx = np.argpartition(a, -top_k)[:, -top_k:]
    part = np.take_along_axis(a, idx, 1)
    return np.take_along_axis(idx, np.argsort(part), 1)[:, ::-1]


def sample_top_ks_old(a, top_ks):
    # O(n * log(n)) + b * O(1)
    idx = np.argsort(a)[:, ::-1]
    for k in top_ks:
        yield idx[:, :k]


def sample_top_ks(a, top_ks):
    # O(b * (n + k * log(k)))
    for k in top_ks:
        idx = np.argpartition(a, -k)[:, -k:]
        part = np.take_along_axis(a, idx, 1)
        yield np.take_along_axis(idx, np.argsort(part), 1)[:, ::-1]


# mrr@K, hit@K, ndcg@k
def get_metric(rank_indices):
    mrr_list, hr_list, ndcg_list = [], [], []
    for t in rank_indices:
        if len(t):
            mrr_list.append(1.0 / (t[0][0] + 1))
            ndcg_list.append(1.0 / np.log2(t[0][0] + 2))
            hr_list.append(1.0)
        else:
            mrr_list.append(0.0)
            ndcg_list.append(0.0)
            hr_list.append(0.0)

    return mrr_list, hr_list, ndcg_list


class SRSMetric:
    def __init__(self, k_list, use_mrr=True, use_hit=True, use_ndcg=True):
        self.k_list = k_list

        self.mrr_list, self.use_mrr = None, use_mrr
        self.hit_list, self.use_hit = None, use_hit
        self.ndcg_list, self.use_ndcg = None, use_ndcg

        self.mrr = None
        self.hit = None
        self.ndcg = None

    def setup_and_clean(self):
        if self.use_mrr:
            self.mrr = {}
            self.mrr_list = {}
            self._setup_one(self.mrr_list)
        if self.use_hit:
            self.hit = {}
            self.hit_list = {}
            self._setup_one(self.hit_list)
        if self.use_ndcg:
            self.ndcg = {}
            self.ndcg_list = {}
            self._setup_one(self.ndcg_list)

    def _setup_one(self, obj):
        for k in self.k_list:
            obj[k] = []

    @staticmethod
    def _get_idx(argsort_res, real_idx):
        equ_array = argsort_res == real_idx
        row_idx = np.argmax(equ_array, 1)
        row_idx[np.any(equ_array, 1)] += 1

        return row_idx

    def submit(self, predict_probs, real_idx):
        # predict_probs [B, num_items]
        # real_idx      [B, 1]
        predict_probs = np.array(predict_probs)
        for raw, k in zip(sample_top_ks(predict_probs, self.k_list), self.k_list):
            row_idx = self._get_idx(raw, real_idx)

            mrr_list, hit_list, ndcg_list = [], [], []
            for t in row_idx:
                if t:
                    mrr_list.append(1.0 / t)
                    ndcg_list.append(1.0 / np.log2(t + 1))
                    hit_list.append(1.0)
                else:
                    mrr_list.append(0.0)
                    ndcg_list.append(0.0)
                    hit_list.append(0.0)

            if self.use_mrr:
                self.mrr_list[k].extend(mrr_list)
            if self.use_hit:
                self.hit_list[k].extend(hit_list)
            if self.use_ndcg:
                self.ndcg_list[k].extend(ndcg_list)

    def calc(self):
        if self.use_mrr:
            self._calc_one(self.mrr, self.mrr_list)
        if self.use_hit:
            self._calc_one(self.hit, self.hit_list)
        if self.use_ndcg:
            self._calc_one(self.ndcg, self.ndcg_list)

    def _calc_one(self, score_dict, metric_list):
        for k in self.k_list:
            score_dict[k] = np.mean(metric_list[k])

    def output_to_logger(self, decimal=4):
        def fmt_f(num, d):
            fmt_string = "{{:.{}f}}".format(d)
            return fmt_string.format(num)

        content = [[""]]
        if self.use_mrr:
            content[0].append("MRR")
        if self.use_hit:
            content[0].append("HIT")
        if self.use_ndcg:
            content[0].append("NDCG")

        for k in self.k_list:
            line = ["K={}".format(k)]
            if self.use_mrr:
                line.append(fmt_f(self.mrr[k], decimal))
            if self.use_hit:
                line.append(fmt_f(self.hit[k], decimal))
            if self.use_ndcg:
                line.append(fmt_f(self.ndcg[k], decimal))
            content.append(line)

        lines = tabular_pretty_print(content)
        for line in lines:
            logging.info(line)


class ClsMetric:
    def __init__(self, num_cls, use_acc=True, use_f1=True, use_prec=True, use_recall=True, use_cm=False):
        self.num_cls = num_cls

        # self.pred_list, self.score_list, self.real_list = None, None, None
        self.pred_list, self.real_list = None, None

        self.use_acc = use_acc
        self.use_f1 = use_f1
        self.use_prec = use_prec
        self.use_recall = use_recall
        self.use_cm = use_cm

        self.acc = None
        self.f1 = None
        self.prec = None
        self.recall = None
        self.cm = None

    def setup_and_clean(self):
        self.acc = None
        self.f1 = None
        self.prec = None
        self.recall = None
        self.cm = None

        # self.pred_list, self.score_list, self.real_list = [], [], []
        self.pred_list, self.real_list = [], []

    @staticmethod
    def squeeze_atleast_1d(arr):
        _arr = np.squeeze(arr)
        return np.atleast_1d(_arr)

    def submit(self, predict_probs, real_idx):
        # probs: [B, num_cls]
        # idx:   [B, 1]
        predict_probs = np.array(predict_probs)
        real_idx = np.array(real_idx)
        idx_sort = np.argsort(predict_probs)[:, -1:]

        self.pred_list.extend(self.squeeze_atleast_1d(idx_sort))
        self.real_list.extend(self.squeeze_atleast_1d(real_idx))
        # self.score_list.extend(self.squeeze_atleast_1d(np.take_along_axis(predict_probs, real_idx, axis=1)))

    def calc(self):
        if self.use_acc:
            self.acc = accuracy_score(self.real_list, self.pred_list)
        if self.use_f1:
            self.f1 = f1_score(self.real_list, self.pred_list, average="binary" if self.num_cls == 2 else "macro")
        if self.use_prec:
            self.prec = precision_score(
                self.real_list, self.pred_list, average="binary" if self.num_cls == 2 else "macro"
            )
        if self.use_recall:
            self.recall = recall_score(
                self.real_list, self.pred_list, average="binary" if self.num_cls == 2 else "macro"
            )
        if self.use_cm:
            self.cm = confusion_matrix(self.real_list, self.pred_list)

    def output_to_logger(self, decimal=4, layout="vertical"):
        def fmt_f(num, d):
            fmt_string = "{{:.{}f}}".format(d)
            return fmt_string.format(num)

        if layout == "vertical":
            content = []
            if self.use_acc:
                content.append(["Accuracy", fmt_f(self.acc, decimal)])
            if self.use_prec:
                content.append(["Precision", fmt_f(self.prec, decimal)])
            if self.use_recall:
                content.append(["Recall", fmt_f(self.recall, decimal)])
            if self.use_f1:
                content.append(["F1-Score", fmt_f(self.f1, decimal)])
        elif layout == "horizontal":
            content = [[], []]
            if self.use_acc:
                content[0].append("Accuracy")
            if self.use_prec:
                content[0].append("Precision")
            if self.use_recall:
                content[0].append("Recall")
            if self.use_f1:
                content[0].append("F1-Score")

            if self.use_acc:
                content[1].append(fmt_f(self.acc, decimal))
            if self.use_prec:
                content[1].append(fmt_f(self.prec, decimal))
            if self.use_recall:
                content[1].append(fmt_f(self.recall, decimal))
            if self.use_f1:
                content[1].append(fmt_f(self.f1, decimal))
        else:
            raise ValueError("Unexpected layout `{}`".format(layout))

        lines = tabular_pretty_print(content)
        logging.info("Classification metrics:")
        for line in lines:
            logging.info(line)

        if self.use_cm:
            logging.info("Confusion matrix:")
            content = [[""] + ["c{}".format(i) for i in range(self.num_cls)]]
            for idx, line in enumerate(self.cm):
                t = ["c{}".format(idx)]
                for e in line:
                    t.append(str(e))
                content.append(t)

            lines = tabular_pretty_print(content)
            for line in lines:
                logging.info(line)


if __name__ == "__main__":
    from utils.logger import setup_simple_logger

    setup_simple_logger()

    # tool = SRSMetric(k_list=[1, 2, 3, 4])
    # tool.setup_and_clean()
    # """
    #     0   1   2   3   4   5   6
    #     0.1 0.2 0.1 0.7 0.1 0.3 0.5  = 3
    #     0.2 0.3 0.4 0.2 0.7 0.8 0.1  = 5
    #     """
    # tool.submit(np.array([[0.1, 0.2, 0.1, 0.7, 0.1, 0.3, 0.5]]), [[3]])
    # tool.submit(np.array([[0.2, 0.3, 0.4, 0.2, 0.7, 0.8, 0.1]]), [[4]])
    # tool.calc()
    #
    # tool.output_to_logger()

    cc = ClsMetric(num_cls=3, use_cm=True)
    cc.setup_and_clean()
    pred = [[0.6, 0.2, 0.2], [0.2, 0.2, 0.6], [0.2, 0.6, 0.2], [0.6, 0.2, 0.2], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2]]
    real = [[0], [1], [2], [0], [1], [2]]
    cc.submit(pred[2:4], real[2:4])
    cc.submit(pred[0:2], real[0:2])
    cc.submit(pred[4:], real[4:])
    # cc.submit(pred, real)
    cc.calc()
    cc.output_to_logger(layout="horizontal")
