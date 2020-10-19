import numpy as np


def dict_to_str(data, title=None):
    _title = title if title is not None else "dict"

    s = "\n"
    s += "-----[{}]-----".format(_title)

    len_k = max([len(str(item)) for item in data.keys()])

    for k, v in data.items():
        line = ("{:>%d}" % len_k).format(k)
        s += "\n{} | {}".format(line, v)

    s += "\n"
    s += "-----[End of {}]-----".format(_title)

    return s


def tabular_pretty_print(grid):
    lens = [max(map(len, col)) for col in zip(*grid)]

    fmt = " | ".join("{{:{}}}".format(x) for x in lens)
    table = [fmt.format(*row) for row in grid]

    sep = ["~" * len(table[0])]
    table = sep + table + sep

    res = []
    for idx, line in enumerate(table):
        if idx == 0 or idx == len(table) - 1:
            ps = "\t* {} *".format(line)
        else:
            ps = "\t| {} |".format(line)
        res.append(ps)
    return res


def random_neg(l, r, s):
    probs = np.ones((r - l + 1,), dtype=np.float)
    probs[s - l] = 0.0
    probs = probs / probs.sum()
    neg = np.random.choice(np.arange(l, r + 1), (1,), p=probs)
    return neg[0]


def random_negs(l, r, size, pos):
    """
    Sample `size` elements from [`l`, `r`] exclude element `pos`
    :param l: start
    :param r: end (included)
    :param size: number of sample
    :param pos: positive element, pos in [l, r]
    :return: negative sampled elements
    """
    probs = np.ones((r - l + 1,), dtype=np.float)
    probs[pos - l] = 0.0
    probs = probs / probs.sum()

    negs = np.random.choice(np.arange(l, r + 1), (size,), p=probs)
    return negs


def neg_sampler_with_probs(item_nums, targets, prob):
    # targets: [B, 1]
    neg = []
    for t in np.squeeze(targets):
        p = np.copy(prob)
        p[t] = 0.0
        p = np.array(p)
        p = p / p.sum()
        neg.append(np.random.choice(item_nums, 1, p=p).tolist())
    return neg


if __name__ == "__main__":
    # print neg_sampler_with_probs(5, [[3], [1], [4]], [0.0, 0.1, 0.2, 0.3, 0.4])
    # print random_neg(1, 10, 5)
    print random_negs(1, 10, 3, 5)
