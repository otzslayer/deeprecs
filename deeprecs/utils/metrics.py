import numpy as np


def hit(true_item, pred_items):
    r"""Hit Ratio 결과를 리턴합니다."""
    if true_item in pred_items:
        return 1
    return 0


def ndcg(true_item, pred_items):
    r"""NDCG 결과를 리턴합니다."""
    if true_item in pred_items:
        index = pred_items.index(true_item)
        return np.reciprocal(np.log2(index + 2))
    return 0
