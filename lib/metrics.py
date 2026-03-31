import numpy as np
import pandas as pd


def IC(y_true,y_pred):
    #mask = ~np.isnan(y_true)
    #df = pd.DataFrame({'pred':y_pred[mask], 'label':y_true[mask]})
    df = pd.DataFrame({'pred':y_pred.flatten(), 'label':y_true})
    ic = df['pred'].corr(df['label'])
    return ic

def RIC(y_true,y_pred):
    #mask = ~np.isnan(y_true)
    #df = pd.DataFrame({'pred':y_pred[mask], 'label':y_true[mask]})
    df = pd.DataFrame({'pred':y_pred.flatten(), 'label':y_true})
    ric = df['pred'].corr(df['label'],method='spearman')
    return ric

# def PrecN(y_true, y_pred, topn, null_val=-1234):
#     mask = np.not_equal(y_true, null_val)
#     rank_gt = np.argsort(y_true)
#     pre_topN=set()
#     for j in range(1, y_pred.shape[0]+1):
#         cur_rank=rank_gt[-1*j]
#         if not mask[cur_rank]:
#             continue
#         if len(pre_topN)<topn:
#             pre_topN.add(cur_rank)
#     precN = 0.0
#     for pre in pre_topN:
#         precN += (y_pred[pre] >= 0)
#     return precN/topn

def PrecN(y_true, y_pred, topn, null_val=-1234):
    """
    Precision@N:
    1. 按 y_pred 从大到小排序，取预测前 topn 个有效标的
    2. 统计这些标的中 y_true > 0 的比例
    3. 分母固定为 topn

    参数:
        y_true: shape [N,]
        y_pred: shape [N,]
        topn: int
        null_val: 无效标签标记
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    mask = np.not_equal(y_true, null_val)

    # 按预测值从小到大排序，后面倒着取就是从大到小
    rank_pred = np.argsort(y_pred)

    pred_topN = []
    for j in range(1, y_pred.shape[0] + 1):
        cur_rank = rank_pred[-j]
        if not mask[cur_rank]:
            continue
        pred_topN.append(cur_rank)
        if len(pred_topN) == topn:
            break

    precN = 0.0
    for idx in pred_topN:
        precN += (y_true[idx] > 0)

    return precN / topn


def IC_RIC(y_true,y_pred):
    return(
        IC(y_true,y_pred),
        RIC(y_true,y_pred),
    )