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

def PrecN(y_true, y_pred, topn):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) == 0:
        return np.nan

    rank_pred = np.argsort(y_pred)[::-1]
    top_idx = rank_pred[:topn]

    if len(top_idx) == 0:
        return np.nan

    return np.mean(y_true[top_idx] >= 0)



def IC_RIC(y_true,y_pred):
    return(
        IC(y_true,y_pred),
        RIC(y_true,y_pred),
    )

def RMSE(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return np.nan

    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def MAE(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return np.nan

    return np.mean(np.abs(y_pred - y_true))

def MAPE(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    valid = np.abs(y_true) > eps
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return np.nan
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def ACC(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return np.nan
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def MAE_MAPE_RMSE_ACC(y_true,y_pred):
    return(
        MAE(y_true,y_pred),
        MAPE(y_true,y_pred),
        RMSE(y_true,y_pred),
        ACC(y_true, y_pred),
    )