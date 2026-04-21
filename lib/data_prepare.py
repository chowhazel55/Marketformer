import pickle
import math
import numpy as np
import torch
from .utils import _init_data_loader, HISTDataLoader
from torch.utils.data import TensorDataset,DataLoader

def dataloaders_select(dataset,model_name):
    dataset=dataset.upper()
    model_name=model_name.upper()

    if dataset in ("CMINUS"):
        return load_cminus
    elif dataset in ("ACL18"):
        return load_acl
    elif dataset in ('NASDAQ','NYSE'):
        return load_masked_data
    # elif dataset in ('SP500'):
    #     return load_sp
    elif dataset in ("CSI"):
        return load_csi_by_year
    elif dataset in ("CSI300"):
        return load_csi300_by_year
    elif dataset in ("CSI2259"):
        return load_csi2259_by_year
    else:
        raise NotImplementedError

    
def load_masked_data(dataset,batch_size,shuffle_train,n_jobs):
    dataset=dataset.upper()
    with open(f'../data/{dataset}/eod_data.pkl','rb')as f:
        eod_data=pickle.load(f)
    with open(f'../data/{dataset}/gt_data.pkl','rb')as f:
        gt_data=pickle.load(f)
    with open(f'../data/{dataset}/price_data.pkl','rb')as f:
        price_data=pickle.load(f)
    with open(f'../data/{dataset}/mask_data.pkl','rb')as f:
        mask_data=pickle.load(f)
    steps=1
    window=16
    x=[]
    y=[]
    #1245-window-steps+1
    valid_index=756-window+steps-1
    test_index=1008-window+steps-1
    for idx in range(1245-window-steps+1):
        # 16-days feature
        x.append(eod_data[:,idx:idx+window,:])
        # 17-days mask + 1 day base + 1 day gt
        mask_sample=mask_data[:,idx:idx+window+1]
        price_sample=np.expand_dims(price_data[:,idx+window-1],axis=1)
        gt_sample=np.expand_dims(gt_data[:,idx+window+steps-1],axis=1)
        mask_sample=np.min(mask_sample,axis=1)
        final_mask=np.expand_dims(mask_sample, axis=1)
        y.append(np.concatenate((gt_sample,price_sample,final_mask),axis=1))
    x=np.array(x)
    y=np.array(y)
    x_train=x[:valid_index]
    x_valid=x[valid_index:test_index]
    x_test=x[test_index:]
    y_train=y[:valid_index]
    y_valid=y[valid_index:test_index]
    y_test=y[test_index:]

    trainset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    valset = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))
    testset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
    trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train)
    valset_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testset_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainset_loader,valset_loader,testset_loader


def load_cminus(dataset, batch_size, shuffle_train, n_jobs):
    dataset=dataset.upper()
    data = np.load('/home/users/hzf/benchmarks/CMINUS/moving_avg.npy')
    price_data = data[:, :, -1]
    mask_data=np.ones((data.shape[0], data.shape[1]))
    eod_data = data
    gt_data = np.zeros((data.shape[0], data.shape[1]))
    steps=1
    window=16
    for ticket in range(0, data.shape[0]):
        for row in range(1, data.shape[1]):
            gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / \
                                data[ticket][row - steps][-1]  
    print(gt_data.shape)
    x=[]
    y=[]
    for idx in range(data.shape[1]-window-steps+1):
        # 16-days feature
        x.append(eod_data[:,idx:idx+window,:])
        # 17-days mask + 1 day base + 1 day gt
        mask_sample=mask_data[:,idx:idx+window+1]
        price_sample=np.expand_dims(price_data[:,idx+window-1],axis=1)
        gt_sample=np.expand_dims(gt_data[:,idx+window+steps-1],axis=1)
        mask_sample=np.min(mask_sample,axis=1)
        final_mask=np.expand_dims(mask_sample, axis=1)
        y.append(np.concatenate((gt_sample,price_sample,final_mask),axis=1))
    x=np.array(x)
    y=np.array(y)
    test_index = x.shape[0] - 85
    valid_index = test_index - 85
    x_train=x[:valid_index]
    x_valid=x[valid_index:test_index]
    x_test=x[test_index:]
    y_train=y[:valid_index]
    y_valid=y[valid_index:test_index]
    y_test=y[test_index:]

    trainset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    valset = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))
    testset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
    trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train)
    valset_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testset_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainset_loader,valset_loader,testset_loader

def load_acl(dataset, batch_size, shuffle_train, n_jobs):
    dataset=dataset.upper()
    data = np.load('/home/users/hzf/benchmarks/ACL18/moving_avg.npy')
    price_data = data[:, :, -1]
    mask_data=np.ones((data.shape[0], data.shape[1]))
    eod_data = data
    gt_data = np.zeros((data.shape[0], data.shape[1]))
    steps=1
    window=16
    for ticket in range(0, data.shape[0]):
        for row in range(1, data.shape[1]):
            gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / \
                                data[ticket][row - steps][-1]  
    print(gt_data.shape)
    x=[]
    y=[]
    for idx in range(data.shape[1]-window-steps+1):
        # 16-days feature
        x.append(eod_data[:,idx:idx+window,:])
        # 17-days mask + 1 day base + 1 day gt
        mask_sample=mask_data[:,idx:idx+window+1]
        price_sample=np.expand_dims(price_data[:,idx+window-1],axis=1)
        gt_sample=np.expand_dims(gt_data[:,idx+window+steps-1],axis=1)
        mask_sample=np.min(mask_sample,axis=1)
        final_mask=np.expand_dims(mask_sample, axis=1)
        y.append(np.concatenate((gt_sample,price_sample,final_mask),axis=1))
    x=np.array(x)
    y=np.array(y)
    test_index = x.shape[0] - 64
    valid_index = test_index - 42
    x_train=x[valid_index-398:valid_index]
    x_valid=x[valid_index:test_index]
    x_test=x[test_index:]
    y_train=y[valid_index-398:valid_index]
    y_valid=y[valid_index:test_index]
    y_test=y[test_index:]

    trainset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    valset = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))
    testset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
    trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train)
    valset_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testset_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainset_loader,valset_loader,testset_loader


def load_csi_by_year(dataset,batch_size,shuffle_train=True,n_jobs=0,
    valid_year=2018,
    test_year=2019,
    window=16,
    steps=1,
):

    data = np.load("/home/hzf/CSI/fq_masked_adj_close.npy").astype(np.float32)

    YEAR_IDX = 0
    PRICE_IDX = 3
    MASK_IDX = 4

    stock_num, total_days, feat_dim = data.shape
    assert feat_dim == 5, f"期望 feature_dim=5，实际为 {feat_dim}"

    years = data[0, :, YEAR_IDX].astype(int)
    price_data = data[:, :, PRICE_IDX]   # [N, T]
    mask_data = data[:, :, MASK_IDX]     # [N, T]

    x = []
    y = []
    sample_years = []

    total_samples = total_days - window - steps + 1

    for idx in range(total_samples):
        x_sample = price_data[:, idx:idx + window]   # [N, window]
        x_sample = np.expand_dims(x_sample, axis=-1)
        last_input_day = idx + window - 1
        target_day = idx + window + steps - 1

        # 输入窗口到目标日都有效，才认为这个样本该股票有效
        mask_sample = mask_data[:, idx:idx + window + steps]   # [N, window+steps]
        final_mask = np.min(mask_sample, axis=1, keepdims=True).astype(np.float32)  # [N, 1]

        # historical window 最后一天价格，作为 base price
        base_price = price_data[:, last_input_day]   # [N]

        # 目标日真实价格
        future_price = price_data[:, target_day]     # [N]

        # 真实 1-day return ratio
        # steps=1 时即 (P_{t+1} - P_t) / P_t
        gt_ratio = (future_price - base_price) / (base_price + 1e-12)   # [N]

        gt_sample = np.expand_dims(gt_ratio, axis=1)      # [N, 1]
        price_sample = np.expand_dims(base_price, axis=1) # [N, 1]

        y_sample = np.concatenate((gt_sample, price_sample, final_mask), axis=1)  # [N, 3]

        x.append(x_sample)
        y.append(y_sample)
        sample_years.append(years[target_day])

    x = np.array(x, dtype=np.float32)   # [num_samples, N, window, 5]
    y = np.array(y, dtype=np.float32)   # [num_samples, N, 3]
    sample_years = np.array(sample_years)

    train_mask = sample_years < valid_year
    val_mask = sample_years == valid_year
    test_mask = sample_years == test_year

    x_train, y_train = x[train_mask], y[train_mask]
    x_valid, y_valid = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    print(f"data shape: {data.shape}")
    print(f"total samples: {len(x)}")
    print(f"train samples (< {valid_year}): {len(x_train)}")
    print(f"valid samples (= {valid_year}): {len(x_valid)}")
    print(f"test samples (= {test_year}): {len(x_test)}")

    trainset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    valset = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))
    testset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=n_jobs
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs
    )

    return train_loader,val_loader,test_loader



def load_csi300_by_year(dataset,batch_size,shuffle_train=True,n_jobs=0,
    valid_year=2018,
    test_year=2019,
    window=16,
    steps=1,
):

    data = np.load("/home/hzf/CSI/fq300_masked_adj_close.npy").astype(np.float32)

    YEAR_IDX = 0
    PRICE_IDX = 3
    MASK_IDX = 4

    stock_num, total_days, feat_dim = data.shape
    assert feat_dim == 5, f"期望 feature_dim=5，实际为 {feat_dim}"

    years = data[0, :, YEAR_IDX].astype(int)
    price_data = data[:, :, PRICE_IDX]   # [N, T]
    mask_data = data[:, :, MASK_IDX]     # [N, T]

    x = []
    y = []
    sample_years = []

    total_samples = total_days - window - steps + 1

    for idx in range(total_samples):
        x_sample = price_data[:, idx:idx + window]   # [N, window]
        x_sample = np.expand_dims(x_sample, axis=-1)
        last_input_day = idx + window - 1
        target_day = idx + window + steps - 1

        # 输入窗口到目标日都有效，才认为这个样本该股票有效
        mask_sample = mask_data[:, idx:idx + window + steps]   # [N, window+steps]
        final_mask = np.min(mask_sample, axis=1, keepdims=True).astype(np.float32)  # [N, 1]

        # historical window 最后一天价格，作为 base price
        base_price = price_data[:, last_input_day]   # [N]

        # 目标日真实价格
        future_price = price_data[:, target_day]     # [N]

        # 真实 1-day return ratio
        # steps=1 时即 (P_{t+1} - P_t) / P_t
        gt_ratio = (future_price - base_price) / (base_price + 1e-12)   # [N]

        gt_sample = np.expand_dims(gt_ratio, axis=1)      # [N, 1]
        price_sample = np.expand_dims(base_price, axis=1) # [N, 1]

        y_sample = np.concatenate((gt_sample, price_sample, final_mask), axis=1)  # [N, 3]

        x.append(x_sample)
        y.append(y_sample)
        sample_years.append(years[target_day])

    x = np.array(x, dtype=np.float32)   # [num_samples, N, window, 5]
    y = np.array(y, dtype=np.float32)   # [num_samples, N, 3]
    sample_years = np.array(sample_years)

    train_mask = sample_years < valid_year
    val_mask = sample_years == valid_year
    test_mask = sample_years == test_year

    x_train, y_train = x[train_mask], y[train_mask]
    x_valid, y_valid = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    print(f"data shape: {data.shape}")
    print(f"total samples: {len(x)}")
    print(f"train samples (< {valid_year}): {len(x_train)}")
    print(f"valid samples (= {valid_year}): {len(x_valid)}")
    print(f"test samples (= {test_year}): {len(x_test)}")

    trainset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    valset = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))
    testset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=n_jobs
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs
    )

    return train_loader,val_loader,test_loader


def load_csi2259_by_year(dataset,batch_size,shuffle_train=True,n_jobs=0,
    valid_year=2018,
    test_year=2019,
    window=16,
    steps=1,
):

    data = np.load("/home/hzf/CSI/fq2259_masked_adj_close.npy").astype(np.float32)

    YEAR_IDX = 0
    PRICE_IDX = 3
    MASK_IDX = 4

    stock_num, total_days, feat_dim = data.shape
    assert feat_dim == 5, f"期望 feature_dim=5，实际为 {feat_dim}"

    years = data[0, :, YEAR_IDX].astype(int)
    price_data = data[:, :, PRICE_IDX]   # [N, T]
    mask_data = data[:, :, MASK_IDX]     # [N, T]

    x = []
    y = []
    sample_years = []

    total_samples = total_days - window - steps + 1

    for idx in range(total_samples):
        x_sample = price_data[:, idx:idx + window]   # [N, window]
        x_sample = np.expand_dims(x_sample, axis=-1)
        last_input_day = idx + window - 1
        target_day = idx + window + steps - 1

        # 输入窗口到目标日都有效，才认为这个样本该股票有效
        mask_sample = mask_data[:, idx:idx + window + steps]   # [N, window+steps]
        final_mask = np.min(mask_sample, axis=1, keepdims=True).astype(np.float32)  # [N, 1]

        # historical window 最后一天价格，作为 base price
        base_price = price_data[:, last_input_day]   # [N]

        # 目标日真实价格
        future_price = price_data[:, target_day]     # [N]

        # 真实 1-day return ratio
        # steps=1 时即 (P_{t+1} - P_t) / P_t
        gt_ratio = (future_price - base_price) / (base_price + 1e-12)   # [N]

        gt_sample = np.expand_dims(gt_ratio, axis=1)      # [N, 1]
        price_sample = np.expand_dims(base_price, axis=1) # [N, 1]

        y_sample = np.concatenate((gt_sample, price_sample, final_mask), axis=1)  # [N, 3]

        x.append(x_sample)
        y.append(y_sample)
        sample_years.append(years[target_day])

    x = np.array(x, dtype=np.float32)   # [num_samples, N, window, 5]
    y = np.array(y, dtype=np.float32)   # [num_samples, N, 3]
    sample_years = np.array(sample_years)

    train_mask = sample_years < valid_year
    val_mask = sample_years == valid_year
    test_mask = sample_years == test_year

    x_train, y_train = x[train_mask], y[train_mask]
    x_valid, y_valid = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    print(f"data shape: {data.shape}")
    print(f"total samples: {len(x)}")
    print(f"train samples (< {valid_year}): {len(x_train)}")
    print(f"valid samples (= {valid_year}): {len(x_valid)}")
    print(f"test samples (= {test_year}): {len(x_test)}")

    trainset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    valset = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))
    testset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=n_jobs
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs
    )

    return train_loader,val_loader,test_loader


def load_csi(
    dataset,
    batch_size,
    shuffle_train=True,
    n_jobs=0,
    train_years=None,
    valid_year=2017,
    test_year=2018,
    window=16,
    steps=1,
):
    dataset = dataset.upper()

    data_path_map = {
        "CSI": "/home/hzf/CSI/fq_masked_adj_close.npy",
        "CSI300": "/home/hzf/CSI/fq300_masked_adj_close.npy",
        "CSI2259": "/home/hzf/CSI/fq2259_masked_adj_close.npy",
    }

    # if dataset not in data_path_map:
    #     raise ValueError(
    #         f"Unsupported dataset: {dataset}. "
    #         f"Available datasets: {list(data_path_map.keys())}"
    #     )

    # if train_years is None:
    #     raise ValueError(
    #         "train_years cannot be None. "
    #         "Example: train_years=[2014, 2015, 2016] or train_years=[2017]"
    #     )

    train_years = list(train_years)
    if len(train_years) == 0:
        raise ValueError("train_years cannot be empty.")
    data = np.load(data_path_map[dataset]).astype(np.float32)

    YEAR_IDX = 0
    PRICE_IDX = 3
    MASK_IDX = 4

    stock_num, total_days, feat_dim = data.shape
    assert feat_dim == 5, f"期望 feature_dim=5，实际为 {feat_dim}"

    years = data[0, :, YEAR_IDX].astype(int)
    price_data = data[:, :, PRICE_IDX]   # [N, T]
    mask_data = data[:, :, MASK_IDX]     # [N, T]

    x = []
    y = []
    sample_years = []

    total_samples = total_days - window - steps + 1

    for idx in range(total_samples):
        x_sample = price_data[:, idx:idx + window]   # [N, window]
        x_sample = np.expand_dims(x_sample, axis=-1) # [N, window, 1]

        last_input_day = idx + window - 1
        target_day = idx + window + steps - 1

        # 输入窗口到目标日都有效，才认为这个样本该股票有效
        mask_sample = mask_data[:, idx:idx + window + steps]   # [N, window+steps]
        final_mask = np.min(mask_sample, axis=1, keepdims=True).astype(np.float32)  # [N, 1]

        # historical window 最后一天价格，作为 base price
        base_price = price_data[:, last_input_day]   # [N]

        # 目标日真实价格
        future_price = price_data[:, target_day]     # [N]

        # 真实 return ratio
        gt_ratio = (future_price - base_price) / (base_price + 1e-12)   # [N]

        gt_sample = np.expand_dims(gt_ratio, axis=1)      # [N, 1]
        price_sample = np.expand_dims(base_price, axis=1) # [N, 1]

        # y: [gt_ratio, base_price, final_mask]
        y_sample = np.concatenate((gt_sample, price_sample, final_mask), axis=1)  # [N, 3]

        x.append(x_sample)
        y.append(y_sample)
        sample_years.append(years[target_day])

    x = np.array(x, dtype=np.float32)   # [num_samples, N, window, 1]
    y = np.array(y, dtype=np.float32)   # [num_samples, N, 3]
    sample_years = np.array(sample_years, dtype=np.int32)

    train_mask = np.isin(sample_years, np.array(train_years, dtype=np.int32))
    val_mask = sample_years == int(valid_year)
    test_mask = sample_years == int(test_year)

    x_train, y_train = x[train_mask], y[train_mask]
    x_valid, y_valid = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    print(f"dataset: {dataset}")
    print(f"data path: {data_path_map[dataset]}")
    print(f"data shape: {data.shape}")
    print(f"total samples: {len(x)}")
    print(f"train years: {train_years}, train samples: {len(x_train)}")
    print(f"valid year: {valid_year}, valid samples: {len(x_valid)}")
    print(f"test year: {test_year}, test samples: {len(x_test)}")

    trainset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    valset = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))
    testset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=n_jobs
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs
    )

    return train_loader, val_loader, test_loader