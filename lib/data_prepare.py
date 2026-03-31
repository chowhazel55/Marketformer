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
    elif dataset in ('SP500'):
        return load_sp
    elif dataset in ('STOCK'):
        return load_mv
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


   
def load_sp(dataset, batch_size, shuffle_train, n_jobs):
    dataset=dataset.upper()
    data = np.load('/home/users/hzf/benchmarks/SP500/moving_avg.npy')
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
    x=[]
    y=[]
    valid_index=1006-window+steps-1
    test_index=1259-window+steps-1
    for idx in range(1611-window-steps+1):
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


def load_mv(dataset, batch_size=1, shuffle_train=True, n_jobs=0):

    dataset = dataset.upper()

    with open(f'../data/{dataset}/eod_data.pkl','rb') as f:
        eod_data = pickle.load(f).astype(np.float32)

    with open(f'../data/{dataset}/gt_data.pkl','rb') as f:
        gt_data = pickle.load(f).astype(np.float32)

    with open(f'../data/{dataset}/price_data.pkl','rb') as f:
        price_data = pickle.load(f).astype(np.float32)

    with open(f'../data/{dataset}/mask_data.pkl','rb') as f:
        mask_data = pickle.load(f)

    with open(f'../data/{dataset}/meta_info.pkl','rb') as f:
        meta_info = pickle.load(f)

    train_idx = meta_info["train_idx"]
    valid_idx = meta_info["valid_idx"]
    test_idx = meta_info["test_idx"]

    window = 16
    steps = 1

    N, T, F = eod_data.shape

    # stock_max_mv = np.max(eod_data[:, :, -1], axis=1, keepdims=True)  # [N, 1]
    # stock_max_mv[stock_max_mv == 0] = 1.0
    # eod_data = eod_data / stock_max_mv[:, :, None]
    # price_data = price_data / stock_max_mv
    x = []
    y = []

    for idx in range(T - window - steps + 1):

        # 16-day features
        x.append(eod_data[:, idx:idx+window, :])
        # mask
        mask_sample = mask_data[:, idx:idx+window+1]
        # base price
        price_sample = np.expand_dims(price_data[:, idx+window-1], axis=1)
        # next day return
        gt_sample = np.expand_dims(gt_data[:, idx+window+steps-1], axis=1)
        mask_sample = np.min(mask_sample, axis=1)
        final_mask = np.expand_dims(mask_sample, axis=1)
        y.append(np.concatenate((gt_sample, price_sample, final_mask), axis=1))

    x = np.array(x)
    y = np.array(y)

    # ---------- split ----------
    valid_index = valid_idx[0] - window + steps
    test_index = test_idx[0] - window + steps

    x_train = x[:valid_index]
    x_valid = x[valid_index:test_index]
    x_test = x[test_index:]

    y_train = y[:valid_index]
    y_valid = y[valid_index:test_index]
    y_test = y[test_index:]

    print("Train shape:", x_train.shape)
    print("Valid shape:", x_valid.shape)
    print("Test shape:", x_test.shape)

    trainset = TensorDataset(
        torch.FloatTensor(x_train),
        torch.FloatTensor(y_train)
    )

    valset = TensorDataset(
        torch.FloatTensor(x_valid),
        torch.FloatTensor(y_valid)
    )

    testset = TensorDataset(
        torch.FloatTensor(x_test),
        torch.FloatTensor(y_test)
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=n_jobs
    )

    valid_loader = DataLoader(
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

    return train_loader, valid_loader, test_loader