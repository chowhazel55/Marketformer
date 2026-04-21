from .AbstractRunner import AbstractRunner

import torch
import numpy as np
import sys
import os

sys.path.append("..")
from lib.utils import print_log
from lib.metrics import IC_RIC,PrecN,MAE_MAPE_RMSE_ACC
import torch.nn.functional as F
import torch

def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask) # mask用于去掉缺失值的干扰
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio

class CSIRunner(AbstractRunner):
    def __init__(self, cfg:dict, device, log=None):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.log = log
        self.clip_grad = cfg.get("clip_grad")
        self.alpha = cfg.get("alpha")

    def train_one_epoch(self,model,trainset_loader,optimizer,scheduler, criterion):
        model.train()
        losses=[]
        for x,y in trainset_loader:
            feature = torch.squeeze(x, dim=0).to(self.device)
            label = torch.squeeze(y, dim=0).to(self.device)     
            pred = model(feature.float()) 
            loss = criterion(pred, label)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_grad)
            optimizer.step()
            #scheduler.step()
        epoch_loss = float(np.mean(losses))
        scheduler.step()
        return epoch_loss
    
    @torch.no_grad()
    def eval_model(self,model, dataset_loader, criterion):
        model.eval()
        losses=[]
        for x,y in dataset_loader:
            feature = torch.squeeze(x, dim=0).to(self.device)
            label = torch.squeeze(y, dim=0).to(self.device)
            pred=model(feature.float())
            loss=criterion(pred,label)
            losses.append(loss.item())
        losses=float(np.mean(losses))
        return losses
    
    @torch.no_grad()
    def predict(self, model, testset_loader):
        model.eval()
        ic=[]
        ric=[]
        prec10=[]
        prec30=[]
        mae = []
        mape = []
        rmse = []
        acc = []
        
        for x,y in testset_loader:
            feature = torch.squeeze(x, dim=0).to(self.device)
            label = torch.squeeze(y, dim=0)
            pred=model(feature.float()).detach().cpu()
            
            base=label[:,1]
            ground_truth=label[:,0]
            mask=label[:,2].bool()

            valid_mask = mask & (base != 0)
            valid_base = base[valid_mask]
            valid_gt = ground_truth[valid_mask]
            valid_pred_price = pred[valid_mask]
            valid_pred_ratio = (valid_pred_price - valid_base) / valid_base

            daily_ic, daily_ric = IC_RIC(valid_gt.numpy(), valid_pred_ratio.numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)
            daily_prec10 = PrecN(valid_gt.numpy(), valid_pred_ratio.numpy(), 10)
            daily_prec30 = PrecN(valid_gt.numpy(), valid_pred_ratio.numpy(), 30)
            prec10.append(daily_prec10)
            prec30.append(daily_prec30)
            daily_mae, daily_mape,daily_rmse, daily_acc=MAE_MAPE_RMSE_ACC(valid_gt.numpy(), valid_pred_ratio.numpy())
            mae.append(daily_mae)
            mape.append(daily_mape)
            rmse.append(daily_rmse)
            acc.append(daily_acc)

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric),
            'Prec@10': np.mean(prec10),
            'Prec@30': np.mean(prec30),
            'MAE': np.mean(mae),
            'MAPE': np.mean(mape),
            'RMSE': np.mean(rmse),
            'ACC': np.mean(acc),

        }
        return metrics



