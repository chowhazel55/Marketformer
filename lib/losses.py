import torch
import torch.nn.functional as F
import math
import numpy as np

def loss_select(name):
    if name in ("mse","masked_mse"):
        return MaskedMSELoss
    elif name in ("reg_rank"):
        return RegressionRankLoss #alpha=0.1
    elif name in ("mae", "masked_mae"):
        return MaskedMAELoss
    elif name in ("mape","masked_mape"):
        return MAPELoss
    elif name in ("huber","huber_loss"):
        return HuberLoss #delta=0.5
    elif name in ("mse_trend","trend"):
        return MSETrendLoss #alpha=0.1
    elif name in ("quantile", "quantile_loss"):
        return QuantileLoss #gamma=0.7
    elif name in ("logcosh","los_cosh"):
        return LogCoshLoss
    elif name in ("pearson_mse"):
        return PearsonMSELoss
    else:
        raise NotImplementedError
    
############# Mean Squared Error ###############
def masked_mse_loss(pred,label,null_val=0.0):
    mask = ~torch.isnan(label)
    loss=(pred[mask]-label[mask])**2
    return torch.mean(loss)

def masked_mse_loss_nasdaq(pred, label, null_val):
    mask=label[:,2]
    base=label[:,1]
    ground_truth=label[:,0]
    return_ratio = torch.div(torch.sub(pred, base), base)
    loss=(return_ratio*mask-ground_truth*mask)**2
    return torch.mean(loss)

class MaskedMSELoss:
    def _get_name(self):
        return self.__class__.__name__
    
    def __call__(self, pred,label,null_val=0.0):
        if label.shape[-1]==3:
            return masked_mse_loss_nasdaq(pred, label, null_val)
        else:
            return masked_mse_loss(pred, label, null_val)
    

############# Mean Squared Error + Return Ratio Rank Error ###############
def reg_rank_loss(pred, label, alpha, null_val=-1234):
    device=pred.device
    batch_size=pred.shape[0]
    all_one=torch.ones(batch_size,1,dtype=torch.float32).to(device)
    mask=label[:,2].unsqueeze(dim=1)
    base=label[:,1].unsqueeze(dim=1)
    ground_truth=label[:,0].unsqueeze(dim=1)
    pred=pred.unsqueeze(dim=1)
    return_ratio = torch.div(torch.sub(pred, base), base)

    # regrassion loss
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask) 
    
    # rank loss 
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
    loss = reg_loss + alpha * rank_loss
    return loss

def reg_rank_loss_csi(pred, label, alpha, null_val):
    label=label.unsqueeze(dim=1)
    mask = ~torch.isnan(label)
    device=pred.device
    batch_size=pred.shape[0]
    all_one=torch.ones(batch_size,1,dtype=torch.float32).to(device)
    reg_loss = F.mse_loss(pred * mask, label * mask) 
    pre_pw_dif = torch.sub(
        pred @ all_one.t(),
        all_one @ pred.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ label.t(), 
        label @ all_one.t()
    )
    mask=torch.tensor([1.00 if value else 0.0 for value in mask])
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
    loss = reg_loss + alpha * rank_loss
    return loss

class RegressionRankLoss:
    def __init__(self,alpha):
        self.alpha=alpha
    def _get_name(self):
        return self.__class__.__name__
    def __call__(self, pred, label, null_val=None):
        if label.shape[-1]==3:
            return reg_rank_loss(pred,label,self.alpha,null_val)
        else:
            return reg_rank_loss_csi(pred, label, self.alpha, null_val)


############# Mean Average Error ###############
def masked_mae_loss(pred,label,null_val=0.0):
    mask = ~torch.isnan(label)
    loss=torch.abs(pred[mask]-label[mask])
    return torch.mean(loss)

def masked_mae_loss_nasdaq(pred,label,null_val=0.0):
    mask=label[:,2]
    base=label[:,1]
    ground_truth=label[:,0]
    return_ratio = torch.div(torch.sub(pred, base), base)

    # regrassion loss
    loss=torch.abs(return_ratio* mask-ground_truth* mask)
    return torch.mean(loss)

class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__
    
    def __call__(self, pred,label,null_val=0.0):
        if label.shape[-1]==3:
            return masked_mae_loss_nasdaq(pred, label, null_val)
        else:
            return masked_mae_loss(pred, label, null_val)
        

############# Mean Average Error ###############
def masked_mape_loss(pred,label,null_val=0.0):
    mask = ~torch.isnan(label)
    loss=torch.abs(pred[mask]-label[mask])/label[mask]
    loss=loss[~torch.isnan(loss)]
    loss=loss[~torch.isinf(loss)]
    return torch.mean(loss)

def masked_mape_loss_nasdaq(pred,label,null_val=0.0):
    mask=label[:,2]
    base=label[:,1]
    ground_truth=label[:,0]
    return_ratio = torch.div(torch.sub(pred, base), base)

    # regrassion loss
    loss=torch.abs(return_ratio* mask-ground_truth* mask)/(ground_truth*mask)
    loss=loss[~torch.isnan(loss)]
    loss=loss[~torch.isinf(loss)]
    return torch.mean(loss)

class MAPELoss:
    def _get_name(self):
        return self.__class__.__name__
    
    def __call__(self, pred,label,null_val=0.0):
        if label.shape[-1]==3:
            return masked_mape_loss_nasdaq(pred, label, null_val)
        else:
            return masked_mape_loss(pred, label, null_val)
        
        
def masked_huber(pred, label,delta, null_val=0.0):
    mask = ~torch.isnan(label)
    residual=torch.abs(pred[mask]-label[mask])
    mask2=residual < delta
    loss=torch.where(mask2,0.5*(residual**2), delta*residual-0.5*(delta**2))
    return torch.mean(loss)

def maseked_huber_nasdaq(pred, label, delta, null_val=0.0):
    mask=label[:,2]
    base=label[:,1]
    ground_truth=label[:,0]
    return_ratio = torch.div(torch.sub(pred, base), base)

    residual=torch.abs(return_ratio* mask-ground_truth* mask)
    mask2=residual < delta
    loss=torch.where(mask2,0.5*(residual**2), delta*residual-0.5*(delta**2))
    return torch.mean(loss)
        
class HuberLoss:
    def __init__(self,delta=1):
        self.delta=delta
    def _get_name(self):
        return self.__class__.__name__
    
    def __call__(self, pred,label, null_val=0.0):
        if label.shape[-1]==3:
            return maseked_huber_nasdaq(pred, label,self.delta, null_val)
        else:
            return masked_huber(pred, label,self.delta, null_val)


def reg_trend_loss(pred, label, alpha, null_val=0.0):
    mask = ~torch.isnan(label)
    mse_loss=(pred[mask]-label[mask])**2
    trend_diff=(pred[mask]*label[mask])
    trend_mask=torch.where(trend_diff<0,1.0,0.0)
    return torch.mean(mse_loss)+alpha*torch.mean(trend_mask)

def reg_trend_loss_nasdaq(pred, label, alpha, null_val=0.0):
    mask=label[:,2].unsqueeze(dim=1)
    base=label[:,1].unsqueeze(dim=1)
    ground_truth=label[:,0].unsqueeze(dim=1)
    return_ratio = torch.div(torch.sub(pred, base), base)
    
    mse_loss=(return_ratio*mask-ground_truth*mask)**2
    trend_diff=(return_ratio * ground_truth)
    trend_mask=torch.where(trend_diff<0,0.1,0.0)
    return torch.mean(mse_loss)+alpha*torch.mean(trend_mask*mask)

class MSETrendLoss:
    def __init__(self, alpha):
        self.alpha=alpha
    def _get_name(self):
        return self.__class__.__name__
    def __call__(self, pred, label, null_val=0.0):
        if label.shape[-1]==3:
            return reg_trend_loss_nasdaq(pred,label, self.alpha, null_val)
        else:
            return reg_trend_loss(pred, label, self.alpha, null_val)
        

def quantile_loss(pred, label, gamma, null_val):
    mask=~torch.isnan(label)
    diff=pred[mask]-label[mask]
    mask2=torch.where(diff>0, (1-gamma),gamma)
    loss=torch.abs(diff)*mask2
    return torch.mean(loss)

def quantile_loss_nasdaq(pred, label, gamma, null_val):
    mask=label[:,2]
    base=label[:,1]
    ground_truth=label[:,0]
    return_ratio = torch.div(torch.sub(pred, base), base)
    diff=return_ratio*mask-ground_truth*mask
    mask2=torch.where(diff>0, (1-gamma),gamma)
    loss=torch.abs(diff)*mask2
    return torch.mean(loss)

class QuantileLoss:
    def __init__(self, gamma):
        self.gamma=gamma
    def _get_name(self):
        return self.__class__.__name__
    def __call__(self, pred, label, null_val=0.0):
        if label.shape[-1]==3:
            return quantile_loss_nasdaq(pred, label, self.gamma, null_val)
        else:
            return quantile_loss(pred, label, self.gamma, null_val)
        

def log_cosh_loss(pred,label,null_val=0.0):
    mask = ~torch.isnan(label)
    loss=torch.log(torch.cosh(pred[mask]-label[mask]))
    return torch.sum(loss)
def log_cosh_loss2(pred,label,null_val=0.0):
    mask = ~torch.isnan(label)
    diff=pred[mask]-label[mask]
    loss=diff+torch.nn.functional.softplus(-2.*diff)-math.log(2.0)
    return torch.mean(loss)

def log_cosh_loss_nasdaq(pred, label, null_val):
    mask=label[:,2]
    base=label[:,1]
    ground_truth=label[:,0]
    return_ratio = torch.div(torch.sub(pred, base), base)
    loss=torch.log(torch.cosh((return_ratio*mask-ground_truth*mask)))
    return torch.sum(loss)

def log_cosh_loss_nasdaq2(pred, label, null_val):
    mask=label[:,2]
    base=label[:,1]
    ground_truth=label[:,0]
    return_ratio = torch.div(torch.sub(pred, base), base)
    diff=return_ratio*mask-ground_truth*mask
    loss=diff+torch.nn.functional.softplus(-2.*diff)-math.log(2.0)
    return torch.mean(loss)

    
class LogCoshLoss:
    def _get_name(self):
        return self.__class__.__name__
    
    def __call__(self, pred,label,null_val=0.0):
        if label.shape[-1]==3:
            return log_cosh_loss_nasdaq2(pred, label, null_val)
        else:
            return log_cosh_loss2(pred, label, null_val)

def pearson_mse_loss_nasdaq(pred, label, alpha, null_val=-1234):
    mask=label[:,2]
    base=label[:,1]
    ground_truth=label[:,0]
    return_ratio=torch.div(torch.sub(pred,base),base)
    ground_truth=ground_truth*mask
    return_ratio=return_ratio*mask
    loss=(return_ratio-ground_truth)**2
    vx=return_ratio-torch.mean(return_ratio)
    vy=ground_truth-torch.mean(ground_truth)
    corr=1-torch.sum(vx*vy)/(torch.sqrt(torch.sum(vx**2))*torch.sqrt(torch.sum(vy**2))+1e-8)
    pearson_mse=loss.mean()+alpha*corr
    return pearson_mse

def pearson_mse_loss_csi(pred,label,alpha,null_val=0.0):
    mask = ~torch.isnan(label)
    loss=(pred[mask]-label[mask])**2
    vx=pred[mask]-torch.mean(pred[mask])
    vy=label[mask]-torch.mean(label[mask])
    corr=1-torch.sum(vx*vy)/(torch.sqrt(torch.sum(vx**2))*torch.sqrt(torch.sum(vy**2))+1e-8)
    pearson_mse=loss.mean()+alpha*corr
    return pearson_mse

class PearsonMSELoss:
    def __init__(self,alpha):
        self.alpha=alpha
    def _get_name(self):
        return self.__class__.__name__
    def __call__(self,pred,label,null_val=0.0):
        if label.shape[-1]==3:
            return pearson_mse_loss_nasdaq(pred,label,self.alpha,null_val)
        else:
            return pearson_mse_loss_csi(pred,label,self.alpha,null_val)
        
# def masked_mse_loss(pred,label,null_val=0.0):
#     mask = ~torch.isnan(label)
#     loss=(pred[mask]-label[mask])**2
#     return torch.mean(loss)

# def masked_mse_loss_nasdaq(pred, label, null_val):
#     mask=label[:,2]
#     base=label[:,1]
#     ground_truth=label[:,0]
#     return_ratio = torch.div(torch.sub(pred, base), base)
#     loss=(return_ratio*mask-ground_truth*mask)**2
#     return torch.mean(loss)