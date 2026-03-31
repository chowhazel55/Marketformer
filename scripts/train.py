import argparse
import numpy as np
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import datetime
import time
import matplotlib.pyplot as plt
import yaml
import json
import sys
import copy

sys.path.append("..")
from lib.utils import (
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from models import model_select
from lib.data_prepare import dataloaders_select
from lib.losses import loss_select
from runners import runner_select



def train(model, 
        runner, 
        trainset_loader,
        valset_loader,
        testset_loader,
        optimizer,
        scheduler,
        criterion,
        max_epochs=100,
        early_stop=20,
        verbose=1,
        log=None,
        save=None,
        savefig=None
):
    best_loss=np.inf
    best_loss_epoch=0

    train_loss_list=[]
    train_ic=[]
    train_icir=[]
    train_ric=[]
    train_ricir=[]

    val_loss_list=[]
    valid_ic=[]
    valid_icir=[]
    valid_ric=[]
    valid_ricir=[]

    test_loss_list=[]
    test_ic=[]
    test_icir=[]
    test_ric=[]
    test_ricir=[]

    epoch_num=[]
    wait=0
    early_stop_flag=0


    for epoch in range(max_epochs):
        train_loss=runner.train_one_epoch(model,trainset_loader,optimizer,scheduler, criterion)
        val_loss = runner.eval_model(model, valset_loader,criterion)
        test_loss = runner.eval_model(model, testset_loader,criterion,test=True)
        train_metrics=test_model(model, runner, trainset_loader, log=log)
        valid_metrics=test_model(model, runner, valset_loader, log=log)
        test_metrics=test_model(model, runner, testset_loader, log=log)

        train_loss_list.append(train_loss)
        train_ic.append(train_metrics['IC'])
        train_icir.append(train_metrics['ICIR'])
        train_ric.append(train_metrics['RIC'])
        train_ricir.append(train_metrics['RICIR'])

        val_loss_list.append(val_loss)
        valid_ic.append(valid_metrics['IC'])
        valid_icir.append(valid_metrics['ICIR'])
        valid_ric.append(valid_metrics['RIC'])
        valid_ricir.append(valid_metrics['RICIR'])

        test_loss_list.append(test_loss)
        test_ic.append(test_metrics['IC'])
        test_icir.append(test_metrics['ICIR'])
        test_ric.append(test_metrics['RIC'])
        test_ricir.append(test_metrics['RICIR'])

        epoch_num.append(epoch+1)

        print_log("Epoch %d, train_loss %.6f, train_ic %.6f, train_icir %.6f, train_ric %.6f, train_ricir %.6f, valid_loss %.10f , valid_ic %.6f, valid_icir %.6f, valid_ric %.6f, valid_ricir %.6f, test_loss %.6f, ic %.6f, icir %.6f, ric %.6f, ricir %.6f, prec@10 %.6f, prec@30 %.6f " % (epoch+1, train_loss, train_metrics['IC'],train_metrics['ICIR'],train_metrics['RIC'],train_metrics['RICIR'], val_loss, valid_metrics['IC'],valid_metrics['ICIR'],valid_metrics['RIC'],valid_metrics['RICIR'], test_loss, test_metrics['IC'],test_metrics['ICIR'],test_metrics['RIC'],test_metrics['RICIR'],test_metrics['Prec@10'], test_metrics['Prec@30']),log=log)
  
        if val_loss < best_loss:
            wait=0
            best_loss=val_loss
            best_loss_epoch=epoch+1
        else:
            wait+=1
            if wait >=early_stop and early_stop_flag==0:
                early_stop_flag=1
                print_log("Early stopping -----",log=log)

    print_log("Best valid loss: %d" % (best_loss),log=log)
    print_log("Best valid loss epoch: %d" % (best_loss_epoch),log=log)
    return model

def test_model(model,runner, testset_loader,log=None):
    model.eval()
    metrics= runner.predict(model, testset_loader)
    return metrics

if __name__=="__main__":

    # -------------------------- set running environment ------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="CSI300")
    parser.add_argument("-m", "--model", type=str, default="MASTER")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("-c", "--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=1)
    args = parser.parse_args()

    seed_everything(args.seed)
    set_cpu_num(args.cpus)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"
    model_name = args.model.upper()
    model_class = model_select(model_name)
    model_name = model_class.__name__
    learning_rate =args.learning_rate

    with open(f"../configs/{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model -------------------------------- #

    model = model_class(**cfg["model_args"]).to(DEVICE)

        
    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/{model_name}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    print_log("Dataset: ",dataset, log=log)
    print_log("Radom seed: %d" % (args.seed),log=log)
    print_log("GPU: %d" % (args.gpu_num),log=log)
    print_log("Learning rate:", learning_rate,log=log)
    # ------------------------------- load dataset ------------------------------- #

    trainset_loader,valset_loader,testset_loader = dataloaders_select(dataset,model_name)(**cfg.get("data_args", {}))
    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/{model_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}")
    if not os.path.exists(save):
        os.makedirs(save)
    savefig = os.path.join(log_path, f"{model_name}-{dataset}-{now}.png")


    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    criterion = loss_select(cfg.get("loss", "mse"))(**cfg.get("loss_args", {}))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.get("milestones", []),
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    # ----------------------------- set model runner ----------------------------- #

    runner = runner_select(cfg.get("runner", "basic"))(cfg, device=DEVICE, log=log)
    
    # --------------------------- train and test model --------------------------- #
    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    print_log("----------",model_name,"----------",log=log)
    print_log(json.dumps(cfg, ensure_ascii=False,indent=4, cls=CustomJSONEncoder), log=log)

    model=train(model, 
                runner, 
                trainset_loader,
                valset_loader,
                testset_loader,
                optimizer,
                scheduler,
                criterion,
                max_epochs=cfg.get("epochs", 100),
                early_stop=cfg.get("early_stop", 100),
                verbose=1,
                log=log,
                save=save,
                savefig=savefig
                )
    
    
    metrics=test_model(model, runner, testset_loader, log=log)
    print_log("Model saved to: ", save, log=log)
    print_log("------- Test -----",log=log)
    print_log(metrics,log=log)
    log.close()
