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
from lib.data_prepare import load_csi
from lib.losses import loss_select
from runners import runner_select




def build_rolling_schedule():
    schedule = [
        {
            "name": "bootstrap_2014_2016",
            "train_years": [2014, 2015, 2016],
            "valid_year": 2017,
            "test_year": 2018,
        }
    ]

    for year in range(2017, 2024):
        schedule.append(
            {
                "name": f"roll_{year}",
                "train_years": [year],
                "valid_year": year + 1,
                "test_year": year + 2,
            }
        )

    return schedule


def train(
    model,
    runner,
    trainset_loader,
    valset_loader,
    testset_loader,
    optimizer,
    scheduler,
    criterion,
    max_epochs=100,
    early_stop=20,
    log=None,
    save=None,
):
    best_loss = np.inf
    best_loss_epoch = 0
    best_state_dict = None
    wait = 0

    for epoch in range(max_epochs):
        train_loss = runner.train_one_epoch(model, trainset_loader, optimizer, scheduler, criterion)
        val_loss = runner.eval_model(model, valset_loader, criterion)
        test_loss = runner.eval_model(model, testset_loader, criterion)

        train_metrics = test_model(model, runner, trainset_loader, log=log)
        valid_metrics = test_model(model, runner, valset_loader, log=log)
        test_metrics = test_model(model, runner, testset_loader, log=log)

        print_log(
            "Epoch %d, train_loss %.6f, train_ic %.6f, train_icir %.6f, train_ric %.6f, train_ricir %.6f, train_mae %.6f, train_rmse %.6f, train_mape %.6f, train_acc %.6f, "
            "valid_loss %.10f, valid_ic %.6f, valid_icir %.6f, valid_ric %.6f, valid_ricir %.6f, valid_prec@10 %.6f, valid_prec@30 %.6f, valid_mae %.6f, valid_rmse %.6f, valid_mape %.6f, valid_acc %.6f, "
            "test_loss %.6f, ic %.6f, icir %.6f, ric %.6f, ricir %.6f, prec@10 %.6f, prec@30 %.6f, test_mae %.6f, test_rmse %.6f, test_mape %.6f, test_acc %.6f"
            % (
                epoch + 1,
                train_loss,
                train_metrics['IC'], train_metrics['ICIR'],
                train_metrics['RIC'], train_metrics['RICIR'],
                train_metrics['MAE'], train_metrics['RMSE'],
                train_metrics['MAPE'], train_metrics['ACC'],

                val_loss,
                valid_metrics['IC'], valid_metrics['ICIR'],
                valid_metrics['RIC'], valid_metrics['RICIR'],
                valid_metrics['Prec@10'], valid_metrics['Prec@30'],
                valid_metrics['MAE'], valid_metrics['RMSE'],
                valid_metrics['MAPE'], valid_metrics['ACC'],

                test_loss,
                test_metrics['IC'], test_metrics['ICIR'],
                test_metrics['RIC'], test_metrics['RICIR'],
                test_metrics['Prec@10'], test_metrics['Prec@30'],
                test_metrics['MAE'], test_metrics['RMSE'],
                test_metrics['MAPE'], test_metrics['ACC'],
            ),
            log=log
        )

        if val_loss < best_loss:
            wait = 0
            best_loss = val_loss
            best_loss_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            if save is not None:
                torch.save(best_state_dict, save)
                print_log(f"Checkpoint updated: {save}", log=log)
        else:
            wait += 1
            if early_stop is not None and wait >= early_stop:
                print_log("Early stopping -----", log=log)
                break

    print_log("Best valid loss: %.10f" % best_loss, log=log)
    print_log("Best valid loss epoch: %d" % best_loss_epoch, log=log)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model


def test_model(model, runner, testset_loader, log=None):
    model.eval()
    metrics = runner.predict(model, testset_loader)
    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="CSI300")
    parser.add_argument("-m", "--model", type=str, default="MASTER")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("-s", "--seq_len", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("-c", "--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    set_cpu_num(args.cpus)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset.upper()
    model_name_input = args.model.upper()
    model_class = model_select(model_name_input)
    model_name = model_class.__name__
    learning_rate = args.learning_rate

    with open(f"../configs/{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]
    cfg["lookback_length"] = args.seq_len

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    log_path = f"../rollings_logs/{model_name}"
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(
        log_path,
        f"{model_name}-{dataset}-rolling-s{args.seq_len}-{now}.log"
    )
    log = open(log_file, "a")
    log.seek(0)
    log.truncate()

    save_root = f"../saved_models/{model_name}"
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(
        save_root,
        f"{model_name}-{dataset}-rolling-s{args.seq_len}-{now}"
    )
    os.makedirs(save_dir, exist_ok=True)

    print_log("Dataset: ", dataset, log=log)
    print_log("Random seed: %d" % args.seed, log=log)
    print_log("GPU: %d" % args.gpu_num, log=log)
    print_log("Learning rate: %s" % learning_rate, log=log)

    criterion = loss_select(cfg.get("loss", "mse"))(**cfg.get("loss_args", {}))
    runner = runner_select(cfg.get("runner", "basic"))(cfg, device=DEVICE, log=log)

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log("---------- %s ----------" % model_name, log=log)
    print_log(json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log)

    rolling_schedule = build_rolling_schedule()
    print_log(f"Rolling schedule: {rolling_schedule}", log=log)

    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 1)
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("data_args", {}).get("n_jobs", 0)
    shuffle_train = cfg.get("data_args", {}).get("shuffle_train", True)

    previous_checkpoint = args.init_checkpoint
    summary = []

    for round_id, item in enumerate(rolling_schedule, start=1):
        print_log("\n" + "=" * 100, log=log)
        print_log(
            f"Round {round_id}/{len(rolling_schedule)} | "
            f"name={item['name']} | "
            f"train_years={item['train_years']} | "
            f"valid_year={item['valid_year']} | "
            f"test_year={item['test_year']}",
            log=log,
        )

        trainset_loader, valset_loader, testset_loader = load_csi(
            dataset=dataset,
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            n_jobs=num_workers,
            train_years=item["train_years"],
            valid_year=item["valid_year"],
            test_year=item["test_year"],
            window=args.seq_len,
            steps=args.steps,
        )

        model = model_class(**cfg["model_args"]).to(DEVICE)

        if previous_checkpoint is not None:
            print_log(f"Loading checkpoint: {previous_checkpoint}", log=log)
            state_dict = torch.load(previous_checkpoint, map_location=DEVICE)
            model.load_state_dict(state_dict, strict=True)

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
        )

        checkpoint_path = os.path.join(
            save_dir,
            f"{model_name}-{dataset}-{item['name']}.pt"
        )

        model = train(
            model,
            runner,
            trainset_loader,
            valset_loader,
            testset_loader,
            optimizer,
            scheduler,
            criterion,
            max_epochs=cfg.get("epochs", 100),
            early_stop=None,
            log=log,
            save=checkpoint_path,
        )

        best_model = model_class(**cfg["model_args"]).to(DEVICE)
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        best_model.load_state_dict(state_dict, strict=True)

        valid_metrics = test_model(best_model, runner, valset_loader, log=log)
        test_metrics = test_model(best_model, runner, testset_loader, log=log)

        print_log("Model saved to: ", checkpoint_path, log=log)
        print_log("------- Validate -----", log=log)
        print_log(valid_metrics, log=log)
        print_log("------- Test -----", log=log)
        print_log(test_metrics, log=log)

        summary.append(
            {
                "round": round_id,
                "name": item["name"],
                "train_years": item["train_years"],
                "valid_year": item["valid_year"],
                "test_year": item["test_year"],
                "checkpoint": checkpoint_path,
                "valid_metrics": valid_metrics,
                "test_metrics": test_metrics,
            }
        )

        previous_checkpoint = checkpoint_path

    # summary_path = os.path.join(save_dir, "rolling_summary.json")
    # with open(summary_path, "w", encoding="utf-8") as f:
    #     json.dump(summary, f, ensure_ascii=False, indent=4)

    # print_log("\nRolling training finished.", log=log)
    # print_log(f"Summary saved to: {summary_path}", log=log)
    # print_log(f"Log saved to: {log_file}", log=log)

    log.close()