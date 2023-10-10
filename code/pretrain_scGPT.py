"""
Author: QI LIU, Data Science Group of GHDDI, qi.liu@ghddi.org
Description: Pretraining a scGPT on single-cell datasets.
"""
import gc
import copy
import json
import os
import sys
import time
import traceback
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import torch
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
from anndata import AnnData
from scipy.sparse import issparse
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

sys.path.insert(0, "../")
sys.path.append('/home/qiliu02/GHDDI/DS-group/ghddixcre_singlecell_gpt/contribs/scGPT/')
import scgpt as scg
from scgpt.utils import set_seed
from scgpt.utils import category_str2int, eval_scib_metrics
from scgpt import SubsetsBatchSampler
from scgpt.preprocess import Preprocessor
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer.gene_tokenizer import GeneVocab

# from config import hyperparameter_defaults
sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")


def _parse_args():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        type=str,
        default="Pretrain_scGPT"
    )
    parser.add_argument(
        "-d",
        "--data_source",
        type=str,
        # required=True,
        default="../data/binned/",
        help='The name of the data source (currently support "scvi" datasets), or the '
        "path to the data file.",
    )

    # cellxgene_census_human-May23-08-36-2023
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        # required=True,
        default="../save/",
        help="The directory to save the trained model and the results.",
    )

    parser.add_argument(
        "-m",
        "--load_model",
        type=str,
        default=None
    )

    # settings for data
    parser.add_argument(
        "--n_hvg",
        type=int,
        default=None,
        help="The number of highly variable genes. If set to 0, will use all genes. "
        "Default is 0, which will determine the n_hvg automatically.",
    )

    parser.add_argument(
        "--valid_size_or_ratio",
        default=0.3
    )

    parser.add_argument(
        "--dist_backend",
        default="nccl"
    )    
    
    parser.add_argument(
        "--grad_accu_steps",
        type=str,
        default=1
    )
    
    # settings for tokenizer
    parser.add_argument(
        "--pad_token",
        type=str,
        default="<pad>",
        help="The token to use for padding. Default is <pad>.",
    )
    
    parser.add_argument(
        "--input_style",
        type=str,
        default="binned"
    )

    parser.add_argument(
        "--input_emb_style",
        type=str,
        default="continuous"
    )
    # # settings for evaluation
    # parser.add_argument(
    #     "--cell-emb-mode",
    #     type=str,
    #     choices=["weighted sum", "non-weighted sum", "cls"],
    #     default="weighted sum",
    #     help="The mode to use for cell embeddings.",
    # )    

    parser.add_argument(
        "--n_bins",
        type=int,
        default=51
    )    

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1200,
        help="The maximum length of the sequence. Default is 1200. The actual used "
        "max length would be the minimum of this value and the length of the longest "
        "sequence in the data.",
    )

    parser.add_argument(
        "--training_tasks",
        type=str,
        default="both"
    )
    parser.add_argument(
        "--dist_url",
        default="tcp://gpu188.cluster.local:53833"
    )
    
    parser.add_argument(
        "--mask_ratio",
        nargs='+', 
        type=float,
        default=[0.25, 0.5, 0.75]
    )

    parser.add_argument(
        "--trunc_by_sample",
        default=True
    )

    parser.add_argument(
        "--vocab_path",
        type=str,
        default="./default_census_vocab.json"
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=0
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        # default=16
        default=16
    )
    
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        # default=64,
        help="The batch size for evaluation. Default is 64.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=6
    )    

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001
    )        
            
    # settings for logging
    parser.add_argument(
        "--scheduler_interval",
        type=int,
        default=100,
        help="The interval for logging. Default is 100.",
    )

    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--warmup_ratio_or_step",
        type=float,
        default=10000.0
    )

    parser.add_argument(
        "--no_cls",
        default=True
    )

    parser.add_argument(
        "--no_cce",
        default=True
    )

    parser.add_argument(
        "--fp16",
        default=True
    )

    parser.add_argument(
        "--fast_transformer",
        action='store_true'
    )

    parser.add_argument(
        "--nlayers",
        type=int,
        default=12
    )

    parser.add_argument(
        "--nheads",
        type=int,
        default=8
    )

    parser.add_argument(
        "--embsize",
        type=int,
        default=512
    )

    parser.add_argument(
        "--d_hid",
        type=int,
        default=512
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--n_layers_cls",
        type=int,
        default=3
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=9000
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        default=27000
    )

    parser.add_argument(
        "--mask_value",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--pad_value",
        type=int,
        default=-2,
        help="The value to use for padding null gene expression. Default is 0.",
    )

    parser.add_argument(
        "--USE_CLS",
        action='store_true'
    )

    parser.add_argument(
        "--USE_CCE",
        action='store_true'
    )

    parser.add_argument(
        "--MVC",
        action='store_true'
    )
    
    parser.add_argument(
        "--USE_GENERATIVE_TRAINING",
        action='store_true'
    )

    parser.add_argument(
        "--world_size",
        type=int,
        default=16
    )
    
    parser.add_argument(
        "--distributed",
        action='store_true'
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=0
    )

    parser.add_argument(
        "--gpu",
        type=int,
        nargs='+',
        default=0
    )

    # extra 
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )    

    parser.add_argument(
        "--amp",  # 
        action='store_true',
        default=False
    )        
    
    parser.add_argument(
        "--GEPC",
        action='store_true'
    )        

    parser.add_argument(
        "--ecs_thres",
        type=float,
        default=0.8
    )        

    parser.add_argument(
        "--explicit_zero_prob",
        action='store_true'
    )
    
    parser.add_argument(
        "--save_eval_interval",
        type=int,
        default=1
    )

    args = parser.parse_args()
    return args


def train(
    config,
    vocab,
    epoch,
    model,
    device,
    loader,
    criterion,
    optimizer,
    scaler,
    scheduler,
    logger,
    ) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = 100 # config.log_interval
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[config['pad_token']])
        with torch.cuda.amp.autocast(enabled=config['amp']):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                MVC=False,
                ECS=True,                
                # MVC=config.GEPC,
                # ECS=config.ecs_thres > 0,
            )

            masked_positions = input_values.eq(config['mask_value'])  # the postions to predict
            loss = loss_mse = criterion(
                output_dict["mlm_output"], 
                target_values, 
                masked_positions
            )
            metrics_to_log = {"train/mse": loss_mse.item()}
            
            if config['explicit_zero_prob']:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], 
                    target_values, 
                    masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                
        model.zero_grad()
        loss = loss.mean()
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        wandb.log(metrics_to_log)

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        total_loss += loss.item()
        total_mse += loss_mse.item()
        # total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_error += mre.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = 0.0
            # cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                # + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
            )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_error = 0
            start_time = time.time()


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(
    config,
    vocab,
    epoch,
    model: nn.Module, 
    device,
    loader: DataLoader,
    criterion,
    ) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    
    pad_token = config['pad_token']
    mask_value = config['mask_value']
    with torch.no_grad():
        for batch_data in loader:
            # input_gene_ids = torch.tensor(batch_data["gene_ids"]).to(device)
            # input_values = torch.tensor(batch_data["values"]).to(device)
            # target_values = torch.tensor(batch_data["target_values"]).to(device)
            # # batch_labels =  torch.tensor(batch_data["batch_labels"]).to(device)
            
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            # batch_labels = batch_data["batch_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config['amp']):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_values, target_values, masked_positions)
                # loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            # total_dab += loss_dab.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/mre": total_error / total_num,
            "epoch": epoch,
        },
    )
    return total_loss / total_num, total_error / total_num


def main(args):
    # logging
    project_name = args.project_name
    save_dir = Path(f"save/dev_{project_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
        
    # saving log
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir/"run.log")
        
    if args.load_model is not None:
        model_dir = Path(args.load_model)
        model_config_file = str(model_dir / "args.json")
        model_file = str(model_dir / "best_model.pt")
        vocab_file = str(model_dir / "vocab.json")
        vocab = GeneVocab.from_file(vocab_file)
        
        # model configs
        with open(model_config_file, "r") as f:
            config = json.load(f)
            
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        config['vocab_path'] = vocab_file
        config['load_model'] = args.load_model
        
    else:
        # config
        config = vars(args)
        # args_dict = vars(args)
        
    # extra_args = {
    #     'seed': 42,
    #     'amp': True,
    #     'GEPC': False,
    #     'ecs_thres': 0.8,
    #     'explicit_zero_prob': False
    # }
    # config = {**config, **extra_args}
    # print(config, type(config))    
    
    # saving the args to the args.json
    with open(save_dir / "args.json", "w") as f:
        json.dump(config, f, indent=4)
        # json.dump(vars(args), f, indent=4)

    # saving vocab
    pad_token = config['pad_token']  # "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    vocab_file = config['vocab_path']
    
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    ntokens = len(vocab)  # size of vocabulary, 60694       
    vocab.save_json(file_path=f"{save_dir}/vocab.json")     
    
    run = wandb.init(
        config=config,
        project=config['project_name'],
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    config = wandb.config
    scg.utils.set_seed(config.seed)
    set_seed(config.seed)
    print(config)
    print("="*100)
    logger.info(f"config: {config}")
    logger.info(f"save_dir: {save_dir}")
    logger.info("="*100)
    
    if torch.cuda.is_available():
        if isinstance(config.gpu, int):
            device = torch.device(f"cuda:{config.gpu}")
        elif (isinstance(config.gpu, list) and len(config.gpu) == 1):
            device = torch.device(f"cuda:{config.gpu[0]}")
        else:
            device = torch.device(f"cuda")
    else:
        device = torch.device(f"cpu")
    print(f"deivce: {device}")
    logger.info(f"deivce: {device}")
    
    model = TransformerModel(
        ntokens,
        d_model=config['embsize'],
        nhead=config['nheads'],
        d_hid=config['d_hid'],
        nlayers=config['nlayers'],
        vocab=vocab,
        dropout=config['dropout'],
        pad_token=config['pad_token'],
        pad_value=config['pad_value'],
        do_mvc=config['MVC'],
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=None,
        domain_spec_batchnorm=False,
        n_input_bins=config['n_bins'],
        ecs_threshold=0.8,
        explicit_zero_prob=config['explicit_zero_prob'],
        use_fast_transformer=config['fast_transformer'],
        pre_norm=False,
    )
    
    if config.load_model is not None:
        try:
            model_dir = Path(config.load_model)
            model_file = model_dir / "best_model.pt"
            model.load_state_dict(torch.load(model_file))
            logger.info(f"Loading all model params from {model_file}")
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    
    model.to(device)
    
    if isinstance(args.gpu, list) and len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
    print(f"model: {type(model)}")
    logger.info(f"model: {type(model)}")
    
    wandb.watch(model)

    lr = config['lr']
    eps = 1e-4 if config['amp'] else 1e-8
    criterion = masked_mse_loss
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        eps=eps
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        1, 
        gamma=0.99  # scheduler_factor?
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    
    best_val_loss = float("inf")
    best_model = None
    define_wandb_metrcis()

    data_source = config['data_source']
    cache_dir = Path(data_source) / "cache"
    cls_prefix_datatable = Path(data_source) / "databanks_binned.parquet"
    dataset = load_dataset(
        "parquet",
        data_files=str(cls_prefix_datatable),
        split="train",
        # cache_dir=str(cache_dir),
    )
    dataset = dataset.with_format("torch")

    epochs = config['epochs']
    batch_size = config['batch_size']
    eval_batch_size = config['eval_batch_size']
    valid_size_or_ratio = config['valid_size_or_ratio']
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        val_size = int(valid_size_or_ratio * len(dataset))  # 80% for training
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
        
        logger.info("===")
        logger.info("config.do_train")
        
        # training one epoch
        train(
            config=config,
            vocab=vocab,
            epoch=epoch,
            model=model,
            device=device,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            logger=logger,          
        )
        
        # evaluation of one epoch
        val_loss, val_mre = evaluate(
            config=config,
            vocab=vocab,
            epoch=epoch,
            model=model, 
            device=device,
            loader=valid_loader,
            criterion=criterion,
        )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")

        save_eval_interval = 100 if 'save_eval_interval' not in config else config['save_eval_interval']
        if epoch % save_eval_interval == 0 or epoch == epochs:
        # if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
            logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

        scheduler.step()

    # save the best model
    if isinstance(best_model, torch.nn.DataParallel):
        model_state_dict = best_model.module.state_dict()
    else:
        model_state_dict = best_model.state_dict()
    torch.save(model_state_dict, save_dir / "best_model.pt")
    # torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    
    artifact = wandb.Artifact(f"best_model", type="model")
    glob_str = os.path.join(save_dir, "best_model.pt")
    artifact.add_file(glob_str)
    run.log_artifact(artifact)
    run.finish()
    wandb.finish()
    gc.collect()    
    

if __name__=="__main__":
    if scg.utils.isnotebook():
        parser = argparse.ArgumentParser()
        model_name = "pbmc-Jun09-22-32-2022"
        args = parser.parse_args(
            args=[
                "-d",
                "pbmc_dataset",
                "-m",
                f"./save/{model_name}",
                "-s",
                f"./save/apply-{model_name}",
            ]
        )
    else:
        args = _parse_args()
       
    main(args)
    