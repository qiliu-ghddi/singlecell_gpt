import gc
import os
import sys
import copy
import json
import time
import traceback
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import torch
import wandb
import scvi
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from anndata import AnnData
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt.data_sampler import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics


from .scgpt.config import hyperparameter_defaults
from .scgpt.config import run, config
from .scgpt.config import (
    per_seq_batch_sample, save_dir, pad_value, load_model,
    pad_token, special_tokens,
    explicit_zero_prob,
    DSBN,
    n_input_bins,
    max_seq_len,
    mask_value,
    mask_ratio
    )


###################################################################
# Dataset 
###################################################################
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
    

class PBMC_10K_Dataset:
    
    def __init__(
        self, 
        dataset_name
        ):
        self.dataset_name = dataset_name
        self.adata = self.load_dataset()
        pass

    def load_dataset(self):
        adata = scvi.data.pbmc_dataset()  # 11990 × 3346
        adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
        adata.var = adata.var.set_index("gene_symbols")
        data_is_raw = True
        
        # make the batch category column
        ori_batch_col = "batch"
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()
        return adata
    
    def get_adata(self):
        return self.adata
    
    def get_sorted_adata(self, adata):
        per_seq_batch_sample = True
        if per_seq_batch_sample:
            # sort the adata by batch_id in advance
            adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()
        return adata_sorted
        
    def load_tokenized_data(
        self,
        train_data,
        valid_data,
        gene_ids,
        max_seq_len,
        vocab,
        pad_token,
        pad_value,
        ):
        tokenized_train = tokenize_and_pad_batch(
            train_data,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=True,
        )
        tokenized_valid = tokenize_and_pad_batch(
            valid_data,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,
            include_zero_gene=True,
        )
        print(
            f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
        )
        print(
            f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
        )
        return tokenized_train, tokenized_valid


    def prepare_data(
        self,
        sort_seq_batch,
        tokenized_train,
        tokenized_valid,
        train_batch_labels,
        valid_batch_labels,
        ):
        masked_values_train = random_mask_value(
            tokenized_train["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        masked_values_valid = random_mask_value(
            tokenized_valid["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        print(
            f"random masking at epoch #, ratio of masked values in train: ",
            f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
        )

        input_gene_ids_train, input_gene_ids_valid = (
            tokenized_train["genes"],
            tokenized_valid["genes"],
        )
        input_values_train, input_values_valid = masked_values_train, masked_values_valid
        target_values_train, target_values_valid = (
            tokenized_train["values"],
            tokenized_valid["values"],
        )

        tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
        tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

        if sort_seq_batch:
            train_sort_ids = np.argsort(train_batch_labels)
            input_gene_ids_train = input_gene_ids_train[train_sort_ids]
            input_values_train = input_values_train[train_sort_ids]
            target_values_train = target_values_train[train_sort_ids]
            tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]

            valid_sort_ids = np.argsort(valid_batch_labels)
            input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
            input_values_valid = input_values_valid[valid_sort_ids]
            target_values_valid = target_values_valid[valid_sort_ids]
            tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]

        train_data_pt = {
            "gene_ids": input_gene_ids_train,
            "values": input_values_train,
            "target_values": target_values_train,
            "batch_labels": tensor_batch_labels_train,
        }
        valid_data_pt = {
            "gene_ids": input_gene_ids_valid,
            "values": input_values_valid,
            "target_values": target_values_valid,
            "batch_labels": tensor_batch_labels_valid,
        }

        return train_data_pt, valid_data_pt


###################################################################
# DataLoader 
###################################################################

class PBMC_10K_Dataloader:
    
    def __init__(self, config=None):
        pass
    

    def load_checkpoint(self, load_model):
        # settings for input and preprocessing
        pad_token = "<pad>"
        special_tokens = [pad_token, "<cls>", "<eoc>"]    
        
        model_dir = Path(load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        print(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        print(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
        

    def load_preprocessor(
        self,
        adata,
        dataset_name
    ):
        data_is_raw = True
        n_hvg = 1200
        n_bins = 51
        
        # set up the preprocessor, use the args to config the workflow
        preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=3,  # step 1
            filter_cell_by_counts=False,  # step 2
            normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=data_is_raw,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
            binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )
        return preprocessor(adata, batch_key="str_batch" if dataset_name != "heart_cell" else None)


    def tokenize_input(
        self, 
        adata
        ):
        input_layer_key = "X_binned"
        all_counts = (
            adata.layers[input_layer_key].A
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        genes = adata.var["gene_name"].tolist()

        celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
        num_types = len(set(celltypes_labels))
        celltypes_labels = np.array(celltypes_labels)

        batch_ids = adata.obs["batch_id"].tolist()
        num_batch_types = len(set(batch_ids))
        batch_ids = np.array(batch_ids)

        (
            train_data,
            valid_data,
            train_celltype_labels,
            valid_celltype_labels,
            train_batch_labels,
            valid_batch_labels,
        ) = train_test_split(
            all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
        )

        # settings for input and preprocessing
        pad_token = "<pad>"
        special_tokens = [pad_token, "<cls>", "<eoc>"]
        
        if config.load_model is None:
            vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
        vocab.set_default_index(vocab["<pad>"])
        gene_ids = np.array(vocab(genes), dtype=int)

        tockenized_data = {
            "num_batch_types": num_batch_types,
            "train_data": train_data,
            "valid_data": valid_data,
            "train_celltype_labels": train_celltype_labels,
            "valid_celltype_labels": valid_celltype_labels,
            "train_batch_labels": train_batch_labels,
            "valid_batch_labels": valid_batch_labels,
            
            "gene_ids": gene_ids,
            "num_types": num_types
        }
        
        return tockenized_data
        

    # data_loader
    def prepare_dataloader(
        data_pt: Dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool = False,
        intra_domain_shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        dataset = SeqDataset(data_pt)
        
        per_seq_batch_sample = True
        
        if per_seq_batch_sample:
            # find the indices of samples in each seq batch
            subsets = []
            batch_labels_array = data_pt["batch_labels"].numpy()
            for batch_label in np.unique(batch_labels_array):
                batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
                subsets.append(batch_indices)
            data_loader = DataLoader(
                dataset=dataset,
                batch_sampler=SubsetsBatchSampler(
                    subsets,
                    batch_size,
                    intra_subset_shuffle=intra_domain_shuffle,
                    inter_subset_shuffle=shuffle,
                    drop_last=drop_last,
                ),
                num_workers=num_workers,
                pin_memory=True,
            )
            return data_loader

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader



###################################################################
# Trainer 
###################################################################
def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


class PBMC_10K_Trainer:
    
    def __init__(
        self,
        criterion,
        criterion_dab,
        scheduler,
        optimizer,
        scaler,
        device,
        vocab,
        genes
        ):
        self.device = device
        self.vocab = vocab
        self.criterion = criterion
        self.scheduler = scheduler
        self.criterion_dab = criterion_dab
        self.scaler = scaler
        self.optimizer = optimizer
        
        self.input_layer_key = "X_binned"
        self.genes = genes
        if config.load_model is None:
            vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
        vocab.set_default_index(vocab["<pad>"])
        self.gene_ids = np.array(vocab(genes), dtype=int)        

    def train(
        self, 
        model, 
        loader,
        epoch
        ) -> None:
        """
        Train the model for one epoch.
        """
        model.train()
        total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
        total_error = 0.0
        log_interval = config.log_interval
        start_time = time.time()

        num_batches = len(loader)
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            input_values = batch_data["values"].to(self.device)
            target_values = batch_data["target_values"].to(self.device)
            batch_labels = batch_data["batch_labels"].to(self.device)

            src_key_padding_mask = input_gene_ids.eq(self.vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                    MVC=config.GEPC,
                    ECS=config.ecs_thres > 0,
                )

                masked_positions = input_values.eq(mask_value)  # the postions to predict
                loss = loss_mse = self.criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                metrics_to_log = {"train/mse": loss_mse.item()}
                if explicit_zero_prob:
                    loss_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_zero_log_prob
                    metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                if config.GEPC:
                    loss_gepc = self.criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    loss = loss + loss_gepc
                    metrics_to_log.update({"train/mvc": loss_gepc.item()})
                if config.GEPC and explicit_zero_prob:
                    loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_gepc_zero_log_prob
                    metrics_to_log.update(
                        {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                    )
                if config.ecs_thres > 0:
                    loss_ecs = 10 * output_dict["loss_ecs"]
                    loss = loss + loss_ecs
                    metrics_to_log.update({"train/ecs": loss_ecs.item()})
                loss_dab = self.criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + config.dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

            model.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if self.scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    print(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {self.scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            wandb.log(metrics_to_log)

            with torch.no_grad():
                mre = masked_relative_error(
                    output_dict["mlm_output"], target_values, masked_positions
                )

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_gepc += loss_gepc.item() if config.GEPC else 0.0
            total_error += mre.item()
            if batch % log_interval == 0 and batch > 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
                cur_error = total_error / log_interval
                # ppl = math.exp(cur_loss)
                print(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                )
                total_loss = 0
                total_mse = 0
                total_gepc = 0
                total_error = 0
                start_time = time.time()

    def evaluate(
        self,
        model, 
        loader,
        device,
        criterion,
        criterion_dab,
        epoch
        ):
        """
        Evaluate the model on the evaluation data.
        """
        model.eval()
        total_loss = 0.0
        total_error = 0.0
        total_dab = 0.0
        total_num = 0
        with torch.no_grad():
            for batch_data in loader:
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                target_values = batch_data["target_values"].to(device)
                batch_labels = batch_data["batch_labels"].to(device)

                src_key_padding_mask = input_gene_ids.eq(self.vocab[pad_token])
                with torch.cuda.amp.autocast(enabled=config.amp):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if DSBN else None,
                    )
                    output_values = output_dict["mlm_output"]

                    masked_positions = input_values.eq(mask_value)
                    loss = criterion(output_values, target_values, masked_positions)
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

                total_loss += loss.item() * len(input_gene_ids)
                total_error += masked_relative_error(
                    output_values, target_values, masked_positions
                ).item() * len(input_gene_ids)
                total_dab += loss_dab.item() * len(input_gene_ids)
                total_num += len(input_gene_ids)

        wandb.log(
            {
                "valid/mse": total_loss / total_num,
                "valid/mre": total_error / total_num,
                "valid/dab": total_dab / total_num,
                "valid/sum_mse_dab": (total_loss + config.dab_weight * total_dab)
                / total_num,
                "epoch": epoch,
            },
        )

        return total_loss / total_num, total_error / total_num


    def eval_testdata(
        self,
        model: nn.Module,
        adata_t: AnnData,
        include_types: List[str] = ["cls"],
    ):
        """evaluate the model on test dataset of adata_t"""
        model.eval()
        
        # copy adata_t to avoid reuse previously computed results stored in adata_t
        adata_t = adata_t.copy()

        all_counts = (
            adata_t.layers[self.input_layer_key].A
            if issparse(adata_t.layers[self.input_layer_key])
            else adata_t.layers[self.input_layer_key]
        )

        celltypes_labels = adata_t.obs["celltype"].tolist()
        celltypes_labels = np.array(celltypes_labels)

        batch_ids = adata_t.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)

        # Evaluate cls cell embeddings
        if "cls" in include_types:
            print("Evaluating cls cell embeddings")
            tokenized_all = tokenize_and_pad_batch(
                all_counts,
                self.gene_ids,
                max_len=max_seq_len,
                vocab=self.vocab,
                pad_token=pad_token,
                pad_value=pad_value,
                append_cls=True,  # append <cls> token at the beginning
                include_zero_gene=True,
            )
            all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
            src_key_padding_mask = all_gene_ids.eq(self.vocab[pad_token])
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
                cell_embeddings = model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=config.batch_size,
                    batch_labels=torch.from_numpy(batch_ids).long() if DSBN else None,
                    time_step=0,
                    return_np=True,
                )
            cell_embeddings = cell_embeddings / np.linalg.norm(
                cell_embeddings, axis=1, keepdims=True
            )

            adata_t.obsm["X_scGPT"] = cell_embeddings

            results = {}
            try:
                results = eval_scib_metrics(adata_t)
            except Exception as e:
                traceback.print_exc()
                print(e)

            sc.pp.neighbors(adata_t, use_rep="X_scGPT")
            sc.tl.umap(adata_t, min_dist=0.3)
            fig = sc.pl.umap(
                adata_t,
                color=["str_batch"],
                title=[f"batch, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
                frameon=False,
                return_fig=True,
                show=False,
            )

            results["batch_umap"] = fig

            sc.pp.neighbors(adata_t, use_rep="X_scGPT")
            sc.tl.umap(adata_t, min_dist=0.3)
            fig = sc.pl.umap(
                adata_t,
                color=["celltype"],
                title=[
                    f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
                ],
                frameon=False,
                return_fig=True,
                show=False,
            )

            results["celltype_umap"] = fig

        if len(include_types) == 1:
            return results



def main():
    dataset = PBMC_10K_Dataset(dataset_name="PBMC_10K")
    dataloader = PBMC_10K_Dataloader()
    trainer = PBMC_10K_Trainer()
    
    # data
    adata = dataset.get_adata()
    adata_sorted = dataset.get_sorted_adata()
    
    tockenized_data = dataloader.tokenize_input(adata)
    num_batch_types = tockenized_data["num_batch_types"]
    
    
    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None
    define_wandb_metrcis()
    
    model_dir = Path(load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        print(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        dropout=config.dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=config.GEPC,
        do_dab=True,
        use_batch_labels=True,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=DSBN,
        n_input_bins=n_input_bins,
        ecs_threshold=config.ecs_thres,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=True,
        pre_norm=config.pre_norm,
    )
    
    if config.load_model is not None:
        try:
            model.load_state_dict(torch.load(model_file))
            print(f"Loading all model params from {model_file}")
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
                print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    model.to(device)
    wandb.watch(model)

    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)

    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

    # main loop for training
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        train_data_pt, valid_data_pt = dataset.prepare_data(sort_seq_batch=per_seq_batch_sample)
        train_loader = dataloader.prepare_dataloader(
            train_data_pt,
            batch_size=config.batch_size,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
        )
        valid_loader = dataloader.prepare_dataloader(
            valid_data_pt,
            batch_size=config.batch_size,
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
        )

        if config.do_train:
            trainer.train(
                model,
                loader=train_loader,
            )
        val_loss, val_mre = trainer.evaluate(
            model,
            loader=valid_loader,
        )
        elapsed = time.time() - epoch_start_time
        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
        )
        print("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            print(f"Best model with score {best_val_loss:5.4f}")

        if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
            print(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

            # eval on testdata
            results = trainer.eval_testdata(
                best_model,
                adata_t=adata_sorted if per_seq_batch_sample else adata,
                include_types=["cls"],
            )
            results["batch_umap"].savefig(
                save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300
            )

            results["celltype_umap"].savefig(
                save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300
            )
            metrics_to_log = {"test/" + k: v for k, v in results.items()}
            metrics_to_log["test/batch_umap"] = wandb.Image(
                str(save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png"),
                caption=f"celltype avg_bio epoch {best_model_epoch}",
            )

            metrics_to_log["test/celltype_umap"] = wandb.Image(
                str(save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png"),
                caption=f"celltype avg_bio epoch {best_model_epoch}",
            )
            metrics_to_log["test/best_model_epoch"] = best_model_epoch
            wandb.log(metrics_to_log)
            wandb.log({"avg_bio": results.get("avg_bio", 0.0)})

        scheduler.step()
        
    # save the best model
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    
    artifact = wandb.Artifact(f"best_model", type="model")
    glob_str = os.path.join(save_dir, "best_model.pt")
    artifact.add_file(glob_str)
    run.log_artifact(artifact)

    run.finish()
    wandb.finish()
    gc.collect()


if __name__=="__main__":
    main()
    
    