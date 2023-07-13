import scvi
import torch
import numpy as np
import scanpy as sc
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Optional, Union
from anndata import AnnData
import scanpy as sc
import warnings
from typing import List, Tuple, Dict, Union, Optional
from scipy.sparse import issparse
from scanpy.get import _get_obs_rep, _set_obs_rep
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind


# from scgpt import logger
import sys
sys.path.append(".")
sys.path.append("..")
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt import SubsetsBatchSampler


def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int
):
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
    # TODO: use torch.quantile and torch.bucketize

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = np.digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = np.digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)


class Preprocessor:
    """
    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """

    def __init__(
        self,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        result_normed_key: Optional[str] = "X_normed",
        log1p: bool = False,
        result_log1p_key: str = "X_log1p",
        subset_hvg: Union[int, bool] = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3",
        binning: Optional[int] = None,
        result_binned_key: str = "X_binned",
    ):
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor
        self.binning = binning
        self.result_binned_key = result_binned_key

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None):
        key_to_process = self.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        # step 1: filter genes
        if self.filter_gene_by_counts:
            print("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )

        # step 2: filter cells
        if isinstance(self.filter_cell_by_counts, int):
            print("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )

        # step 3: normalize total
        if self.normalize_total:
            print("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                if isinstance(self.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.result_normed_key or key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        # step 4: log1p
        if self.log1p:
            print("Log1p transforming ...")
            if is_logged:
                print(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.result_log1p_key,
                )
                key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)

        # step 5: subset hvg
        if self.subset_hvg:
            print("Subsetting highly variable genes ...")
            if batch_key is None:
                print(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg
                if isinstance(self.subset_hvg, int)
                else None,
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=True,
            )

        # step 6: binning
        if self.binning:
            print("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.binning)
                )
            n_bins = self.binning  # NOTE: the first bin is always a spectial for zero
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)
            layer_data = layer_data.A if issparse(layer_data) else layer_data
            for row in layer_data:
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the each category
                # has different relative meaning across datasets
                non_zero_digits = self._digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            adata.layers[self.result_binned_key] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)

    def _digitize(self, x: np.ndarray, bins: np.ndarray):
        assert x.ndim == 1 and bins.ndim == 1

        left_digits = np.digitize(x, bins)
        right_difits = np.digitize(x, bins, right=True)

        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_difits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None):
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True


class CRE_SingleCell_Config:
    
    def __init__(
        self, 
        data_dir=Path("/home/lushi02/project/sl_data/cellgene20230606")
        ):
        self.data_dir = Path(data_dir)
        self.h5ad_fns = self.get_h5ad_fns()
        self.rds_fns = self.get_h5ad_fns()
        self.size = len(self.h5ad_fns)
        
    def get_h5ad_fns(self):
        file_pattern = "*/local.h5ad"
        data_fns = list(self.data_dir.glob(file_pattern))
        data_fns_sorted = sorted(data_fns, key=lambda x: int(x.parts[-2]))
        return list(data_fns_sorted)

    def get_rds_fns(self):
        file_pattern = "*/local.rds"
        data_fns = list(self.data_dir.glob(file_pattern))
        data_fns_sorted = sorted(data_fns, key=lambda x: int(x.parts[-2]))
        return list(data_fns_sorted)


class CRE_SingleCell_Dataset(Dataset):
    
    def __init__(
        self,
        config,
        transforms=None
        ):
        self.config = config
        self.transforms = transforms
    
    def __getitem__(self, index):
        h5ad_fn = self.config.h5ad_fns[index]
        adata = sc.read_h5ad(h5ad_fn)
        adata.uns['h5ad_fn'] = h5ad_fn
        if self.transforms:
            adata = self.transforms(adata)
        return adata
    
    def __len__(self):
        return self.config.size


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


class DataPreprocessor:
    
    def preprocess_pbmc_10k(self, adata):
        dataset_name = "PBMC_10K"
        batch_key = "str_batch" if dataset_name != "heart_cell" else None
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
        preprocessor(adata, batch_key=batch_key)
        return adata

    def preprocess_human_data(self, adata, dataset_name=None):
        dataset_name = dataset_name if dataset_name is not None else "pretraining_human"
        batch_key = "str_batch" if dataset_name != "heart_cell" else None
        data_is_raw = True
        n_hvg = None 
        n_bins = 51
        
        # set up the preprocessor, use the args to config the workflow
        preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=3,  # step 1
            filter_cell_by_counts=False,  # step 2
            normalize_total=1e4,  # step 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=data_is_raw,  # step 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=n_hvg,  # step 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
            binning=n_bins,  # step 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )
        preprocessor(adata, batch_key=batch_key)
        return adata

    def get_all_counts_from_adata(
        self, 
        adata,
        input_layer_key = "X_binned"
        ):
        all_counts = (
            adata.layers[input_layer_key].A
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        return all_counts        

    def get_genes_from_adata(
        self, 
        adata,
        var_name="gene_name"
        ):
        genes = adata.var[var_name].tolist()
        return genes  


    def get_celltypes_labels_from_adata(
        self, 
        adata,
        obs_name="celltype"
        ):
        celltypes_labels = adata.obs[obs_name].tolist()  # make sure count from 0
        num_types = len(set(celltypes_labels))
        celltypes_labels = np.array(celltypes_labels)
        return celltypes_labels  

    def get_batch_ids_from_adata(
        self, 
        adata,
        obs_name="batch_id"
        ):
        batch_ids = adata.obs[obs_name].tolist()
        batch_ids = np.array(batch_ids)
        return batch_ids  

    def get_input_data_from_adata(
        self, 
        adata
        ):
        # input_layer_key = "X_binned"
        # all_counts = (
        #     adata.layers[input_layer_key].A
        #     if issparse(adata.layers[input_layer_key])
        #     else adata.layers[input_layer_key]
        # )
        
        # genes = adata.var["gene_name"].tolist()
        # celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
        # num_types = len(set(celltypes_labels))
        # celltypes_labels = np.array(celltypes_labels)
        
        # batch_ids = adata.obs["batch_id"].tolist()
        # num_batch_types = len(set(batch_ids))
        # batch_ids = np.array(batch_ids)
        
        all_counts = self.get_all_counts_from_adata(adata)
        celltypes_labels = self.get_celltypes_labels_from_adata(adata)
        genes = self.get_genes_from_adata(adata)
        batch_ids = self.get_batch_ids_from_adata(adata)
        num_batch_types = len(set(batch_ids))
        
        dat = {
            "all_counts": all_counts, 
            "celltypes_labels": celltypes_labels, 
            "genes": genes,
            "num_batch_types": num_batch_types,
            "batch_ids": batch_ids
        }
        return dat
    
    def get_input_data(
        self,
        adata,
        test_size=0.1, shuffle=True
        ):
        dat = self.get_input_data_from_adata(adata)
        all_counts = dat["all_counts"]
        celltypes_labels = dat["celltypes_labels"]
        batch_ids = dat["batch_ids"]
        
        (
            train_data,
            valid_data,
            train_celltype_labels,
            valid_celltype_labels,
            train_batch_labels,
            valid_batch_labels,
        ) = train_test_split(
            all_counts, celltypes_labels, batch_ids, test_size=test_size, shuffle=shuffle
        )
        
        return (
            train_data,
            valid_data,
            train_celltype_labels,
            valid_celltype_labels,
            train_batch_labels,
            valid_batch_labels,
        )

    def prepare_data(
        self,
        adata,
        sort_seq_batch=False):
        mask_ratio = 0.4  # config.mask_ratio
        max_seq_len = 1200
        pad_token = "<pad>"
        special_tokens = [pad_token, "<cls>", "<eoc>"]
        pad_value = -2
        mask_value = -1
        genes = adata.var["gene_name"].tolist()
        
        vocab = Vocab(
                VocabPybind(genes + special_tokens, None)
            )  # bidirectional lookup [gene <-> int]
        vocab.set_default_index(vocab["<pad>"])
        gene_ids = np.array(vocab(genes), dtype=int)
        
        (
            train_data,
            valid_data,
            train_celltype_labels,
            valid_celltype_labels,
            train_batch_labels,
            valid_batch_labels,
        ) = self.get_input_data(adata)

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
            f"random masking at epoch, ratio of masked values in train: ",
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

    def prepare_dataloader(
        self,
        data_pt: Dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool = False,
        intra_domain_shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
        per_seq_batch_sample = True
    ):
        dataset = SeqDataset(data_pt)
        
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
        else:
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                pin_memory=True,
            )
            return data_loader


class PBMC_10K_Dataset:
    
    def __init__(self, config=None):
        self.dataset_name = "PBMC_10K"
        self.config = config
        self.raw_adata = self.load_raw_pbmc10k()
        self.adata = self.load_preprocessed_dataset()

    def load_raw_pbmc10k(self):
        adata = scvi.data.pbmc_dataset()
        return adata

    def _load_dataset(self):
        data_is_raw = False
        dataset_name = "PBMC_10K"
        ori_batch_col = "batch"
                
        adata = scvi.data.pbmc_dataset()  # 11990 × 3346
        adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
        adata.var = adata.var.set_index("gene_symbols")
        data_is_raw = True
        
        # make the batch category column
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()
        
        return adata
        
    def load_preprocessed_dataset(
        self,
        per_seq_batch_sample=True
        ):
        
        adata = self._load_dataset()
        dpp = DataPreprocessor()
        adata_preprocessed = dpp.preprocess_pbmc_10k(adata)
        
        if per_seq_batch_sample:
            # sort the adata by batch_id in advance
            adata_sorted = adata_preprocessed[adata_preprocessed.obs["batch_id"].argsort()].copy()
            return adata_sorted
        else:
            return adata_preprocessed

    def vars_from_adata(
        self, 
        adata,
        ):
        from sklearn.model_selection import train_test_split
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
        
        if self.config and self.config.load_model is not None:
            # load_model = True    
            model_dir = Path(self.config.load_model)
            model_config_file = model_dir / "args.json"
            model_file = model_dir / "best_model.pt"
            vocab_file = model_dir / "vocab.json"
            vocab = GeneVocab.from_file(vocab_file)
        else:
            # load_model = False
            # settings for input and preprocessing
            pad_token = "<pad>"
            special_tokens = [pad_token, "<cls>", "<eoc>"]
                        
            vocab = Vocab(
                VocabPybind(genes + special_tokens, None)
            )  # bidirectional lookup [gene <-> int]
            
        vocab.set_default_index(vocab["<pad>"])
        gene_ids = np.array(vocab(genes), dtype=int)
        
        
        dat = {
            "input_layer_key": input_layer_key,
            
            "all_counts": all_counts,
            
            "genes": genes,
            "num_types": num_types,
            "celltypes_labels": celltypes_labels,
            
            "num_batch_types": num_batch_types,
            "batch_ids": batch_ids,
            
            "vocab": vocab,
            "gene_ids": gene_ids
        }
        
        return dat


    def train_test_split(
        self, 
        adata,
        test_size=0.1, 
        shuffle=True
        ):
        from sklearn.model_selection import train_test_split
        dat = self.vars_from_adata(adata)
        all_counts = dat["all_counts"]
        celltypes_labels = dat["celltypes_labels"]
        batch_ids = dat["batch_ids"]

        (
            train_data,
            valid_data,
            train_celltype_labels,
            valid_celltype_labels,
            train_batch_labels,
            valid_batch_labels,
        ) = train_test_split(
            all_counts, 
            celltypes_labels, 
            batch_ids, 
            test_size=test_size, 
            shuffle=shuffle
        )
        return (
            train_data,
            valid_data,
            train_celltype_labels,
            valid_celltype_labels,
            train_batch_labels,
            valid_batch_labels,
        )

