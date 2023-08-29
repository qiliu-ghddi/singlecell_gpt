import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import torch
import numpy as np
from typing import Dict, Optional, Union
from datasets import Dataset, load_dataset

sys.path.insert(0, "../")
sys.path.append('/home/qiliu02/GHDDI/DS-group/ghddixcre_singlecell_gpt/contribs/scGPT/')
import scgpt
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab

t_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
Path("./log").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename='./log/binning_mask_allcounts.log',  # Specify the file name for the log
    level=logging.DEBUG,  # Set the minimum log level (you can adjust this)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info(t_stamp)


parser = argparse.ArgumentParser("Processing allcounts.")
parser.add_argument(
    '--data_source',
    type=str,
    default='./preprocessed/all_counts/'
)
parser.add_argument(
    '--cache_dir',
    type=str,
    default='./cache/'
)
parser.add_argument(
    '--masked_dir',
    type=str,
    default='./masked/'
)
parser.add_argument(
    '--binned_dir',
    type=str,
    default='./binned/'
)
parser.add_argument(
    '--n_bins',
    type=int,
    default=51
)
# vocabulary
parser.add_argument(
    "--vocab_file",
    type=str,
    default=None,
    help="File containing the gene vocabulary, default to None. If None, will "
    "use the default gene vocabulary from scFormer, which use HGNC gene symbols.",
)
args = parser.parse_args()



# Global variables
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]

if args.vocab_file is None:
    vocab = scgpt.tokenizer.get_default_gene_vocab()
else:
    # vocab_file = "./default_census_vocab.json"
    vocab = scgpt.tokenizer.GeneVocab.from_file(args.vocab_file)

for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
ntokens = len(vocab)  # size of vocabulary, 60694
logging.info(f"ntokens: {ntokens}")


def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins."""
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


def _prepare_data(
    data,
    gene_ids,
    vocab,
    max_seq_len=1200,
    mask_ratio=0.4,
    mask_value=-1,
    pad_value=-2,
    append_cls=True,
    include_zero_gene=True
):
    tokenized_x = tokenize_and_pad_batch(
        data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=append_cls,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene
    )
    masked_values_x = random_mask_value(
        tokenized_x["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    data_pt = {
        "gene_ids": tokenized_x['genes'].view(-1),
        "values": masked_values_x.view(-1),  # to input the masked values
        "target_values": tokenized_x['values'].view(-1),  # unmasked values
        
        # "gene_ids": np.array(tokenized_x['genes']).flatten().tolist(),
        # "values": np.array(masked_values_x).flatten().tolist(),  # to input the masked values
        # "target_values": np.array(tokenized_x['values']).flatten().tolist(),  # unmasked values
    }
    return data_pt

def prepare_data(
    x,
    vocab,
    n_bins=51
):
    x_arr = np.array(x['expressions'])
    data = x_arr.reshape(1, -1)
    
    data = binning(data, n_bins=n_bins)
    
    gene_ids = x['genes']
    data_pt = _prepare_data(data, gene_ids, vocab=vocab)
    return data_pt
    

data_source = args.data_source
cache_dir = args.cache_dir
Path(cache_dir).mkdir(parents=True, exist_ok=True)
n_bins = args.n_bins

parquet_files = [str(f) for f in Path(data_source).glob("*.parquet")]
num_parquet_files = len(parquet_files)
logging.info(f"#parquet_files: {num_parquet_files}")

logging.info(f"Loading raw_dataset ...")
raw_dataset = load_dataset(
    "parquet",
    data_files=parquet_files,
    split="train",
    cache_dir=str(cache_dir)
)
logging.info(f"raw_dataset {raw_dataset}")


binned_dataset_dir = Path(args.binned_dir)
binned_dataset_dir.mkdir(parents=True, exist_ok=True)
binned_dataset_path = f"{binned_dataset_dir}/databanks_binned.parquet"

logging.info(f"Creating binned_dataset ...")
binned_dataset = raw_dataset.map(
    lambda example: prepare_data(
            example,
            vocab,
            n_bins=n_bins
        ),
        num_proc=len(os.sched_getaffinity(0)),
        remove_columns=raw_dataset.column_names
    )
binned_dataset.to_parquet(str(binned_dataset_path))
logging.info(f"Finished.")

