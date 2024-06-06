# build large-scale data in scBank format from a group of AnnData objects
import gc
import json
from pathlib import Path
import argparse
import shutil
import traceback
from typing import Dict, List, Optional
import warnings
import numpy as np
import os
from datetime import datetime
import logging

import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

import sys
# sys.path.append('/home/qiliu02/GHDDI/DS-group/ghddixcre_singlecell_gpt/contribs/scGPT')
sys.path.append('..')
import scgpt
from scgpt import scbank


parser = argparse.ArgumentParser(
    description="Build large-scale data in scBank format from a group of AnnData objects"
)
parser.add_argument(
    "--input-dir",
    type=str,
    required=True,
    help="Directory containing AnnData objects",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./data.scb",
    help="Directory to save scBank data, by default will make a directory named "
    "data.scb in the current directory",
)
parser.add_argument(
    "--include-files",
    type=str,
    nargs="*",
    help="Space separated file names to include, default to all files in input_dir",
)

parser.add_argument(
    "--include-ids",
    type=int,
    nargs="*",
    help="Space separated ids to include, default to all files in input_dir",
)

parser.add_argument(
    "--metainfo",
    type=str,
    default=None,
    help="Json file containing meta information for each dataset, default to None.",
)

# vocabulary
parser.add_argument(
    "--vocab-file",
    type=str,
    required=True,
    # default=None,
    help="File containing the gene vocabulary, default to None. If None, will "
    "use the default gene vocabulary from scFormer, which use HGNC gene symbols.",
)

parser.add_argument(
    "--N",
    type=int,
    default=200000,
    help="Hyperparam for filtering genes, default to 200000.",
)

args = parser.parse_args()

logging.basicConfig(
    filename='./build_large_scale_data.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

vocab_path = Path(args.vocab_file)
print(args)
logging.info("-" * 89)
logging.info(f"args: {args}")
logging.info("-" * 89)

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
files = [f for f in input_dir.glob("*/*.h5ad")]
# files = [f for f in input_dir.glob("*.h5ad")]
print(f"Found {len(files)} files in {input_dir}")
if args.include_files is not None:
    files = [f for f in files if f.name in args.include_files]
if args.metainfo is not None:
    metainfo = json.load(open(args.metainfo))
    files = [f for f in files if f.stem in metainfo]
    include_obs = {
        f.stem: {"disease": metainfo[f.stem]["include_disease"]}
        for f in files
        if "include_disease" in metainfo[f.stem]
    }

if args.include_ids is not None:
    files = [f for f in files if int(f.parts[-2]) in args.include_ids]
print(f"Found {len(files)} files filtered.")

    
# if args.metainfo is not None:
#     metainfo = json.load(open(args.metainfo))
#     files = [f for f in files if f.stem in metainfo]
#     include_obs = {
#         f.stem: {"disease": metainfo[f.stem]["include_disease"]}
#         for f in files
#         if "include_disease" in metainfo[f.stem]
#     }

if args.vocab_file is None:
    vocab = scgpt.tokenizer.get_default_gene_vocab()
else:
    vocab = scgpt.tokenizer.GeneVocab.from_file(args.vocab_file)


# # preprocessing data
def preprocess(
    adata: sc.AnnData,
    main_table_key: str = "counts",
    include_obs: Optional[Dict[str, List[str]]] = None,
    N = 200000
) -> sc.AnnData:
    """
    Preprocess the data for scBank. This function will modify the AnnData object in place.

    Args:
        adata: AnnData object to preprocess
        main_table_key: key in adata.layers to store the main table
        include_obs: dict of column names and values to include in the main table

    Returns:
        The preprocessed AnnData object
    """
    if include_obs is not None:
        # include only cells that have the specified values in the specified columns
        for col, values in include_obs.items():
            adata = adata[adata.obs[col].isin(values)]

    if adata.raw is not None:
        adata.X = adata.raw.X.copy()

    # filter genes
    print(f".. before filter_genes: {adata},{adata.shape}")
    logging.info(f"before filter_genes: {adata},{adata.shape}")
    
    sc.pp.filter_genes(adata, min_counts=(3 / 10000) * N)
    
    print(f".. after filter_genes: {adata},{adata.shape}")
    logging.info(f"after filter_genes: {adata},{adata.shape}")
    # sc.pp.filter_genes(adata, min_counts=(3 / 10000) * N)

    # TODO: add binning in sparse matrix and save in separate datatable
    # preprocessor = Preprocessor(
    #     use_key="X",  # the key in adata.layers to use as raw data
    #     filter_gene_by_counts=False,  # step 1
    #     filter_cell_by_counts=False,  # step 2
    #     normalize_total=False,  # 3. whether to normalize the raw data and to what sum
    #     log1p=False,  # 4. whether to log1p the normalized data
    #     binning=51,  # 6. whether to bin the raw data and to what number of bins
    #     result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    # )
    # preprocessor(adata)

    try:
        adata.layers[main_table_key] = adata.raw.X.copy()  # preserve counts
    except Exception as e:
        print(f"Error {e}! while preprocessing adata")
        adata.layers[main_table_key] = adata.X.copy()  # preserve counts

    # adata.layers[main_table_key] = adata.X.copy()  # preserve counts
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    # adata.raw = adata  # freeze the state in `.raw`

    # apply a hard clip to the data for now
    print(
        f"original mean and max of counts: {adata.layers[main_table_key].mean():.2f}, "
        f"{adata.layers[main_table_key].max():.2f}"
    )
    # if isinstance(adata.layers[main_table_key], np.ndarray):
    #     adata.layers[main_table_key] = adata.layers[main_table_key].clip(0, 30)
    # else:  # assume it is a sparse matrix
    #     adata.layers[main_table_key].data = adata.layers[main_table_key].data.clip(0, 30)

    return adata


main_table_key = "counts"
token_col = "feature_name"
for i, f in enumerate(files):
    try:
        print(f".. {i} {f}")
        logging.info(f".. {i} {f}")
        adata = sc.read(f, cache=True)
        adata = preprocess(
            adata, main_table_key, 
            N = args.N
        )
        print(f"read {adata.shape} valid data from {f}")

        # TODO: CHECK AND EXPAND VOCABULARY IF NEEDED
        # NOTE: do not simply expand, need to check whether to use the same style of gene names

        # BUILD SCBANK DATA
        to_dir = f"{output_dir}/{f.parts[-2]}.scb"        
        db = scbank.DataBank.from_anndata(
            adata,
            vocab=vocab,
            to=to_dir,
            # to=output_dir / f"{f.stem}.scb",
            main_table_key=main_table_key,
            token_col=token_col,
            immediate_save=False,
        )
        db.meta_info.on_disk_format = "parquet"
        # sync all to disk
        db.sync()

        # clean up
        del adata
        del db
        gc.collect()
    except Exception as e:
        traceback.print_exc()
        warnings.warn(f"failed to process {f.name}: {e}")
        shutil.rmtree(output_dir / f"{f.stem}.scb", ignore_errors=True)

# or run scbank.DataBank.batch_from_anndata(files, to=args.output_dir)
# test loading from disk
# db = scbank.DataBank.from_path(args.output_dir)

# run this to copy all parquet datatables to a single directory
target_dir = output_dir / f"all_{main_table_key}"
target_dir.mkdir(parents=True, exist_ok=True)
for f in files:
    output_parquet_dt = (
        output_dir / f"{f.parts[-2]}.scb" / f"{main_table_key}.datatable.parquet"
    )    
    if output_parquet_dt.exists():
        src = output_parquet_dt.resolve()
        dst = target_dir.resolve() / f"{f.parts[-2]}.datatable.parquet"
        print(f"src: {src} <--> dst: {dst}")
        os.symlink(src, dst)    
