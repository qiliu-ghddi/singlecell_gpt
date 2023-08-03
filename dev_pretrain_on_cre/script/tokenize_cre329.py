"""
Tokenize all binned X and save the results to a .pt file.
"""
import time
import gc
import sys
import json
import logging
import warnings
from datetime import datetime

import torch
import numpy as np
import anndata as ad
import scanpy as sc
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Dict
from scipy.sparse import issparse
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

sys.path.append("..")
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt import SubsetsBatchSampler
from preprocess import Preprocessor

warnings.filterwarnings('ignore')


def preprocess_cre329(adata, dataset_name='cre329'):
    batch_key = None  # "str_batch" if dataset_name != "heart_cell" else None
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


def tokenize_preprocessed_adata(
    adata,
    vocab,
    input_layer_key = "X_binned",
    gene_key="feature_name",
    pad_value=-2,
    max_seq_len=1200+1
    ):
    
    if issparse(adata.layers[input_layer_key]):
        all_counts = adata.layers[input_layer_key].A
    else:
        all_counts = adata.layers[input_layer_key]
    genes = adata.var[gene_key].tolist()
    
    if vocab is None:
        # if config.load_model is None:
        # bidirectional lookup [gene <-> int]
        vocab = Vocab(VocabPybind(genes + special_tokens, None))
        vocab.set_default_index(vocab["<pad>"])
        
    # convert gene names to gene indices    
    gene_ids = np.array(vocab(genes), dtype=int)
    
    tokenized_data = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,
            include_zero_gene=False,  # !!!
        )
    return tokenized_data
  


# global settings
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
vocab_file = "default_census_vocab.json"
vocab = GeneVocab.from_file(vocab_file)
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
        

t_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_dir = Path("/home/lushi02/project/sl_data/cellgene20230606")


ids = [4, 6, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 31, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 56, 57, 58, 59, 71, 72, 85, 86, 87, 88, 114, 116, 120, 121, 122, 123, 124, 142, 145, 147, 148, 149, 150, 151, 152, 153, 157, 163, 164, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 222, 255, 268, 285, 289, 290, 291, 298, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 373, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 393, 397, 399, 415, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 438, 439, 440, 444, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 462, 475, 476, 477, 478, 479, 481, 483, 500, 501, 502, 503, 504, 505, 506, 507, 509, 514, 515, 516, 517, 519, 524, 525, 526, 527, 528, 529, 530, 531, 533, 534, 537, 538, 539, 540, 541, 542, 543, 546, 547, 548, 551, 552, 557, 559, 560, 563, 564, 565, 567, 568, 570, 571, 572, 573, 574, 575, 577, 579, 580, 581, 584, 585, 587, 590, 594, 601, 605, 606, 608, 609, 610, 611, 612, 613, 617, 618, 621, 622, 623, 624, 625, 627, 628, 629, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 745, 750, 752, 755, 756, 757, 758, 759, 761, 763, 765, 767, 768, 769, 770, 773, 782, 783, 787, 789, 794, 795, 796, 800, 804]

# ids = [4, 6, 8]

h5ad_fns = {
    id_: f"{data_dir}/{id_}/local.h5ad" for id_ in ids
}


# Steps:
# 1. read .h5ad
# 2. preprocess
# 3. tokenize and pad
# 4. save
# 5. merge and train_test_split
# 6. input for training
# id_ = 8


save_dir = Path(f"save/cre329_tokenized/{t_stamp}")
save_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=f'{save_dir}/run.log',  # Specify the file name for the log
    level=logging.DEBUG,  # Set the minimum log level (you can adjust this)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

run_config = {
    'data_dir': str(data_dir),
    'ids': ids,
    'save_dir': str(save_dir),
    'vocab_file': "default_census_vocab.json",
    'input_layer_key': "X_binned",
    'gene_key': "feature_name",
    'pad_value': -2,
    'max_seq_len': 1201
}
vocab.save_json(f"{save_dir}/vocab.json")

with open(f"{save_dir}/config.json", "w") as fout:
    json.dump(run_config, fout, indent=4)
logging.info(run_config)

for id_ in ids:
    try:
        h5ad_fn = h5ad_fns[id_]
        logging.info(f" ... {id_} {h5ad_fn}")

        t_start = time.time()
        adata = sc.read_h5ad(h5ad_fn, backed=None)
        # logging.info(adata)

        do_preprocess = True
        if do_preprocess:
            adata = preprocess_cre329(adata)
            logging.info(adata)

        tokenized_data = tokenize_preprocessed_adata(
            adata,
            vocab,
            input_layer_key="X_binned",
            gene_key="feature_name",
            pad_value=-2,
            max_seq_len=1200+1
            )
        # logging.info(tokenized_data)
        t_end = time.time()
        logging.info(f"time elasped: {t_end-t_start}")
        
        torch.save(tokenized_data, f"{save_dir}/cre{id_}_tokenized_data.pt")

        del adata
        del tokenized_data
        gc.collect()
    except Exception as e:
        logging.info(f"Error! {e}")
        continue
