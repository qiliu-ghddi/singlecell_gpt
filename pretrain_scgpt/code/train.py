"""
Pretraining the scGPT (Transformer) Model
Author: Qi Liu
"""

import sys
import json
import torch
from torch import nn
from pathlib import Path
from dataset import DataPreprocessor, PBMC_10K_Dataset

sys.path.insert(0, "../")
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.loss import masked_mse_loss

from trainer import scGPT_Trainer


class scGPT_Config:

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    def __init__(
        self,
        model_dir
        ):
        self.model_dir = Path(model_dir)
        self.model_config_file = self.model_dir / "args.json"
        self.model_file = self.model_dir / "best_model.pt"
        self.vocab_file = self.model_dir / "vocab.json"

        vocab = GeneVocab.from_file(self.vocab_file)
        for s in self.special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        self.vocab = vocab

        # model
        with open(self.model_config_file, "r") as f:
            model_configs = json.load(f)
            self.model_configs = model_configs

        # embsize = model_configs["embsize"]
        # nhead = model_configs["nheads"]
        # d_hid = model_configs["d_hid"]
        # nlayers = model_configs["nlayers"]
        # n_layers_cls = model_configs["n_layers_cls"]


def main(args=None):
    # Global configuration
    
    config_dir = "/home/qiliu02/GHDDI/DS-group/ghddixcre_singlecell_gpt/dev_pretrain_dataloader/save/scGPT_human"
    config = scGPT_Config(config_dir)
    print(config)
    print("="*90)
        
    # Dataset and dataloader
    
    dataset_name = "PBMC_10K"
    if dataset_name == "PBMC_10K":
        ds = PBMC_10K_Dataset()
    else:
        # TODO: using our own datasets
        ds = PBMC_10K_Dataset()
    
    batch_size = 10  # config.model_configs["batch_size"]
    per_seq_batch_sample = True
    
    batch_ids = ds.adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))
    dp = DataPreprocessor()
    
    
    # Model and training config
    
    DSBN = False
    load_model = None
    fast_transformer = False
    GEPC = False 
    do_dab = False
    use_batch_labels = False
    explicit_zero_prob = True
    ecs_thres = 0.8
    
    vocab = config.vocab
    embsize = 128  # config.model_configs["embsize"]
    nhead = 4  # config.model_configs["nheads"]
    d_hid = 128  # config.model_configs["d_hid"]
    nlayers = 4  # config.model_configs["nlayers"]

    pad_token = config.model_configs["pad_token"]
    pad_value = config.model_configs["pad_value"]
    n_input_bins = config.model_configs["n_bins"]
    use_fast_transformer = False  # config.model_configs["fast_transformer"]
    dropout = 0.2
    pre_norm = False
    
    ntokens = len(vocab)  # size of vocabulary
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f"... device: {device}")
    
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=GEPC,
        do_dab=do_dab,
        use_batch_labels=use_batch_labels,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=DSBN,
        n_input_bins=n_input_bins,
        ecs_threshold=ecs_thres,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=use_fast_transformer,
        pre_norm=pre_norm,
    )
    model.to(device)
    
    
    # Training
    
    lr = config.model_configs["lr"]
    amp = True
    eps = 1e-4 if amp else 1e-8
    schedule_ratio = 0.9
    
    criterion = masked_mse_loss
    # criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        eps=eps
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 1, gamma=schedule_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    trainer = scGPT_Trainer(
        config=config,
        model=model,
        device=device,
        dataset=ds,
        datapreprocessor=dp,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        
    )
    trainer.train()
    
    
if __name__=="__main__":
    main()
