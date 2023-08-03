

#######################################
# Change 1
#######################################
# hyperparameter_defaults = dict(
# seed=42,
# dataset_name="PBMC_10K", # Dataset name
# do_train=True, # Flag to indicate whether to do update model parameters during training
# load_model="../save/scGPT_human", # Path to pre-trained model
# GEPC=True,  # Gene expression modelling for cell objective
# ecs_thres=0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
# dab_weight=1.0, # DAR objective weight for batch correction
# mask_ratio=0.4, # Default mask ratio
# epochs=15, # Default number of epochs for fine-tuning
# n_bins=51, # Default number of bins for value binning in data pre-processing
# lr=1e-4, # Default learning rate for fine-tuning
# batch_size=16, # Default batch size for fine-tuning
# layer_size=128,
# nlayers=4,
# nhead=4, # if load model, batch_size, layer_size, nlayers, nhead will be ignored
# dropout=0.2, # Default dropout rate during model fine-tuning
# schedule_ratio=0.9,  # Default rate for learning rate decay
# save_eval_interval=5, # Default model evaluation interval
# log_interval=100, # Default log interval
# fast_transformer=True, # Default setting
# pre_norm=False, # Default setting
# amp=True,  # # Default setting: Automatic Mixed Precision
# )

hyperparameter_defaults = {
    'seed': 42,
    'dataset_name': "PBMC_10K",  # Dataset name
    'do_train': True,  # Flag to indicate whether to do update model parameters during training
    'load_model': "../save/scGPT_human",  # Path to pre-trained model
    'GEPC': True,  # Gene expression modelling for cell objective
    'ecs_thres': 0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    'dab_weight': 1.0,  # DAR objective weight for batch correction
    'mask_ratio': 0.4,  # Default mask ratio
    'epochs': 15,  # Default number of epochs for fine-tuning
    'n_bins': 51,  # Default number of bins for value binning in data pre-processing
    'lr': 1e-4,  # Default learning rate for fine-tuning
    'batch_size': 16,  # Default batch size for fine-tuning
    'layer_size': 128,
    'nlayers': 4,
    'nhead': 4,  # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    'dropout': 0.2,  # Default dropout rate during model fine-tuning
    'schedule_ratio': 0.9,  # Default rate for learning rate decay
    'save_eval_interval': 5,  # Default model evaluation interval
    'log_interval': 100,  # Default log interval
    'fast_transformer': True,  # Default setting
    'pre_norm': False,  # Default setting
    'amp': True,  # Default setting: Automatic Mixed Precision
}

hyperparameter_comments = {
    'seed': "Random seed for reproducibility",
    'dataset_name': "Dataset name",
    'do_train': "Flag to indicate whether to update model parameters during training",
    'load_model': "Path to pre-trained model",
    'GEPC': "Gene expression modelling for cell objective",
    'ecs_thres': "Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable",
    'dab_weight': "DAR objective weight for batch correction",
    'mask_ratio': "Default mask ratio",
    'epochs': "Default number of epochs for fine-tuning",
    'n_bins': "Default number of bins for value binning in data pre-processing",
    'lr': "Default learning rate for fine-tuning",
    'batch_size': "Default batch size for fine-tuning",
    'layer_size': "Number of units in the hidden layers",
    'nlayers': "Number of transformer layers",
    'nhead': "Number of attention heads, ignored if load_model is used",
    'dropout': "Default dropout rate during model fine-tuning",
    'schedule_ratio': "Default rate for learning rate decay",
    'save_eval_interval': "Default model evaluation interval",
    'log_interval': "Default log interval",
    'fast_transformer': "Default setting: Use fast transformer implementation",
    'pre_norm': "Default setting: Use pre-normalization",
    'amp': "Default setting: Use Automatic Mixed Precision",
}

# Combine the two dictionaries
hyperparameter_dict_with_comments = {
    param: {
        'default': hyperparameter_defaults[param], 
        'comment': hyperparameter_comments[param]
        } for param in hyperparameter_defaults
    }


scGPT_human_official_config = {
  "data_source": "/scratch/ssd004/datasets/cellxgene/scb_strict/human",
  "save_dir": "/scratch/ssd004/datasets/cellxgene/save/cellxgene_census_human-May23-08-36-2023",
  "load_model": None,
  "n_hvg": None,
  "valid_size_or_ratio": 0.003,
  

  "dist_backend": "nccl",
  "grad_accu_steps": 1,


  "pad_token": "<pad>",
  "input_style": "binned",
  "input_emb_style": "continuous",
  "n_bins": 51,
  "max_seq_len": 1200,
  "training_tasks": "both",
  "dist_url": "tcp://gpu188.cluster.local:53833",
  "mask_ratio": [
    0.25,
    0.5,
    0.75
  ],
  "trunc_by_sample": True,
  "vocab_path": "/scratch/ssd004/datasets/cellxgene/scFormer/scformer/tokenizer/default_census_vocab.json",
  "rank": 0,
  "batch_size": 32,
  "eval_batch_size": 64,
  "epochs": 6,
  "lr": 0.0001,
  "scheduler_interval": 100,
  "scheduler_factor": 0.99,
  "warmup_ratio_or_step": 10000.0,
  "no_cls": True,
  "no_cce": True,
  "fp16": True,
  "fast_transformer": True,
  "nlayers": 12,
  "nheads": 8,
  "embsize": 512,
  "d_hid": 512,
  "dropout": 0.2,
  "n_layers_cls": 3,
  "log_interval": 9000,
  "save_interval": 27000,
  "mask_value": -1,
  "pad_value": -2,


  "USE_CLS": False,
  "USE_CCE": False,
  "MVC": True,
  "USE_GENERATIVE_TRAINING": True,
  "world_size": 16,
  "distributed": True,
  "local_rank": 0,
  "gpu": 0
}

