from pathlib import Path
import json
import sys
import time
from datetime import datetime
from copy import deepcopy

import warnings
import wandb
import torch
from torch import nn

sys.path.insert(0, "../")
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)



class scGPT_Checkpoint:

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


# def define_wandb_metrcis():
#     wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
#     wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
#     wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
#     wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
#     wandb.define_metric("test/avg_bio", summary="max")


class scGPT_Trainer:
    
    def __init__(
        self,
        config,
        model,
        device,
        dataset,
        datapreprocessor,
        criterion,
        optimizer,
        scheduler,
        scaler,
        ):
        self.config = config
        self.model = model
        # self.train_loader = train_loader
        # self.valid_loader = valid_loader
        self.dataset = dataset
        self.datapreprocessor = datapreprocessor
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        
        self.epochs = self.config.model_configs["epochs"]
        self.save_period = 10
        self.start_epoch = 1
        self.save_eval_interval = 5
        
        run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self.save_dir = Path(f"./save/model/{run_id}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    
    def train(self):
        best_val_loss = float("inf")
        best_avg_bio = 0.0
        best_model = None
        # define_wandb_metrcis()

        # TODO: add to config
        batch_size = 16  # config.model_configs["batch_size"]
        per_seq_batch_sample = True
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start_time = time.time()
            
            train_data_pt, valid_data_pt = self.datapreprocessor.prepare_data(
                self.dataset.adata, 
                sort_seq_batch=per_seq_batch_sample
                )
            
            train_loader = self.datapreprocessor.prepare_dataloader(
                train_data_pt,
                batch_size=batch_size,
                shuffle=False,
                intra_domain_shuffle=True,
                drop_last=False,
            )
            valid_loader = self.datapreprocessor.prepare_dataloader(
                valid_data_pt,
                batch_size=batch_size,
                shuffle=False,
                intra_domain_shuffle=False,
                drop_last=False,
            )            
            
            result_train = self._train_epoch(epoch, train_loader)
            result_valid = self._valid_epoch(epoch, valid_loader)
            
            val_loss = result_valid["valid_mse"]
            val_mre = result_valid["valid_mre"]
            
            elapsed = time.time() - epoch_start_time
            
            print("-"*100)
            print(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
            )
            print("-"*100)
            

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = deepcopy(self.model)
                best_model_epoch = epoch
                print(f"Best model with score {best_val_loss:5.4f}")

            if epoch % self.save_eval_interval == 0 or epoch == self.epochs:
                print(f"Saving model to {self.save_dir}")
                torch.save(
                    best_model.state_dict(), 
                    self.save_dir / f"model_{best_model_epoch}.pt")
                
            self.scheduler.step()
        
        
    def _train_epoch(self, epoch, train_loader):
        # model, 
        # loader,
        # criterion,
        # scheduler,
        # scaler,
        # optimizer,
        # DSBN,
        # vocab
        
        self.model.train()
        
        total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
        total_error = 0.0
        log_interval = 100
        vocab = self.config.vocab
        pad_token = self.config.pad_token
        
        amp = True
        GEPC = False
        ecs_thres = 0.8
        DSBN = False
        mask_value = -1
        explicit_zero_prob = True
        
        # load_model = None
        # fast_transformer = False
        # GEPC = False 
        # do_dab = False
        # use_batch_labels = False
        # explicit_zero_prob = True
        # ecs_thres = 0.8        
        
        start_time = time.time()


        num_batches = len(train_loader)
        
        metrics_to_log = {}
        for batch, batch_data in enumerate(train_loader):
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            input_values = batch_data["values"].to(self.device)
            target_values = batch_data["target_values"].to(self.device)
            batch_labels = batch_data["batch_labels"].to(self.device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                    MVC=GEPC,
                    ECS=ecs_thres > 0,
                )

                masked_positions = input_values.eq(mask_value)  # the postions to predict
                loss = loss_mse = self.criterion(
                    output_dict["mlm_output"], 
                    target_values, 
                    masked_positions
                )
                metrics_to_log["train/mse"] = loss_mse.item()
                if explicit_zero_prob:
                    loss_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_zero_log_prob
                    metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                    

            self.model.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
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

            with torch.no_grad():
                mre = masked_relative_error(
                    output_dict["mlm_output"], target_values, masked_positions
                )

            total_loss += loss.item()
            total_mse += loss_mse.item()
            # total_gepc += loss_gepc.item() if config.GEPC else 0.0
            total_error += mre.item()
            if batch % log_interval == 0 and batch > 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_gepc = total_gepc / log_interval if GEPC else 0.0
                cur_error = total_error / log_interval

                print(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    + (f"gepc {cur_gepc:5.2f} |" if GEPC else "")
                )

                total_loss = 0
                total_mse = 0
                total_gepc = 0
                total_error = 0
                start_time = time.time()

        return metrics_to_log
        
        
    def _valid_epoch(
        self,
        epoch,
        valid_loader
        ):
        """
        Evaluate the model on the evaluation data.
        """
        # model, 
        # loader, 
        # criterion, 
        # vocab,
        # DSBN
        
        vocab = self.config.vocab
        pad_token = self.config.pad_token
        
        amp = True
        DSBN = False
        mask_value = -1
        
        total_loss = 0.0
        total_error = 0.0
        total_dab = 0.0
        total_num = 0        
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(valid_loader):
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                target_values = batch_data["target_values"].to(self.device)
                batch_labels = batch_data["batch_labels"].to(self.device)

                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                with torch.cuda.amp.autocast(enabled=amp):
                    output_dict = self.model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if DSBN else None,
                    )
                    output_values = output_dict["mlm_output"]

                    masked_positions = input_values.eq(mask_value)
                    loss = self.criterion(output_values, target_values, masked_positions)

                total_loss += loss.item() * len(input_gene_ids)
                total_error += masked_relative_error(
                    output_values, target_values, masked_positions
                ).item() * len(input_gene_ids)
                total_num += len(input_gene_ids)

        valid_log = {
                "valid_mse": total_loss / total_num,
                "valid_mre": total_error / total_num,
                "epoch": epoch,
            }
        
        return valid_log
        # return total_loss / total_num, total_error / total_num



    