import os
import random
import pickle
import torch
from torch.cuda import amp
from torch.optim import AdamW
from loguru import logger
from dataclasses import dataclass
from transformers import get_cosine_schedule_with_warmup
from heuds.constant import register_task
from heuds.task.base_task import BasePytorchTask, Config
from heuds.data.mono_dataset import MonoDataset, MonoDatasetConfig


@dataclass
class UDistillConfig(Config):
    def __post_init__(self):
        self.dataset = MonoDatasetConfig()

class UDistillTask(BasePytorchTask):
    _name = 'UDistillTask'

    def __init__(self, cfg, model): 
        super().__init__(cfg)

        self.train_dataset = MonoDataset(cfg.dataset)
        self.model = model(cfg.model_name, cfg.model, self.train_dataset)
        optim_cfg = cfg.optimization

        if cfg.optimization.freeze_pretrained:
            for n, p in self.model.named_parameters():
                if 'bert_model' in n:
                    p.requires_grad = False
        if 'Bert' in cfg.model_name:
            optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if 'bert_model' not in n], 
                'lr': cfg.optimization.lr, 'weight_decay': cfg.optimization.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if 'bert_model' in n], 'lr': cfg.optimization.pretrained_lr, 'weight_decay': 0}]
            self.optimizer = AdamW(optimizer_grouped_parameters, amsgrad=cfg.optimization.amsgrad)
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=cfg.optimization.lr, 
                amsgrad=cfg.optimization.amsgrad, weight_decay=cfg.optimization.weight_decay)

        self.update_per_epoch = self.data_loader.get_batch_num(self.train_dataset)
        self.updates = cfg.max_epoch * self.update_per_epoch
        if optim_cfg.use_lr_scheduler:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=optim_cfg.num_warmup_steps,
                num_training_steps=self.updates
            )
        self.use_lr_scheduler = optim_cfg.use_lr_scheduler

        self.model.to(self.device)
        total = sum([param.nelement() for param in self.model.parameters()])
        logger.info("Number of parameter: %.2fM" % (total / 1e6))

        self.thresholds = None

    def base_generate(self):
        # Generate pseudo target for monolingual data
        assert self.model is not None
        batch_size = self.cfg.iterator.batch_size
        logger.info("Start Base Generating")
        logger.info("\tTotal examples Num = {}".format(
            len(self.train_dataset)))
        logger.info("\tBatch size = {}".format(batch_size))

        self.resume_checkpoint(cpt_file_name='checkpoint_best.pt', strict=False, model_only=True)
        train_dataloader = self.data_loader(self.train_dataset)

        distilled_dataset = []
        for step, batch in enumerate(train_dataloader):
            self.model.eval()
            batch = self.set_batch_to_device(batch)
            with torch.no_grad():
                distilled_dataset.append(self.model.distill(batch))
            if (step + 1) % 50 == 0:
                logger.info(f"Step {step + 1} Finished!")

        to_select = 1e5 / self.cfg.iterator.batch_size
        distilled_dataset = random.sample(distilled_dataset, int(to_select))
        pickle_file = self.cfg.dataset.mono_file + ".pickle"
        with open(pickle_file, 'wb') as f:
            pickle.dump(distilled_dataset, f)

    def base_train(self):
        assert self.model is not None
        del self.train_dataset.vectors
        logger.info("Start Distilled Training")

        if not os.path.exists(self.cfg.checkpoint.save_dir):
            os.makedirs(self.cfg.checkpoint.save_dir)

        global_step = 0
        pickle_file = self.cfg.dataset.mono_file + ".pickle"
        with open(pickle_file, 'rb') as f:
            train_dataloaders = pickle.load(f)
        train_dataloader = self.set_batch_to_device(train_dataloaders, 'cpu')
        del train_dataloaders

        for epoch_idx in range(int(self.cfg.max_epoch)):
            logger.info(f"Begin Training Epoch {epoch_idx}!")
            logger.info(f"Begin Preprocessing Batch!")
            random.shuffle(train_dataloader)
            total_loss = 0
            log_loss = 0
            for step, batch in enumerate(train_dataloader):
                try:
                    self.model.train()
                    batch = self.set_batch_to_device(batch)
                    if self.cfg.fp16:
                        with amp.autocast():
                            loss = self.model.compute_soft_loss(batch)
                    else:
                        loss = self.model.compute_soft_loss(batch)
                    if self.cfg.update_freq > 1:
                        loss = loss / self.cfg.update_freq
                    
                    if torch.isnan(loss):
                        logger.warning(f"NAN occurs in step {step} before normalization!")
                        continue
                    total_loss += loss.item()
                    log_loss += loss.item()

                    if (step + 1) % self.cfg.log_interval_update == 0:
                        logger.info(f'epoch {epoch_idx} step {step + 1}: loss {log_loss / float(self.cfg.log_interval_update)}')
                        log_loss = 0

                    # backward
                    if self.cfg.fp16:
                        self.scaler.scale(loss).backward()
                        if (step + 1) % self.cfg.update_freq == 0:
                            if self.cfg.clip_norm > 0:
                                batch_grad_norm = self.rescale_gradients()
                                if torch.isnan(batch_grad_norm):
                                    logger.warning(f"NAN occurs in step {step} after normalization!")
                                    continue
                                # torch.nn.utils.clip_grad_norm_(
                                #     self.model.parameters(), self.cfg.clip_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.model.zero_grad()
                            global_step += 1
                    else:
                        loss.backward()
                        if (step + 1) % self.cfg.update_freq == 0:
                            if self.cfg.clip_norm > 0:
                                batch_grad_norm = self.rescale_gradients()
                                # torch.nn.utils.clip_grad_norm_(
                                #     self.model.parameters(), self.cfg.clip_norm)
                            self.optimizer.step()
                            self.model.zero_grad()
                        global_step += 1
                        
                    if self.use_lr_scheduler:
                        self.lr_scheduler.step()
                    if self.cfg.clear_cache_interval > 0 and (step + 1) % self.cfg.clear_cache_interval == 0:
                        torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"Out of memory in step {step}. Skipping this step.")
                        torch.cuda.empty_cache() # 清空显存占用（可选，但有助于防止未来的显存不足）
                    else: # 如果是其他类型的 RuntimeError，重新引发异常
                        raise e

            total_loss = total_loss / self.update_per_epoch
            if self.cfg.checkpoint.save_interval > 0 and (epoch_idx + 1) % self.cfg.checkpoint.save_interval == 0:
                self.save_checkpoint('checkpoint_last.pt', epoch_idx + 1, 0, self.metric)

        self.save_checkpoint('checkpoint_last.pt', epoch_idx + 1, 0, self.metric)

register_task((UDistillTask, UDistillConfig), 'UDistillTask')
