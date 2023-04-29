import os
from torch.cuda import amp
from torch.optim import AdamW
from loguru import logger
from dataclasses import dataclass
from transformers import get_cosine_schedule_with_warmup
from heuds.constant import register_task
from heuds.task.base_task import BasePytorchTask, Config
from heuds.data.predpatt_dataset import PredPattDataset, PredPattDatasetConfig

@dataclass
class PredPattConfig(Config):
    def __post_init__(self):
        self.dataset = PredPattDatasetConfig()

class PredPattTask(BasePytorchTask):
    _name = 'PredPattTask'
    
    def __init__(self, cfg, model):
        cfg.validate_interval = -1  # disable validation
        super().__init__(cfg)

        self.train_dataset = PredPattDataset(cfg.dataset)
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

register_task((PredPattTask, PredPattConfig), 'PredPattTask')
