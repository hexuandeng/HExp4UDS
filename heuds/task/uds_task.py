import torch
from loguru import logger
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from dataclasses import dataclass, field, fields
from heuds.constant import register_task
from heuds.task.base_task import BasePytorchTask, TaskConfig, Config
from heuds.uds.data.uds_dataset import UDSDataset, UDSDatasetConfig

@dataclass
class UDSConfig(Config):
    def __post_init__(self):
        self.dataset = UDSDatasetConfig()

class UDSTask(BasePytorchTask):
    _name = 'UDSTask'
    def __init__(self, cfg, model):
        super().__init__(cfg)

        self.train_dataset = UDSDataset(cfg.dataset, 'train')
        self.dev_dataset = UDSDataset(cfg.dataset, 'dev', self.train_dataset)
        self.test_dataset = UDSDataset(cfg.dataset, 'test', self.train_dataset)

        self.model = model(cfg.model_name, cfg.model, self.train_dataset)
        optim_cfg = cfg.optimization
        self.init_optimizer()

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

    def init_optimizer(self, missing_keys=[]):
        if self.cfg.optimization.freeze_pretrained:
            for n, p in self.model.named_parameters():
                if 'bert_model' in n:
                    p.requires_grad = False

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        if 'Bert' in self.cfg.model_name:
            optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if n in missing_keys], 
                    'lr': self.cfg.optimization.missing_lr, 'weight_decay': self.cfg.optimization.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if 'bert_model' not in n and n not in missing_keys], 
                    'lr': self.cfg.optimization.lr, 'weight_decay': self.cfg.optimization.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if 'bert_model' in n and not any(nd in n for nd in no_decay) and n not in missing_keys], 
                    'lr': self.cfg.optimization.pretrained_lr, 'weight_decay': self.cfg.optimization.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if 'bert_model' in n and any(nd in n for nd in no_decay) and n not in missing_keys], 
                    'lr': self.cfg.optimization.pretrained_lr, 'weight_decay': 0}]
            self.optimizer = AdamW(optimizer_grouped_parameters, amsgrad=self.cfg.optimization.amsgrad)
        else:
            optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if n in missing_keys], 
                    'lr': self.cfg.optimization.missing_lr, 'weight_decay': self.cfg.optimization.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if n not in missing_keys], 
                    'lr': self.cfg.optimization.lr, 'weight_decay': self.cfg.optimization.weight_decay}]
            self.optimizer = AdamW(optimizer_grouped_parameters, amsgrad=self.cfg.optimization.amsgrad)

    def base_eval(self):
        metric = super().base_eval()
        try:
            self.thresholds = metric['Attr']['thresh']
            del metric['Attr']['thresh']
        except:
            self.thresholds = None

        return metric

    def base_test(self):
        # prepare data loader
        test_dataloader = self.dev_data_loader(self.test_dataset)

        logger.info("Start Base Test: ")
        logger.info("\tTotal examples Num = {}".format(
            len(self.test_dataset)))
        logger.info("\tBatch size = {}".format(self.cfg.iterator.dev_batch_size))

        log_loss = 0
        self.model.eval()
        self.model.reset_metric()
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                batch = self.set_batch_to_device(batch)
                loss = self.model.compute_loss(batch)
                log_loss += loss.item()
                if (step + 1) % self.cfg.log_interval_update == 0:
                    logger.info(f'Test step {step + 1}: loss {log_loss / float(self.cfg.log_interval_update)}')
                    log_loss = 0
                if self.cfg.clear_cache_interval > 0 and (step + 1) % self.cfg.clear_cache_interval == 0:
                    torch.cuda.empty_cache()

        metric = self.model.get_metric(self.thresholds)
        self.model.reset_metric()
        self.thresholds = None
        try:
            del metric['Attr']['thresh']
        except:
            pass
        logger.info(metric)

        return metric

register_task((UDSTask, UDSConfig), 'UDSTask')
