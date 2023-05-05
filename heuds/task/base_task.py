import os
import random
import torch
import numpy as np
from loguru import logger
from torch.cuda import amp
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import List, Optional, Any
from heuds.base.base_dataset import DatasetConfig
from heuds.base.base_iterator import BaseIterator, IteratorConfig
from heuds.base.base_config import BaseConfig, CheckpointConfig, OptimizationConfig, GenerationConfig

@dataclass
class TaskConfig(BaseConfig):
    # This is the core dataclass including common parameters shared by all different jobs. Please append your params to other dataclasses if they were
    # used for a particular purpose or task, such as those dedicated for `distributed training`, `optimization`, etc.
    no_progress_bar: bool = field(
        default=False, metadata={"help": "disable progress bar"}
    )
    log_interval_update: int = field(
        default=20,
        metadata={
            "help": "log progress every N updates (when progress bar is disabled)"
        },
    )
    log_file: Optional[str] = field(
        default=None, metadata={"help": "log file to copy metrics to."}
    )
    seed: int = field(
        default=3407, metadata={"help": "pseudo random number generator seed"}
    )
    cpu: bool = field(default=False, metadata={
                      "help": "use CPU instead of CUDA"})
    gpu: int = field(default=0, metadata={
                      "help": "which GPU to use"})
    fp16: bool = field(default=False, metadata={"help": "use FP16"})
    max_epoch: int = field(
        default=300, metadata={"help": "force stop training at specified epoch"}
    )
    max_update: int = field(
        default=0, metadata={"help": "force stop training at specified update"}
    )
    stop_time_hours: float = field(
        default=0,
        metadata={
            "help": "force stop training after specified cumulative time (if >0)"
        },
    )
    update_freq: int = field(
        default=1,
        metadata={"help": "update parameters every N_i batches, when in epoch i"},
    )
    clip_norm: float = field(
        default=5.0, metadata={"help": "clip threshold of gradients"}
    )
    validate_interval: int = field(
        default=2, metadata={"help": "validate every N epochs"}
    )
    validate_interval_updates: int = field(
        default=0, metadata={"help": "validate every N updates"}
    )
    disable_validation: bool = field(
        default=False, metadata={"help": "disable validation"}
    )
    update_ordered_indices_seed: bool = field(
        default=False,
        metadata={
            "help": "if true then increment seed with epoch for getting batch iterators, defautls to False.",
        },
    )
    clear_cache_interval: int = field(
        default=-1, metadata={ "help": "if true then."}
    )

@dataclass
class Config(BaseConfig):
    task: Any = TaskConfig()
    dataset: DatasetConfig = DatasetConfig()
    iterator: IteratorConfig = IteratorConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    generation: GenerationConfig = GenerationConfig()
    criterion: Any = None
    optimizer: Any = None
    lr_scheduler: Any = None
    scoring: Any = None
    tokenizer: Any = None

def set_optimizer_params_grad(
    named_params_optimizer, named_params_model, test_nan=False
):
    """
    Utility function for optimize_on_cpu and 16-bits training.
    Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(
        named_params_optimizer, named_params_model
    ):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(
                name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(
                    param_opti.data.new().resize_(*param_opti.data.size())
                )
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """
    Utility function for optimize_on_cpu and 16-bits training.
    Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(
        named_params_optimizer, named_params_model
    ):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(
                name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


class BasePytorchTask(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self._check_setting_validity()
        self._init_device()
        self.reset_random_seed()

        # ==> task-specific initialization
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

        self.data_loader = BaseIterator(cfg.iterator)
        self.dev_data_loader = BaseIterator(cfg.iterator, train=False)
        self.metric = float('-inf')

        if self.cfg.fp16:
            self.scaler = amp.GradScaler()

        if not os.path.isabs(self.cfg.checkpoint.save_dir):
            self.cfg.checkpoint.save_dir = os.path.join("checkpoints/", self.cfg.checkpoint.save_dir)
        if self.cfg.checkpoint.pretrained_model_dir is not None and \
            not os.path.isabs(self.cfg.checkpoint.pretrained_model_dir):
            self.cfg.checkpoint.pretrained_model_dir = os.path.join("checkpoints/", self.cfg.checkpoint.pretrained_model_dir)

    def _check_setting_validity(self):
        pass

    def _init_device(self):
        if torch.cuda.is_available() and not self.cfg.cpu:
            logger.info(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")
            for i in range(torch.cuda.device_count()):
                info = torch.cuda.get_device_properties(i)
                logger.info(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            logger.info(f"CUDA Not Available! Using CPU Instead!")
        self.n_gpu = torch.cuda.device_count()

    def reset_random_seed(self, seed=None):
        if seed is None:
            seed = self.cfg.seed
        logger.info("Reset Random Seed to {}".format(seed))
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 1:
            torch.cuda.manual_seed_all(seed)

    def set_batch_to_device(self, batch, device=None):
        # move mini-batch data to the proper device
        if device is None:
            device = self.device
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)
            return batch
        elif isinstance(batch, dict):
            for key, value in batch.items():
                batch[key] = self.set_batch_to_device(value)
            return batch
        elif isinstance(batch, list):
            return batch
        elif isinstance(batch, Sequence):
            new_batch = []
            for value in batch:
                new_batch.append(self.set_batch_to_device(value))
            return new_batch
        else:
            raise Exception("Unsupported batch type {}".format(type(batch)))

    def base_train(self):
        assert self.model is not None
        self.update_per_epoch = self.data_loader.get_batch_num(self.train_dataset)

        logger.info("Start Base Training")
        logger.info("\tTotal examples Num = {}".format(len(self.train_dataset)))
        logger.info("\tBatch size = {}".format(self.cfg.iterator.batch_size))
        logger.info("\tNum epochs = {}".format(self.cfg.max_epoch))

        if not os.path.exists(self.cfg.checkpoint.save_dir):
            os.makedirs(self.cfg.checkpoint.save_dir)
        
        if self.cfg.checkpoint.pretrained_model_dir is not None:
            missing_keys, _ = self.resume_model(cpt_file_name='checkpoint_last.pt')
            self.init_optimizer(missing_keys)
            checkpoint_epoch = 0
            checkpoint_step = 0
            global_step = 0
        checkpoint_epoch, checkpoint_step, _, _ = self.resume_checkpoint(
            cpt_file_name='checkpoint_last.pt', strict=False)
        global_step = checkpoint_epoch * self.update_per_epoch + checkpoint_step + 2

        for epoch_idx in range(checkpoint_epoch, int(self.cfg.max_epoch)):
            logger.info(f"Begin Training Epoch {epoch_idx}!")
            logger.info(f"Begin Preprocessing Batch!")
            train_dataloader = self.data_loader(self.train_dataset)
            total_loss = 0
            log_loss = 0
            for step, batch in enumerate(train_dataloader):
                try:
                    self.model.train()
                    if step < checkpoint_step:
                        continue
                    batch = self.set_batch_to_device(batch)
                    if self.cfg.fp16:
                        with amp.autocast():
                            loss = self.model.compute_loss(batch)
                    else:
                        loss = self.model.compute_loss(batch)
                    if self.cfg.update_freq > 1:
                        loss = loss / self.cfg.update_freq

                    if torch.isnan(loss):
                        logger.warning(f"NAN occurs in step {step} before normalization!")
                        self.model.zero_grad()
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
                                    self.model.zero_grad()
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
                                if torch.isnan(batch_grad_norm):
                                    logger.warning(f"NAN occurs in step {step} after normalization!")
                                    self.model.zero_grad()
                                    continue
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
                        torch.cuda.empty_cache()
                    else:
                        raise e
            
            metric = self.model.get_metric()
            logger.info(metric)
            self.model.reset_metric()
            
            total_loss = total_loss / self.update_per_epoch
            # cfg.validate_interval <= 0 means no validation needed
            if self.cfg.validate_interval > 0 and (epoch_idx + 1) % self.cfg.validate_interval == 0:
                metric = self.base_eval()['Metric']
            if not self.cfg.checkpoint.no_save:
                if self.cfg.validate_interval > 0 and (epoch_idx + 1) % self.cfg.validate_interval == 0 and self.metric < metric:
                    logger.info(f'New best in Dev: {metric}!')
                    self.metric = metric
                    self.save_checkpoint('checkpoint_best.pt', epoch_idx, step, self.metric)
                if self.cfg.checkpoint.save_interval > 0 and (epoch_idx + 1) % self.cfg.checkpoint.save_interval == 0 and self.cfg.checkpoint.keep_epoch_checkpoints:
                    if self.cfg.validate_interval <= 0:
                        self.save_checkpoint(f'{epoch_idx}_epoch_{total_loss}_loss.pt', epoch_idx, step, 0)
                    else:
                        self.save_checkpoint(f'{epoch_idx}_epoch_{metric}_metric.pt', epoch_idx, step, self.metric)
                self.save_checkpoint('checkpoint_last.pt', epoch_idx + 1, 0, self.metric)
            checkpoint_step = 0

        if self.cfg.validate_interval > 0:
            checkpoint_epoch, checkpoint_step, _, _ = self.resume_checkpoint(cpt_file_name='checkpoint_best.pt', strict=False)
            self.base_eval()
            self.base_test()

    def base_eval(self):
        # prepare data loader
        dev_dataloader = self.dev_data_loader(self.dev_dataset)

        logger.info("Start Base Dev: ")
        logger.info("\tTotal examples Num = {}".format(
            len(self.dev_dataset)))
        logger.info("\tBatch size = {}".format(self.cfg.iterator.dev_batch_size))
        
        log_loss = 0
        self.model.eval()
        self.model.reset_metric()
        with torch.no_grad():
            for step, batch in enumerate(dev_dataloader):
                batch = self.set_batch_to_device(batch)
                loss = self.model.compute_loss(batch)
                log_loss += loss.item()
                if (step + 1) % self.cfg.log_interval_update == 0:
                    logger.info(f'Dev step {step + 1}: loss {log_loss / float(self.cfg.log_interval_update)}')
                    log_loss = 0
                if self.cfg.clear_cache_interval > 0 and (step + 1) % self.cfg.clear_cache_interval == 0:
                    torch.cuda.empty_cache()

        metric = self.model.get_metric()
        self.model.reset_metric()
        logger.info(metric)
        
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

        metric = self.model.get_metric()
        self.model.reset_metric()
        logger.info(metric)

        return metric

    def test_model(self):
        assert self.model is not None
        self.resume_checkpoint(cpt_file_name='checkpoint_best.pt', strict=False, model_only=True)
        self.base_eval()
        self.base_test()

    def save_checkpoint(self, cpt_file_name, epoch, step, metric):
        cpt_file_path = os.path.join(self.cfg.checkpoint.save_dir, cpt_file_name)
        logger.info(f"Dump checkpoint into {cpt_file_path}")
        if not os.path.exists(self.cfg.checkpoint.save_dir):
            os.makedirs(self.cfg.checkpoint.save_dir)
        store_dict = {}

        if self.model:
            if isinstance(self.model, torch.nn.parallel.DataParallel) or isinstance(
                self.model, torch.nn.parallel.DistributedDataParallel
            ):
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()
            store_dict["model_state"] = model_state
        else:
            logger.warning("No model state is dumped")

        if self.optimizer:
            store_dict["optimizer_state"] = self.optimizer.state_dict()
        else:
            logger.warning("No optimizer state is dumped")

        if self.lr_scheduler:
            store_dict["lr_scheduler"] = self.lr_scheduler.state_dict()

        store_dict["epoch"] = epoch
        store_dict["step"] = step
        store_dict["metric"] = metric

        torch.save(store_dict, cpt_file_path)

    def resume_checkpoint(
        self,
        cpt_file_path=None,
        cpt_file_name=None,
        strict=False,
        model_only=False
    ):
        # decide cpt_file_path to resume
        if cpt_file_path is None:  # use provided path with highest priority
            cpt_file_path = os.path.join(self.cfg.checkpoint.save_dir, cpt_file_name)
        elif cpt_file_name is not None:  # error when path and name are both provided
            raise Exception(
                "Confused about path {} or file name {} to resume".format(
                    cpt_file_path, cpt_file_name
                )
            )

        if os.path.exists(cpt_file_path):
            logger.info("Resume checkpoint from {}".format(cpt_file_path))
        elif strict:
            raise Exception(
                "Checkpoint does not exist, {}".format(cpt_file_path))
        else:
            logger.warning(
                "Checkpoint does not exist, {}".format(cpt_file_path))
            return 0, 0, 0, 0

        store_dict = torch.load(cpt_file_path, map_location=self.device)

        if self.model and "model_state" in store_dict:
            if isinstance(self.model, torch.nn.parallel.DataParallel) or isinstance(
                self.model, torch.nn.parallel.DistributedDataParallel
            ):
                missing_keys, unexpected_keys = self.model.module.load_state_dict(store_dict["model_state"], strict=False)
            else:
                missing_keys, unexpected_keys = self.model.load_state_dict(store_dict["model_state"], strict=False)
            logger.info("Resume model successfully")
            if len(missing_keys):
                logger.warning(f"Missing Keys: {missing_keys}.")
            if len(unexpected_keys):
                logger.warning(f"Unexpected Keys: {unexpected_keys}.")
        elif strict:
            raise Exception(
                "Resume model failed, dict.keys = {}".format(store_dict.keys())
            )
        
        if not model_only:
            if self.optimizer and "optimizer_state" in store_dict:
                self.optimizer.load_state_dict(store_dict["optimizer_state"])
                logger.info("Resume optimizer successfully")
            elif strict:
                raise Exception(
                    "Resume optimizer failed, dict.keys = {}".format(
                        store_dict.keys())
                )

            if self.lr_scheduler and "lr_scheduler" in store_dict:
                self.lr_scheduler.load_state_dict(store_dict["lr_scheduler"])
                logger.info("Resume lr_scheduler successfully")
            elif strict:
                raise Exception(
                    "Resume lr_scheduler failed, dict.keys = {}".format(
                        store_dict.keys())
                )
            self.metric = store_dict["metric"]

        return store_dict["epoch"], store_dict["step"], missing_keys, unexpected_keys

    def resume_model(
        self,
        cpt_file_path=None,
        cpt_file_name=None,
        strict=False
    ):
        # decide cpt_file_path to resume
        if cpt_file_path is None:  # use provided path with highest priority
            cpt_file_path = os.path.join(
                self.cfg.checkpoint.pretrained_model_dir, cpt_file_name)
        elif cpt_file_name is not None:  # error when path and name are both provided
            raise Exception(
                "Confused about path {} or file name {} to resume".format(
                    cpt_file_path, cpt_file_name
                )
            )

        if os.path.exists(cpt_file_path):
            logger.info("Resume checkpoint from {}".format(cpt_file_path))
        elif strict:
            raise Exception(
                "Checkpoint does not exist, {}".format(cpt_file_path))
        else:
            logger.warning(
                "Checkpoint does not exist, {}".format(cpt_file_path))
            return 0, 0

        store_dict = torch.load(cpt_file_path, map_location=self.device)

        if self.model and "model_state" in store_dict:
            if isinstance(self.model, torch.nn.parallel.DataParallel) or isinstance(
                self.model, torch.nn.parallel.DistributedDataParallel
            ):
                missing_keys, unexpected_keys = self.model.module.load_state_dict(store_dict["model_state"], strict=False)
            else:
                missing_keys, unexpected_keys = self.model.load_state_dict(store_dict["model_state"], strict=False)
            logger.info("Resume model successfully")
        elif strict:
            raise Exception(
                "Resume model failed, dict.keys = {}".format(store_dict.keys())
            )

        return missing_keys, unexpected_keys

    def rescale_gradients(self) -> Optional[float]:
        return rescale_gradients(self.model, self.cfg.clip_norm)

def average_gradients(model):
    """Gradient averaging."""
    size = float(torch.distributed.get_world_size())
    for name, param in model.named_parameters():
        try:
            torch.distributed.all_reduce(
                param.grad.data, op=torch.distributed.reduce_op.SUM)
            param.grad.data /= size
        except Exception as e:
            logger.error(
                "Error when all_reduce parameter {}, size={}, grad_type={}, error message {}".format(
                    name, param.size(), param.grad.data.dtype, repr(e)
                )
            )

def rescale_gradients(model, grad_norm: Optional[float] = None, norm_type=2) -> Optional[float]:
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Supports sparse gradients.

    Parameters
    ----------
    parameters : ``(Iterable[torch.Tensor])``
        An iterable of Tensors that will have gradients normalized.
    grad_norm : ``float``
        The max norm of the gradients.
    norm_type : ``float``
        The type of the used p-norm. Can be ``'inf'`` for infinity norm.

    Returns
    -------
    Total norm of the parameters (viewed as a single vector).
    """
    if grad_norm:
        # pylint: disable=invalid-name,protected-access
        parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad and param.grad is not None]
        max_norm = float(grad_norm)
        norm_type = float(norm_type)
        if norm_type == float('inf'):
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            total_norm = 0
            for name, p in parameters:
                if p.grad.is_sparse:
                    # need to coalesce the repeated indices before finding norm
                    grad = p.grad.data.coalesce()
                    param_norm = grad._values().norm(norm_type)
                else:
                    param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for _, p in parameters:
                if p.grad.is_sparse:
                    p.grad.data._values().mul_(clip_coef)
                else:
                    p.grad.data.mul_(clip_coef)
        return total_norm
    return None

def init_optimizer(self, missing_keys=[]):
    pass
