import torch
from loguru import logger
from dataclasses import dataclass
from heuds.constant import register_task
from heuds.task.base_task import BasePytorchTask, Config
from heuds.data.mono_dataset import MonoDataset, MonoDatasetConfig


@dataclass
class ConlluConfig(Config):
    def __post_init__(self):
        self.dataset = MonoDatasetConfig()

class ConlluTask(BasePytorchTask):
    _name = 'ConlluTask'
    def __init__(self, cfg, model):
        super().__init__(cfg)

        self.train_dataset = MonoDataset(cfg.dataset)
        self.thresholds = None
        self.model = model(cfg.model_name, cfg.model, self.train_dataset)
        self.model.to(self.device)

        total = sum([param.nelement() for param in self.model.parameters()])
        logger.info("Number of parameter: %.2fM" % (total / 1e6))

    def base_generate(self):
        # generating conllu (ud syntax info.) for given monolingual data
        assert self.model is not None
        logger.info("Start Base Generating")
        logger.info("\tTotal examples Num = {}".format(len(self.train_dataset)))
        logger.info("\tBatch size = {}".format(self.cfg.iterator.batch_size))

        self.resume_checkpoint(cpt_file_name='checkpoint_best.pt', strict=False)
        train_dataloader = self.data_loader(self.train_dataset)

        for step, batch in enumerate(train_dataloader):
            self.model.eval()
            batch = self.set_batch_to_device(batch)
            with torch.no_grad():
                self.model.get_conllu(batch)
            if (step + 1) % 50 == 0:
                logger.info(f"Step {step + 1} Finished!")

register_task((ConlluTask, ConlluConfig), 'ConlluTask')
