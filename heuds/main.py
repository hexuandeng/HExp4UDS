# -*- coding: utf-8 -*-

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.getcwd())

# import for register of task and model
from heuds.base.parser import PytorchParser
from heuds.constant import MODEL_REGISTRY, TASK_REGISTRY
from heuds.task import uds_task, conllu_task, udistill_task, predpatt_task, empty_task
from heuds.uds import cascade_model, syntactic_model

if __name__ == "__main__":
    cfg = PytorchParser()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    model = MODEL_REGISTRY[cfg.model_name][0]
    task = TASK_REGISTRY[cfg.task_name][0](cfg, model)
    if cfg.mission == "train":
        task.base_train()
    elif cfg.mission == "test":
        task.test_model()
    elif cfg.mission == "generate":
        task.base_generate()
    elif cfg.mission == "preprocess":
        task.base_preprocess()
    else:
        raise NotImplementedError("Unsupport mission received!")
