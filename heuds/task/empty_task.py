import pickle
import random
from heuds.constant import register_task
from heuds.task.base_task import BasePytorchTask, Config
    
class EmptyTask(BasePytorchTask):
    _name = 'EmptyTask'
    def __init__(self, cfg, model):
        pass

register_task((EmptyTask, Config), 'EmptyTask')
