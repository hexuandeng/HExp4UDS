"""Sequence-to-sequence metrics"""
from typing import Dict
import torch

class Precision():
    def __init__(self) -> None:
        super().__init__()
        self.tp = 0.0
        self.total = 0.0

    def __call__(self, pred, gold, mask) -> None:
        tp = (pred == gold).masked_fill(~mask.bool(), False)
        self.tp += tp.sum().item()
        self.total += mask.sum().item()

    def get_metric(self, reset: bool = False) -> Dict:
        if self.total == 0:
            metrics = {
                "precision": 0.0
            }
        else:
            metrics = {
                "precision": self.tp / self.total
            }
            
        if reset:
            self.reset()
        return metrics

    def reset(self) -> None:
        self.tp = 0.0
        self.total = 0.0
