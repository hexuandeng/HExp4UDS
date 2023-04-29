"""Sequence-to-sequence metrics"""
from typing import Dict, List
import logging
import torch
from torch import nn

class F1Metrics():
    def __init__(self) -> None:
        self.match_num = 0.
        self.test_num = 0.
        self.gold_num = 0.

    def __call__(self, golds, tests) -> None:
        assert len(golds) == len(tests)
        for gold, test in zip(golds, tests):
            TP = 0
            for i in range(min(len(gold), len(test))):
                if gold[i] == test[i]:
                    TP += 1
            self.match_num += TP
            self.test_num += len(test)
            self.gold_num += len(gold)

    def update(self, match_num, test_num, gold_num):
        self.match_num += match_num
        self.test_num += test_num
        self.gold_num += gold_num

    def get_metric(self, reset: bool = False) -> Dict:
        match_num = self.match_num
        test_num = self.test_num
        gold_num = self.gold_num

        if test_num == 0 or gold_num == 0:
            if reset:
                self.reset()
            return {
            "precision": 0.0,
            "recall": 0.0,
            "f_score": 0.0
        }
        precision = float(match_num) / float(test_num)
        recall = float(match_num) / float(gold_num)
        if (precision + recall) != 0:
            f_score = 2 * precision * recall / (precision + recall)
        else:
            f_score = 0.0

        if reset:
            self.reset()

        return {
            "precision": precision,
            "recall": recall,
            "f_score": f_score
        }

    def reset(self) -> None:
        self.match_num = 0.0
        self.test_num = 0.0
        self.gold_num = 0.0
