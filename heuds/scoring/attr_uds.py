"""Sequence-to-sequence metrics"""
from typing import Dict, List
import math
from scipy.stats import pearsonr
import numpy as np
import logging
import torch
from heuds.constant import NODE_ONTOLOGY, EDGE_ONTOLOGY
import pickle
import math

class DecompAttrMetrics(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.node_pred_attr = [None for _ in NODE_ONTOLOGY]
        self.node_true_attr = [None for _ in NODE_ONTOLOGY]
        self.edge_pred_attr = [None for _ in EDGE_ONTOLOGY]
        self.edge_true_attr = [None for _ in EDGE_ONTOLOGY]

    def __call__(self,
                 pred_attr: torch.Tensor,
                 pred_mask: torch.Tensor,
                 true_attr: torch.Tensor,
                 true_mask: torch.Tensor,
                 node_or_edge: str
                 ) -> None:
        # Attributes
        pred_mask = torch.gt(pred_mask, 0)
        true_mask = torch.gt(true_mask, 0)

        # for train time pearson, only look where attributes predicted
        if node_or_edge == "node":
            for i in range(true_mask.shape[-1]):
                pred = pred_attr[:, :, i][true_mask[:, :, i] == 1]
                true = true_attr[:, :, i][true_mask[:, :, i] == 1]
                if self.node_pred_attr[i] is None or self.node_true_attr[i] is None:
                    self.node_pred_attr[i] = pred
                    self.node_true_attr[i] = true
                else:
                    self.node_pred_attr[i] = torch.cat([self.node_pred_attr[i], pred])
                    self.node_true_attr[i] = torch.cat([self.node_true_attr[i], true])
        elif node_or_edge == "edge":
            for i in range(true_mask.shape[-1]):
                pred = pred_attr[:, i][true_mask[:, i] == 1]
                true = true_attr[:, i][true_mask[:, i] == 1]
                if self.edge_pred_attr[i] is None or self.edge_true_attr[i] is None:
                    self.edge_pred_attr[i] = pred
                    self.edge_true_attr[i] = true
                else:
                    self.edge_pred_attr[i] = torch.cat([self.edge_pred_attr[i], pred])
                    self.edge_true_attr[i] = torch.cat([self.edge_true_attr[i], true])

    def get_metric(self, reset: bool = False, thresholds = None) -> Dict:
        pred_attr = self.node_pred_attr + self.edge_pred_attr
        true_attr = self.node_true_attr + self.edge_true_attr

        if None in true_attr:
            if reset:
                self.reset()
            return {
                "pearson_r": 0,
                "pearson_F1": 0,
                "thresh": None
            }

        def pearson(pred_attr, true_attr):
            try:
                pearson_r, _ = pearsonr(pred_attr, true_attr)
            except ValueError:
                pearson_r = 0.0
            return pearson_r

        def f1score(pred_attr, true_attr, thresh=None):
            flat_true_threshed = torch.gt(true_attr, 0)
            if thresh is None:
                scores = []
                for thresh in np.linspace(-3, 3, 600):
                    flat_pred_threshed = torch.gt(pred_attr, float(thresh))
                    tp = torch.sum(flat_pred_threshed * flat_true_threshed).item()
                    fp = torch.sum(flat_pred_threshed * ~flat_true_threshed).item()
                    fn = torch.sum(~flat_pred_threshed * flat_true_threshed).item()
                    try:
                        p = tp / (tp + fp)
                        r = tp / (tp + fn)
                        f1 = 2 * p * r / (p + r)
                    except:
                        f1 = 0
                    scores.append((f1, thresh))
                return max(scores)
            else:
                flat_pred_threshed = torch.gt(pred_attr, float(thresh))
                tp = torch.sum(flat_pred_threshed * flat_true_threshed).item()
                fp = torch.sum(flat_pred_threshed * ~flat_true_threshed).item()
                fn = torch.sum(~flat_pred_threshed * flat_true_threshed).item()
                try:
                    p = tp / (tp + fp)
                    r = tp / (tp + fn)
                    f1 = 2 * p * r / (p + r)
                except:
                    f1 = 0
                return (f1, thresh)

        pearson_r = []
        pearson_f1 = []
        if thresholds is not None:
            for pred, true, threshold in zip(pred_attr, true_attr, thresholds):
                pearson_f1.append(f1score(pred, true, threshold))
                pred = pred.cpu().detach().numpy()
                true = true.cpu().detach().numpy()
                p = pearson(pred, true)
                if math.isnan(p):
                    p = 0
                pearson_r.append(p)
        else:
            for pred, true in zip(pred_attr, true_attr):
                pearson_f1.append(f1score(pred, true))
                pred = pred.cpu().detach().numpy()
                true = true.cpu().detach().numpy()
                pearson_r.append(pearson(pred, true))

        thresholds = [i[1] for i in pearson_f1]
        pearson_f1 = [i[0] for i in pearson_f1]

        metrics = {
            # "node_pearson_r": pearson_r,
            # "node_pearson_F1": pearson_f1,
            "pearson_r": sum(pearson_r) / len(pearson_r),
            "pearson_F1": sum(pearson_f1) / len(pearson_f1),
            "thresh": thresholds
        }
        if reset:
            self.reset()
        return metrics

    def reset(self) -> None:
        self.node_pred_attr = [None for _ in NODE_ONTOLOGY]
        self.node_true_attr = [None for _ in NODE_ONTOLOGY]
        self.edge_pred_attr = [None for _ in EDGE_ONTOLOGY]
        self.edge_true_attr = [None for _ in EDGE_ONTOLOGY]
