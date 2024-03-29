import torch

class LabelSmoothing(torch.nn.Module):
    """Implement label smoothing."""

    def __init__(self,
                 pad_index: int = 0,
                 smoothing: float = 0.0) -> None:
        super().__init__()
        self._criterion = torch.nn.KLDivLoss(reduction="sum")
        self._pad_index = pad_index
        self._smoothing = smoothing
        self._confidence = 1.0 - smoothing

    def reset_parameters(self,
                         pad_index: int = None,
                         smoothing: float = None) -> None:
        if pad_index is not None:
            self._pad_index = pad_index
        if smoothing is not None:
            self._smoothing = smoothing
            self._confidence = 1.0 - smoothing

    def forward(self,
                x: torch.Tensor,
                target: torch.Tensor,
                padding_mask=None) -> torch.Tensor:
        """
        :param x: log-probs [num_instances, vocab_size]
        :param target: [num_instances]
        """
        vocab_size = x.size(1)
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self._smoothing / (vocab_size - 2))  # Exclude pad and target.
        true_dist.scatter_(1, target.unsqueeze(1), self._confidence)
        true_dist[:, self._pad_index] = 0
        if padding_mask is None:
            padding_mask = target.eq(self._pad_index)
        true_dist.masked_fill_(padding_mask.unsqueeze(1), 0.0)

        return self._criterion(x, true_dist)
