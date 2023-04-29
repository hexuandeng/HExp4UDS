import torch
from torch.nn import MSELoss, BCELoss, BCEWithLogitsLoss

class MSECrossEntropyLoss(torch.nn.Module): 
    def __init__(self):
        super(MSECrossEntropyLoss, self).__init__()
        self.mse_criterion = MSELoss()
        self.xent_criterion = BCEWithLogitsLoss()

    def forward(self, output, target):
        mse_value = self.mse_criterion(output, target)
        # try:
        #     xent_value = self.xent_criterion(thresholded_output, thresholded_target)
        # except: # RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
        xent_value = self.xent_criterion(output, torch.gt(target, 0).float())

        if xent_value + mse_value != 0:
            harmonic_mean = 2*(xent_value * mse_value)/(xent_value + mse_value)
        else:
            # 0 by default, all 0's
            harmonic_mean = xent_value + mse_value
        return harmonic_mean
