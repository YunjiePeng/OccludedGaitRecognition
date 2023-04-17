import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BasicLoss
from utils import ddp_all_gather

class NonOccRecMSELoss(BasicLoss):
    def __init__(self, in_c, loss_term_weights=1.0):
        super(NonOccRecMSELoss, self).__init__()
        self.loss_term_weights = loss_term_weights

    def forward(self, nonOcc_x, nonOcc_y, occ_x, occ_y):
        # x/y: [s*n, c, p]

        nonOcc_loss = F.mse_loss(nonOcc_x, nonOcc_y, reduction='none').mean() # Overall Avg
        occ_loss = F.mse_loss(occ_x, occ_y, reduction='none').mean() # Overall Avg

        loss = nonOcc_loss

        self.info.update({
            'occ': occ_loss,
            'nonOcc': nonOcc_loss,
            'loss': loss})

        return loss, self.info