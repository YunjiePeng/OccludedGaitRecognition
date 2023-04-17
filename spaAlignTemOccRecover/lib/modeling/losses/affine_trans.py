import torch
import torch.nn.functional as F

from .base import BasicLoss
from utils import ddp_all_gather


class AffineTransformationLoss(BasicLoss):
    def __init__(self, loss_term_weights=1.0, alpha=1.0, beta=0.1):
        super(AffineTransformationLoss, self).__init__()
        self.loss_term_weights = loss_term_weights
        self.alpha = alpha
        self.beta = beta

    def forward(self, theta, theta_gt, x, x_gt):
        # theta: [n, 3], theta_gt: [n, 3]; resize_ratio; translation_x; translation_y
        # x: [n, 1, h, w], x_gt: [n, 1, h, w]
        n = theta.size()[0]
        theta = theta.view(n, -1)
        theta_gt = theta_gt.view(n, -1)
        x = x.view(n, -1)
        x_gt = x_gt.view(n, -1)

        theta_num = theta.size()[-1]
        pixel_num = x.size()[-1]
        
        affineMatrixloss = torch.norm(theta - theta_gt, p=2, dim=-1).mean()
        affineTransloss = torch.norm(x - x_gt, p=2, dim=-1).mean() / pixel_num

        loss_sum = self.alpha * affineMatrixloss + self.beta * affineTransloss

        self.info.update({
            'affineMatrixloss': affineMatrixloss,
            'affineTransloss': affineTransloss,
            'loss_num': n,
            'loss_sum': loss_sum})

        return loss_sum, self.info