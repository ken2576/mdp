import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vgg import BatchNormVgg19

class VggLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(VggLoss, self).__init__()
        self.vgg = BatchNormVgg19()
    def forward(self, src, tgt):
        src_prime = self.normalize_batch(src)
        tgt_prime = self.normalize_batch(tgt)
        src_output = self.vgg(src_prime)
        tgt_output = self.vgg(tgt_prime)
        p0 = self.compute_error(src_prime, tgt_prime)
        p1 = self.compute_error(src_output[0], tgt_output[0]) / 2.6
        p2 = self.compute_error(src_output[1], tgt_output[1]) / 4.8
        p3 = self.compute_error(src_output[2], tgt_output[2]) / 3.7
        p4 = self.compute_error(src_output[3], tgt_output[3]) / 5.6
        p5 = self.compute_error(src_output[4], tgt_output[4]) * 10 / 1.5
        total_loss = p0 + p1 + p2 + p3 + p4 + p5
        return total_loss

    def compute_error(self, fake, real):
        return torch.mean(torch.abs((fake - real)))

    @staticmethod
    def normalize_batch(batch):
        """Normalize batch with VGG mean and std
        """
        mean = torch.zeros_like(batch)
        std = torch.zeros_like(batch)

        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225

        ret = batch - mean
        ret = ret / std
        return ret