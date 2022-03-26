import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpi_module import MPIModule
from projector import (divide_safe, over_composite, projective_forward_warp)

class RenderNet(nn.Module):
    def __init__(self, mode='single_mpi'):
        super(RenderNet, self).__init__()
        self.mpi_generator = MPIModule()
        self.mode = mode

    def forward(self, inputs):
        src_imgs, src_exts, inv_src_exts, src_ints, inv_src_ints = inputs[:5]
        tgt_ext, tgt_int = inputs[5:7]
        bd, disps = inputs[7:]

        b, v, c, h, w = src_imgs.shape

        min_disps, max_disps = self.get_disp_bounds(disps[:, -1], bd[:, 0], bd[:, 1])
        poses = torch.cat([inv_src_exts, torch.inverse(tgt_ext).unsqueeze(1)], 1)
        baseline = self.get_baseline(poses)
        focal_length = tgt_int[:, 0, 0]
        disp_range = max_disps - min_disps
        num_depths = self.get_num_depths(baseline, focal_length, disp_range, h, w)

        # Initialize depth array
        depths = []
        for b_idx in range(b):
            far = min_disps[b_idx].item()
            near = max_disps[b_idx].item()
            depth_arr = 1 / torch.linspace(far, near,
                    steps=num_depths[b_idx].item(), device=src_imgs.device)
            depths.append(depth_arr)
        depths = torch.stack(depths, 1)

        if self.mode == 'single_mpi':
            # Create single MPI
            mpi = self.mpi_generator(src_imgs.reshape([b, -1, h, w]),
                src_exts, inv_src_exts,
                src_ints, inv_src_ints, depths)
            transforms = torch.matmul(tgt_ext, inv_src_exts[:, 0])

            rgba_layers = mpi.permute([2, 0, 3, 4, 1])
            mpis = projective_forward_warp(rgba_layers, src_ints[:, 0], tgt_int,
                transforms, depths)
            rendering, _, acc_alpha, disp = over_composite(mpis, divide_safe(1., depths))

        else:
            raise NotImplementedError('mpi mode [%s] is not implemented' % self.mode)

        return rendering.permute([0, 3, 1, 2]), acc_alpha, disp

    def get_disp_bounds(self, tgt_disp, near_depth, far_depth, safe_bound=1e-8, use_disp=True):

        infty = 1e10

        random_factor = torch.rand(1).item()
        upper_random = 1. + 0.1 * random_factor ** 2

        min_disps = []
        max_disps = []

        for disp, nd, fd in zip(tgt_disp, near_depth, far_depth):
            if use_disp:
                max_disp = min([torch.max(disp) * upper_random, 1./safe_bound])
                min_disp = max([torch.min(disp) * .5, 1./infty])

            else:
                max_disp = min([torch.max(disp) * upper_random, 1./safe_bound])
                use_nd = (nd <= 0).float()
                max_disp = use_nd * max_disp + (1-use_nd) * 1. / nd

                min_disp = max([torch.min(disp) * .5, 1./infty])
                use_fd = (fd <= 0).float()
                min_disp = use_fd * min_disp + (1-use_fd) * 1. / fd

            min_disps.append(min_disp)
            max_disps.append(max_disp)

        max_disps = torch.stack(max_disps, 0)
        min_disps = torch.stack(min_disps, 0)

        return min_disps, max_disps

    def get_baseline(self, inv_exts):
        dt = inv_exts[:, 1:, :3, 3] - inv_exts[:, :1, :3, 3]
        dists = torch.sum(dt ** 2, 2) ** 0.5

        return torch.mean(dists, 1)

    def get_num_depths(self, baseline, focal_length, dist_range, h, w, num_rounding=16):
        num_depths_required = baseline * focal_length * dist_range
        im_size = h * w

        lower = 16.
        upper = 128. if im_size < 640*480 else 32.
        num_depths_required = torch.clamp(num_depths_required, lower, upper)
        num_depths = (num_depths_required // num_rounding) * num_rounding
        return num_depths.int()
