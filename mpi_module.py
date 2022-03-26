import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from projector import (projective_inverse_warp, batch_inverse_warp, meshgrid_pinhole)
from util import save_psv

class MPINet3d(nn.Module):
    def __init__(self, in_c, out_c, nf=8):
        super(MPINet3d, self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv3d(in_c, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv3d(nf, nf*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*2)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv3d(nf*2, nf*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*2)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv3d(nf*2, nf*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*4)
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv3d(nf*4, nf*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*4)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv3d(nf*4, nf*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*4)
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv3d(nf*4, nf*8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*8)
        )

        self.conv4_1 = nn.Sequential(
            nn.Conv3d(nf*8, nf*8, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*8)
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv3d(nf*8, nf*8, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*8)
        )
        self.conv4_3 = nn.Sequential(
            nn.Conv3d(nf*8, nf*8, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*8)
        )

        '''Upsampling'''
        self.conv5_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(nf*16, nf*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*4)
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv3d(nf*4, nf*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*4)
        )
        self.conv5_3 = nn.Sequential(
            nn.Conv3d(nf*4, nf*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*4)
        )

        '''Upsampling'''
        self.conv6_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(nf*8, nf*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*2)
        )
        self.conv6_2 = nn.Sequential(
            nn.Conv3d(nf*2, nf*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf*2)
        )

        '''Upsampling'''
        self.conv7_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(nf*4, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf)
        )
        self.conv7_2 = nn.Sequential(
            nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(1, nf)
        )

        self.conv7_3 = nn.Sequential(
            nn.Conv3d(nf, out_c, kernel_size=3, stride=1, padding=1)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        conv1 = self.conv1_2(self.conv1_1(x))
        conv2 = self.conv2_2(self.conv2_1(conv1))
        conv3 = self.conv3_3(self.conv3_2(self.conv3_1(conv2)))
        conv4 = self.conv4_3(self.conv4_2(self.conv4_1(conv3)))

        x = torch.cat([conv4, conv3], 1)
        conv5 = self.conv5_3(self.conv5_2(self.conv5_1(x)))
        x = torch.cat([conv5, conv2], 1)
        conv6 = self.conv6_2(self.conv6_1(x))
        x = torch.cat([conv6, conv1], 1)
        x = self.conv7_3(self.conv7_2(self.conv7_1(x)))

        weights = x[:, 1:]
        weights = torch.cat([torch.zeros_like(x[:, :1]), weights], 1)
        weights = F.softmax(weights, 1)
        alpha = self.sigmoid(x[:, :1])

        return torch.cat([alpha, weights], 1)

class MPIModule(nn.Module):
    '''Computes disparity map from 3 input views (left, center, right)
    '''
    def __init__(self):
        super(MPIModule, self).__init__()
        self.views = 5
        self.volume_generator = MPINet3d(in_c=self.views*3, out_c=5)

    def forward(self, im_arr, src_exts, inv_src_exts, src_ints, inv_src_ints, depths):
        b, c, h, w = im_arr.shape
        d = depths.shape[0]
        v = self.views

        im_arr = im_arr.reshape([b*v, int(c/v), h, w])

        # Create PSV at the reference camera
        src_ints_prime = src_ints.reshape([b*v, 3, 3])
        src_exts_prime = src_exts.reshape([b*v, 4, 4])
        inv_src_ints_prime = inv_src_ints.reshape([b*v, 3, 3])

        psv_tgt = inv_src_exts[:, 0:1].repeat([1, v, 1, 1])
        psv_tgt = psv_tgt.reshape([b*v, 4, 4])

        feature = im_arr + 1.
        depths = depths.unsqueeze(2).expand([-1, -1, v]).reshape([d, -1])
        psv = self.create_psv(feature, src_exts_prime, src_ints_prime,
                    psv_tgt, inv_src_ints_prime, depths)
        psv = psv.reshape([b, v * 3, d, h, w]) - 1.
    
        sv = self.volume_generator(psv)
        weights = sv[:, 1:]
        
        alpha = sv[:, :1]
        mpi = torch.zeros_like(sv[:, :4])
        tmp = torch.zeros_like(sv[:, :3])

        for v_idx in range(v):
            tmp += weights[:, v_idx:v_idx+1] * psv[:, 3*v_idx:3*(v_idx+1)]

        mpi[:, :3] = tmp
        mpi[:, 3:4] = alpha
        return mpi
    
    def create_psv(self, src_imgs, src_exts, src_ints, inv_tgt_exts, inv_tgt_ints, depths):
        '''Create PSV from inputs

        Args:
            src_imgs: source images [batch, #channels, height, width]
            src_exts: source extrinsics [batch, 4, 4]
            src_ints: source intrinsics [batch, 3, 3]
            inv_tgt_exts: inverse target extrinsics [batch, 4, 4]
            inv_tgt_ints: inverse target intrinsics [batch, 3, 3]
            depths: depth values [#planes, batch]
        Returns:
            Plane sweep volume [batch, #channels, #depth_planes, height, width]
        '''
        trnfs = torch.matmul(src_exts, inv_tgt_exts)

        psv = batch_inverse_warp(src_imgs, depths,
                src_ints, inv_tgt_ints, trnfs)

        return psv
