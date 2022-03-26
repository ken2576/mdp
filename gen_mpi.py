import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from loader import RenderDataset
from mpi_module import MPIModule
from projector import (divide_safe, over_composite, projective_forward_warp)
from util import create_folder

class Renderer(nn.Module):
    def __init__(self, mode='single_mpi'):
        super(Renderer, self).__init__()
        self.mpi_generator = MPIModule()
        self.mode = mode

    def forward(self, inputs, near, far, num_depths, indices):
        imgs, exts, ints, bd = inputs

        src_imgs = torch.stack([imgs[:, x] for x in indices], 1)
        src_ints = torch.stack([ints[:, x] for x in indices], 1)
        src_exts = torch.stack([exts[:, x] for x in indices], 1)
        inv_src_ints = torch.inverse(src_ints)
        inv_src_exts = torch.inverse(src_exts)

        b, v, c, h, w = src_imgs.shape
        print(src_imgs.shape)

        # Initialize depth array
        depths = []
        for b_idx in range(b):
            depth_arr = 1 / torch.linspace(1 / far, 1 / near,
                    steps=num_depths, device=src_imgs.device)
            depths.append(depth_arr)
        depths = torch.stack(depths, 1)

        if self.mode == 'single_mpi':
            # Create single MPI
            mpi = self.mpi_generator(src_imgs.reshape([b, -1, h, w]),
                src_exts, inv_src_exts,
                src_ints, inv_src_ints, depths)

        else:
            raise NotImplementedError('mpi mode [%s] is not implemented' % self.mode)

        return mpi, depths

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scene', type=str,
                        help='path to scene folder')
    parser.add_argument('--out', type=str,
                        help='path to output folder')
    parser.add_argument('--model_path', type=str,
                        help='path to checkpoint')
    parser.add_argument('--type', type=str,
                        default='default',
                        help='pose type (\'llff\' or \'default\')')
    args = parser.parse_args()

    if args.type == 'llff':
        train_dataset = RenderDataset(args.scene, 'llff', scale=100)
    else:
        train_dataset = RenderDataset(args.scene, 'default', scale=1)

    out_folder = args.out
    create_folder(out_folder)

    imgs, exts, ints, bd = train_dataset[0]
    inputs = [imgs, exts, ints, bd]
    inputs = [x.unsqueeze(0).float().to('cuda') for x in inputs]

    model = Renderer().to('cuda')
    state = model.state_dict()     
    state.update(torch.load(args.model_path)['model_state_dict'])
    model.load_state_dict(state)

    c_idx = 0
    neigh_idx = np.array([
        c_idx,
        c_idx-1,
        c_idx+1,
        c_idx-2,
        c_idx+2
    ])

    cam_indices = [
        neigh_idx-2,
        neigh_idx-1,
        neigh_idx,
        neigh_idx+1,
        neigh_idx+2
    ]

    cam_indices = [neigh_idx+i for i in range(16)]
    cam_indices = [ids % 16 for ids in cam_indices]

    model.eval()
    with torch.no_grad():
        for idx, indices in enumerate(cam_indices):
            if args.type == 'llff':    
                mpi, depths = model(inputs, 0.45, 8, 32, indices) # unreal
            else:
                mpi, depths = model(inputs, 1.1, 30, 32, indices) # real
            
            mpi = mpi.permute([0, 3, 4, 2, 1])
            dispvals = divide_safe(1, depths * 100).squeeze(1)

            np.save(out_folder + 'mpi_' + str(idx).zfill(2) + '.npy', mpi.cpu().numpy())
            np.save(out_folder + 'dispval_' + str(idx).zfill(2) + '.npy', dispvals.cpu().numpy())

if __name__ == '__main__':
    main()