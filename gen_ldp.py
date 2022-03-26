import os
import glob
import json
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from ldp import *
from projector import divide_safe, poses2mat
from llff2sidewinder import llff2sidewinder
from util import create_folder

def read_real_data(folder):
    ### Read MPI and dispval ###
    indices = [x for x in range(16)]
    mpi_paths = sorted(glob.glob(os.path.join(folder, 'mpi_*.npy')))
    disp_paths = sorted(glob.glob(os.path.join(folder, 'dispval_*.npy')))
    mpi_paths = [mpi_paths[x] for x in indices]
    disp_paths = [disp_paths[x] for x in indices]
    
    mpis = []
    dispvals = []
    for mpath, dpath in zip(mpi_paths, disp_paths):
        mpi = torch.from_numpy(np.load(mpath)).permute([3, 0, 1, 2, 4])
        mpis.append(mpi)
        dispval = torch.from_numpy(np.load(dpath)).unsqueeze(0)
        dispvals.append(dispval)
    
    mpis = torch.cat(mpis, 1).to('cuda')
    dispvals = torch.cat(dispvals, 0).to('cuda')
    
    return mpis, dispvals

def read_real_pose(folder, pose_type='default', scale=0.01):

    ### Set path ###
    pose_path = os.path.join(folder, 'poses_bounds.npy')

    ### Read Poses ###
    pose_indices = [x for x in range(16)]
    # pose_indices = [10, 11, 12, 13, 14]
    
    data = torch.from_numpy(np.load(pose_path))
    poses = data[:, :-2].reshape([-1, 3, 5])

    if pose_type == 'llff':
        new_poses = [llff2sidewinder(x, False) for x in poses]
        poses = torch.from_numpy(np.stack(new_poses, 0))

    exts, ints = poses2mat(poses, scale)
    
    inv_exts = torch.inverse(exts)
    cam_center = torch.mean(inv_exts[:, :3, 3], 0)
    
    # exts = torch.stack([exts[x] for x in pose_indices], 0)
    # ints = torch.stack([ints[x] for x in pose_indices], 0)
    # TODO read everything for now
    
    tgt_ext = exts[0].clone()
    inv_tgt_ext = torch.inverse(tgt_ext)
    inv_tgt_ext[:3, 3] = cam_center
    tgt_ext = torch.inverse(inv_tgt_ext).to('cuda')
    ints = ints.to('cuda')
    exts = exts.to('cuda')

    return ints, exts, tgt_ext

def translation(ext, angle=315, num=300):
    # angles = np.linspace(0, 360, num)
    rad = torch.linspace(0, 8*np.pi, num)
    radius = 25
    shifts = torch.stack([
        (rad / (8*np.pi)) * radius * np.cos(rad),
        torch.zeros(num),
        (rad / (8*np.pi)) * radius * np.sin(rad),
    ]).T.to(ext.device)

    ext_arr = []
    for shift in shifts:
        t = torch.eye(4, device=ext.device)
        t[:3, 3] = shift

        curr_ext = ext.clone()
        rot_mat = torch.eye(4).to(curr_ext.device)
        rot_mat[:3, :3] = pitch_mat(angle / 180 * np.pi)
        curr_ext = torch.matmul(rot_mat, curr_ext)

        inv_ext = torch.inverse(curr_ext)
        new_inv_ext = torch.matmul(inv_ext, t)
        curr_ext = torch.inverse(new_inv_ext)
        
        ext_arr.append(curr_ext)

    return ext_arr

def pitch_mat(p):
    # Actually yaw for real data
    p_mat = torch.tensor([
        [np.cos(p), 0, -np.sin(p)],
        [        0, 1,          0],
        [np.sin(p), 0,  np.cos(p)],
    ])
    return p_mat

def yaw_mat(y):
    # Actually roll for real data
    y_mat = torch.tensor([
        [ np.cos(y), np.sin(y), 0],
        [-np.sin(y), np.cos(y), 0],
        [         0,         0, 1],
    ])
    return y_mat

def linear(ext, angle=30, num=150):
    rad = torch.linspace(0, 2*np.pi, num)
    radius = 25
    shifts = torch.stack([
        (rad / (2*np.pi)) * radius * torch.cos(rad),
        torch.zeros(num),
        torch.zeros(num),
    ]).T.to(ext.device)

    ext_arr = []
    for shift in shifts:
        t = torch.eye(4, device=ext.device)
        t[:3, 3] = shift

        curr_ext = ext.clone()
        rot_mat = torch.eye(4).to(curr_ext.device)
        rot_mat[:3, :3] = pitch_mat(angle / 180 * np.pi)
        curr_ext = torch.matmul(rot_mat, curr_ext)

        inv_ext = torch.inverse(curr_ext)
        new_inv_ext = torch.matmul(inv_ext, t)
        curr_ext = torch.inverse(new_inv_ext)
        
        ext_arr.append(curr_ext)
    return ext_arr

def rotation(ext, num=600):
    angles = torch.linspace(0, 360, num)

    ext_arr = []
    for angle in angles:
        curr_ext = ext.clone()
        rot_mat = torch.eye(4).to(curr_ext.device)
        rot_mat[:3, :3] = pitch_mat(angle / 180 * np.pi)
        curr_ext = torch.matmul(rot_mat, curr_ext)
        ext_arr.append(curr_ext)
    return ext_arr

def rotation_translation(ext, num=600):
    rad = torch.linspace(0, 8*np.pi, num)
    angles = torch.cos(rad) * 20 + 300
    radius = 25
    shifts = torch.stack([
        (rad / (8*np.pi)) * radius * torch.cos(rad),
        torch.zeros(num),
        (rad / (8*np.pi)) * radius * torch.sin(rad),
    ]).T.to(ext.device)

    ext_arr = []
    for angle, shift in zip(angles, shifts):
        inv_ext = torch.inverse(ext)
        inv_ext[:3, 3] += shift
        curr_ext = torch.inverse(inv_ext)

        rot_mat = torch.eye(4).to(curr_ext.device)
        rot_mat[:3, :3] = pitch_mat(angle / 180 * np.pi)
        curr_ext = torch.matmul(rot_mat, curr_ext)
        ext_arr.append(curr_ext)
    return ext_arr

def gen_real_ldp(mpi_folder, pose_folder, out_folder):
    from util import create_folder
    create_folder(out_folder)

    mpis, dispvals = read_real_data(mpi_folder)
    ints, exts, tgt_ext = read_real_pose(pose_folder)
    new_ext = torch.eye(4).to(tgt_ext.device)
    new_ext[:3, 3] = tgt_ext[:3, 3] # to fix stupid rotation
    tgt_ext = new_ext

    # HACK
    ints[:, 0, 0] = 320.
    ints[:, 1, 1] = 320.

    num_planes = 5 
    tgt_h = 640
    tgt_w = 2560

    v = mpis.shape[1]

    ldps = []
    for i in range(v):
        mpi = mpis[:, i]
        dispval = dispvals[i] # if deep sidewinder

        # TODO calculate warped disp instead...
        max_disp = torch.max(dispval)
        min_disp = 0
        disp_delta = (max_disp - min_disp) / num_planes

        depth_arr = divide_safe(1, dispval)

        ldp = mpi_to_ldp(mpi, depth_arr, disp_delta,
                ints[i], exts[i], tgt_ext, tgt_h, tgt_w, num_planes)
        ldps.append(ldp)
    ldps = torch.stack(ldps, 0)
    weights = cosine_weights(ldps, ints, exts, tgt_ext)
    weights = torch.exp((weights - 1)*40)

    print(ldps.shape)
    final_ldp = merge_ldp(ldps, weights)

    print(final_ldp.shape)
    np.save(out_folder + 'ldp.npy', final_ldp.cpu().numpy())

def render_image(ldp_folder, output_folder):

    device = 'cuda'
    create_folder(output_folder)

    ldp_path = os.path.join(ldp_folder, 'ldp.npy')
    ldp = torch.from_numpy(np.load(ldp_path)).float().to(device)
    
    tgt_int = torch.tensor([
        [320, 0, 320],
        [0, 320, 320],
        [0, 0, 1]
    ]).float().to(device)

    src_ext = torch.eye(4).to(device)
    tgt_exts = [torch.eye(4).to(device)]

    for idx, tgt_ext in enumerate(tgt_exts):
        rendering = render_ldp(ldp, src_ext, tgt_int, tgt_ext)
        rendering = torch.clamp(rendering, 0.0, 1.0)

        plt.imsave(os.path.join(output_folder, str(idx) + '.png'), rendering[0].cpu().numpy())

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scene', type=str,
                        help='path to scene folder (for camera poses)')
    parser.add_argument('--mpi_folder', type=str,
                        help='path to mpi folder')
    parser.add_argument('--ldp_folder', type=str,
                        help='path to output ldp folder')
    parser.add_argument('--out_folder', type=str,
                        help='path to output folder')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    gen_real_ldp(args.mpi_folder, args.scene, args.ldp_folder)

    create_folder(args.out_folder)
    render_image(args.ldp_folder, args.out_folder)