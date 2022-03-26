import os
import glob

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from projector import poses2mat, pose2mat
from llff2sidewinder import llff2sidewinder

class LLFFDataset(Dataset):
    def __init__(self, data_root, transforms=None, ret_disp=False):
        self.data_root = data_root
        self.transforms = transforms
        self.ret_disp = ret_disp
        self._init_dataset()

    def _init_dataset(self):
        self.poses = []
        self.bds = []
        self.imgs = []
        self.disps = []
        for scene in sorted(glob.glob(self.data_root + '/*/*/')):
            path_arr = scene.split('/')
            if 'arch1' in path_arr or \
                'arch2' in path_arr or \
                'arch3' in path_arr:
                print('bad data')
                fix_data = True
            else:
                fix_data = False

            # Read poses
            pose_path = scene + 'poses_bounds.npy'
            tmp = np.load(pose_path)
            poses = tmp[:, :-2].reshape([-1, 3, 5])
            bds = tmp[:, -2:]
            new_poses = [llff2sidewinder(x, fix_data) for x in poses]
            new_poses = torch.from_numpy(np.stack(new_poses, 0))
            # Convert poses
            exts, ints = poses2mat(new_poses, 1)
            self.poses.append([exts, ints])
            self.bds.append(bds)

            # Store image paths
            im_paths = sorted(glob.glob(scene + 'img*.png'))
            disps_paths = sorted(glob.glob(scene + 'disp*.npy'))
            self.imgs.append(im_paths)
            self.disps.append(disps_paths)

        total_pose_count = sum(len(x[0]) for x in self.poses)
        total_bd_count = len(self.bds)
        total_img_count = sum(len(x) for x in self.imgs)
        total_disp_count = sum(len(x) for x in self.disps)
        print(f'Read {len(self.imgs)} folders, {total_img_count} images,',
            f'{total_pose_count} poses, {total_disp_count} disps,',
            f'{total_bd_count} bounds')

    def __getitem__(self, idx):
        if not self.transforms:
            trnfs = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            trnfs = self.transforms

        # Read data
        imgs = []
        disps = []
        
        for im_path, disp_path in zip(self.imgs[idx], self.disps[idx]):
            # Read images
            im = trnfs(imageio.imread(im_path))
            imgs.append(im)
            

            if self.ret_disp:
                # Read depth images
                disp = trnfs(np.load(disp_path).astype(np.float32))
                disps.append(disp)
        exts, ints = self.poses[idx]

        imgs = torch.stack(imgs, 0)
        if self.ret_disp:
            disps = torch.cat(disps, 0)

        # Acquire depth boundary
        tmp = torch.from_numpy(self.bds[idx])
        bd = torch.Tensor([torch.min(tmp[:, 0]), torch.max(tmp[:, 1])])


        # Random permutation
        # First image is the reference image for MPI
        # Last image is the target image (supervision)
        perm = np.random.permutation(len(imgs))
        # print(perm)

        imgs = torch.stack([imgs[x] for x in perm], 0)
        if self.ret_disp:
            disps = torch.stack([disps[x] for x in perm], 0)
        exts = torch.stack([exts[x] for x in perm], 0)
        ints = torch.stack([ints[x] for x in perm], 0)

        # Inverse matrices
        inv_exts = torch.inverse(exts)
        inv_ints = torch.inverse(ints)

        # Data and target
        if self.ret_disp:
            data = imgs[:-1], exts[:-1], inv_exts[:-1], \
                ints[:-1], inv_ints[:-1], \
                exts[-1], ints[-1], bd, disps
            target = imgs[-1]
        else:
            data = imgs[:-1], exts[:-1], inv_exts[:-1], \
                ints[:-1], inv_ints[:-1], \
                exts[-1], ints[-1], bd
            target = imgs[-1]

        return (data, target)

    def __len__(self):
        return len(self.poses)

class RenderDataset(Dataset):
    def __init__(self, data_root, pose_type='default', scale=1, transforms=None):
        self.data_root = data_root
        self.transforms = transforms
        self.scale = scale
        self.pose_type = pose_type
        self._init_dataset()

    def _init_dataset(self):
        self.poses = []
        self.bds = []
        self.imgs = []
        # Read poses
        pose_path = os.path.join(self.data_root, 'poses_bounds.npy')
        tmp = torch.from_numpy(np.load(pose_path))
        poses = tmp[:, :-2].reshape([-1, 3, 5])
        bds = tmp[:, -2:]
        if self.pose_type == 'llff':
            new_poses = [llff2sidewinder(x, False) for x in poses]
            poses = torch.from_numpy(np.stack(new_poses, 0))
        # Convert poses
        exts, ints = poses2mat(poses, self.scale)
        self.poses.append([exts, ints])
        self.bds.append(bds)

        # Store image paths
        im_paths = sorted(glob.glob(os.path.join(self.data_root, 'img*.png')))
        self.imgs.append(im_paths)

    def __getitem__(self, idx):
        if not self.transforms:
            trnfs = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            trnfs = self.transforms

        # Read data
        imgs = []
        
        for im_path in self.imgs[idx]:
            # Read images
            im = trnfs(imageio.imread(im_path)[..., :3])
            imgs.append(im)
            
        exts, ints = self.poses[idx]

        imgs = torch.stack(imgs, 0)

        # Acquire depth boundary
        tmp = self.bds[idx]
        bd = torch.Tensor([torch.min(tmp[:, 0]), torch.max(tmp[:, 1])])

        return imgs, exts, ints, bd

    def __len__(self):
        return len(self.poses)