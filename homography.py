import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util import bilinear_wrapper

def divide_safe(num, den):
    eps = 1e-8
    den += eps * torch.eq(den, 0).type(torch.float)
    return num / den

def inv_homography(k_s, k_t, rot, t, n_hat, a):
    '''Compute inverse homography matrix between two cameras via a plane.
    
    Args:
        k_s: source camera intrinsics [..., 3, 3]
        k_t: target camera intrinsics [..., 3, 3]
        rot: relative roation [..., 3, 3]
        t: translation from source to target camera [..., 3, 1]
        n_hat: plane normal w.r.t source camera frame [..., 1, 3]
        a: plane equation displacement [..., 1, 1]        
    Returns:
        Inverse homography matrices (mapping from target to source)
        [..., 3, 3]
    '''
    rot_t = rot.transpose(-2, -1)
    k_t_inv = torch.inverse(k_t)
    
    denom = a - torch.matmul(torch.matmul(n_hat, rot_t), t)
    numerator = torch.matmul(torch.matmul(torch.matmul(rot_t, t),
                                          n_hat),
                             rot_t)
    inv_hom = torch.matmul(
            torch.matmul(k_s, rot_t + divide_safe(numerator, denom)),
            k_t_inv)
    return inv_hom

def transform_points(points, homography):
    '''Transforms input points according to the homography.
    
    Args:
        points: pixel coordinates [..., height, width, 3]
        homography: transformation [..., 3, 3]
    Returns:
        Transformed coordinates [..., height, width, 3]
    '''
    orig_shape = points.shape
    points_reshaped = points.view(orig_shape[:-3] +
                                  (-1,) +
                                  (orig_shape[-1:]))
    dim0 = len(homography.shape) - 2
    dim1 = dim0 + 1
    transformed_points = torch.matmul(points_reshaped,
                                      homography.transpose(dim0, dim1))
    transformed_points = transformed_points.view(orig_shape)
    return transformed_points

def normalize_homogenous(points):
    '''Converts homogenous coordinates to euclidean coordinates.
    
    Args:
        points: points in homogenous coordinates [..., #dimensions + 1]
    Returns:
        Points in standard coordinates after dividing by the last entry
        [..., #dimensions]
    '''
    uv = points[..., :-1]
    w = points[..., -1].unsqueeze(-1)
    return divide_safe(uv, w)

def transform_plane_imgs(imgs, pixel_coords, k_s, k_t, rot, t, n_hat, a):
    '''Transforms input images via homographies for corresponding planes.
    
    Args:
        imgs: input images [..., height_s, width_t, #channels]
        pixel_coords: pixel coordinates [..., height_t, width_t, 3]
        k_s: source camera intrinsics [..., 3, 3]
        k_t: target camera intrinsics [..., 3, 3]
        rot: relative rotation [..., 3, 3]
        t: translation from source to target camera [..., 3, 1]
        n_hat: plane normal w.r.t source camera frame [..., 1, 3]
        a: plane equation displacement [..., 1, 1]
    Returns:
        Images after bilinear sampling from the input.
    '''
    tgt2src = inv_homography(k_s, k_t, rot, t, n_hat, a)
    pixel_coords_t2s = transform_points(pixel_coords, tgt2src)
    pixel_coords_t2s = normalize_homogenous(pixel_coords_t2s)
    pixel_coords_t2s[..., 0] = pixel_coords_t2s[..., 0] /\
                               pixel_coords_t2s.shape[-2] * 2 - 1
    pixel_coords_t2s[..., 1] = pixel_coords_t2s[..., 1] /\
                               pixel_coords_t2s.shape[-3] * 2 - 1
    imgs_s2t = bilinear_wrapper(imgs, pixel_coords_t2s)
    
    return imgs_s2t

def planar_transform(src_imgs, pixel_coords, k_s, k_t, rot, t, n_hat, a):
    '''Transforms images, masks and depth maps according to 
       planar transformation.

    Args:
        src_imgs: input images [layer, batch, height_s, width_s, #channels]
        pixel_coords: coordinates of target image pixels
                      [batch, height_t, width_t, 3]
        k_s: source camera intrinsics [batch, 3, 3]
        k_t: target camera intrinsics [batch, 3, 3]
        rot: relative rotation [batch, 3, 3]
        t: translation from source to target camera [batch, 3, 1]
        n_hat: plane normal w.r.t source camera frame [layer, batch, 1, 3]
        a: plane equation displacement [layer, batch, 1, 1]
    Returns:
        Images projected to target frame [layer, height, width, #channels]
    '''
    layer = src_imgs.shape[0]
    rot_rep_dims = [layer]
    rot_rep_dims += [1 for _ in range(len(k_s.shape))]
    
    cds_rep_dims = [layer]
    cds_rep_dims += [1 for _ in range(len(pixel_coords.shape))]

    k_s = k_s.repeat(rot_rep_dims)
    k_t = k_t.repeat(rot_rep_dims)
    t = t.repeat(rot_rep_dims)
    rot = rot.repeat(rot_rep_dims)
    pixel_coords = pixel_coords.repeat(cds_rep_dims)
    
    tgt_imgs = transform_plane_imgs(
            src_imgs, pixel_coords, k_s, k_t, rot, t, n_hat, a)
    
    return tgt_imgs
    
    
if __name__ == '__main__':
    device = torch.device('cpu')
    num = torch.eye(4, device=device)
    den = torch.zeros([4, 4], device=device)
    print(divide_safe(num, den))
    
    k_s = torch.eye(3, device=device).expand(6, -1, -1)
    k_t = torch.eye(3, device=device).expand(6, -1, -1)
    rot = torch.eye(3, device=device).expand(6, -1, -1)
    t = torch.zeros([3, 1], device=device).expand(6, -1, -1)
    n_hat = torch.Tensor([[0, 0, 1]]).to(device).expand(3, 6, -1, -1)
    a = torch.Tensor([[0.0]]).to(device).expand(3, 6, -1, -1)
    inv_hom = inv_homography(k_s, k_t, rot, t, n_hat, a)
    
    points = torch.zeros([6, 64, 64, 3])
    homography = torch.zeros([6, 3, 3])
    transformed = transform_points(points, homography)
    print(transformed.shape)
    
    norm = normalize_homogenous(points)
    print(norm.shape)
    
    imgs = torch.randn([3, 6, 64, 64, 3], device=device)
    coords = torch.randn([3, 6, 32, 32, 2], device=device)
    
    res = bilinear_wrapper(imgs, coords)
    print(res.shape)
    
    
    coords = torch.randn([6, 32, 32, 3], device=device)
    print(k_s.shape)
    print(k_t.shape)
    print(rot.shape)
    print(t.shape)
    print(n_hat.shape)
    print(a.shape)
    
    res = planar_transform(imgs, coords, k_s, k_t, rot, t, n_hat, a)
    print(res.shape)
    