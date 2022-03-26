import torch
import numpy as np
import matplotlib.pyplot as plt
from projector import (pose2mat, calc_adaptive_mpi, cylinder_coord, sphere_coord,
        meshgrid_pinhole, divide_safe, sphere_coord, over_composite, over_composite_sm)
from splat import project_points, project_depth, splat_image, project_points_spherical, project_depth_spherical


def pix_to_world(coord, depth_map, inv_src_int, inv_src_ext):
    '''Project point from source pixel coordinates to world coordinates

    Args:
        coord: input pixel coordinates [height, width, 3]
        depth_map: per-pixel depth map [batch, height, width]
        inv_src_int: inverse source intrinsic [3, 3]
        inv_src_ext: inverse source extrinsic [4, 4]
    Returns:
        World coordinates [batch, height * width, 4, 1]
    '''
    b, h, w = depth_map.shape
    flattened_depth = depth_map.reshape([b, h*w])
    pix_coord = coord.view([h*w, 3, 1])
    cam_coord = torch.matmul(inv_src_int, pix_coord)
    cam_coord = cam_coord * flattened_depth[..., None, None]
    ones = torch.ones_like(flattened_depth).unsqueeze(2).unsqueeze(2)
    cam_coord = torch.cat([cam_coord, ones], 2)
    world_coord = torch.matmul(inv_src_ext, cam_coord)
    return world_coord

def world_to_pix(world_coord, tgt_int, tgt_ext):
    '''Project point from world coordinates to pixel coordinates

    Args:
        world_coord: input world coordinates [batch, ..., 4, 1]
        tgt_int: target intrinsic [3, 3]
        tgt_ext: target extrinsic [4, 4]
    Returns:
        Pixel coordinates [batch, ..., 2]
        Warped depth [batch, ...]
    '''
    new_cam_coord = torch.matmul(tgt_ext, world_coord)
    warped_depth = new_cam_coord[..., 2, 0]
    pix_coord = torch.matmul(tgt_int, new_cam_coord[..., :3, :])
    pix_coord = pix_coord.squeeze(-1)
    zeros = torch.zeros_like(pix_coord[..., 2:3])
    pix_coord[..., 2:3] = torch.where(pix_coord[..., 2:3] > 0, pix_coord[..., 2:3], zeros)
    pix_coord = divide_safe(pix_coord[..., :2], pix_coord[..., 2:3])
    return pix_coord, warped_depth

def world_to_cylindrical(world_coord, tgt_h, tgt_w, tgt_ext):
    '''Project point from world coordinates to cylindrical pixel coordinates

    Args:
        world_coord: input world coordinates [batch, ..., 4, 1]
        tgt_h: target height
        tgt_w: target width
        tgt_ext: target extrinsic [4, 4]
    Returns:
        Pixel coordinates [batch, ..., 2]
        Warped depth [batch, ...]
    '''
    cam_coord = torch.matmul(tgt_ext, world_coord)
    depth = torch.sqrt(cam_coord[..., 0, 0] ** 2 + cam_coord[..., 2, 0] ** 2)
    
    theta = torch.atan2(cam_coord[..., 0, 0], cam_coord[..., 2, 0]) / np.pi
    h = cam_coord[..., 1, 0] / torch.sqrt(cam_coord[..., 0, 0] ** 2 + cam_coord[..., 2, 0] ** 2)
    
    theta = (theta + 1) / 2. * tgt_w
    h = (h + 1) / 2. * tgt_h
    pix_coord = torch.stack([theta, h], -1)

    return pix_coord, depth

def cylindrical_to_world(coord, depth_map, inv_src_ext):
    '''Project point from cylindrical camera coordinates to world coordinates
    
    Args:
        coord: input cylindrical camera coordinates [height, width, 3]
        depth_map: per-pixel depth map [batch, height, width]
        src_int: source intrinsic [3, 3]
        inv_src_ext: inverse source extrinsic [4, 4]
    Returns:
        World coordinates [batch, height*width, 4, 1]
    '''
    b, h, w = depth_map.shape
    flattened_depth = depth_map.reshape([b, h*w])
    cam_coord = coord.view([h*w, 3, 1])
    cam_coord = cam_coord * flattened_depth[..., None, None]
    ones = torch.ones_like(flattened_depth).unsqueeze(2).unsqueeze(2)
    cam_coord = torch.cat([cam_coord, ones], 2)
    world_coord = torch.matmul(inv_src_ext, cam_coord)
    return world_coord

def world_to_cam(world_coord, tgt_ext):
    '''Project point from world coordinates to camera coordinates

    Args:
        world_coord: input world coordinates [batch, ..., 4, 1]
        tgt_ext: target extrinsic [4, 4]
    Returns:
        Camera coordinates [batch, ..., 4, 1]
    '''
    return torch.matmul(tgt_ext, world_coord)

def splat_point(coord_arr, rgba_arr, tgt_int, tgt_ext, tgt_imgs, scale=25):
    '''Forward splat 3D points onto images

    Args:
        coord_arr: list of 3D point coordinate arrays [#points, 4, 1]
        rgba_arr: list of 3D point RGBA arrays [#points, 4]
        tgt_int: target intrinsic [3, 3]
        tgt_ext: target extrinsic [4, 4]
        tgt_imgs: list of target images to be splatted onto [batch, tgt_height, tgt_width, #channels]
    Returns:
        Splatted images
    '''
    splatted_imgs = []
    for coord, rgba, tgt_im in zip(coord_arr, rgba_arr, tgt_imgs):
        pix_coord, depth = world_to_pix(coord, tgt_int, tgt_ext)
        disp = divide_safe(1, depth)
        max_disp = torch.max(disp)
        splatted = splat_image(rgba.unsqueeze(0).unsqueeze(2),
                               disp.unsqueeze(0).unsqueeze(2),
                               pix_coord.unsqueeze(0).unsqueeze(2),
                               max_disp,
                               tgt_im=tgt_im,
                               scale=scale
                            )
        splatted_imgs.append(splatted)
    return torch.stack(splatted_imgs, 0)

def pinhole_to_cylindrical(h, w, depth, src_int, src_ext, tgt_h, tgt_w, tgt_ext):
    '''Project pinhole coordinates to world coordinate system, then cylindrical
    
    Args:
        h: input grid height
        w: input grid width
        depth: input plane depth
        src_int: source intrinsic [3, 3]
        src_ext: source extrinsic [4, 4]
        tgt_h: target height
        tgt_w: target width
        tgt_ext: target extrinsic [4, 4]
    Returns:
        Cylindrical coordinates and corresponding depth (radius)
        [batch, ..., 2] and [batch, ...]
    '''
    grid = meshgrid_pinhole(h, w,
                        device=src_int.device)
    depth_map = depth * torch.ones_like(grid[..., 0]).unsqueeze(0)
    world_coord = pix_to_world(grid, depth_map,
            torch.inverse(src_int), torch.inverse(src_ext))
    pix_coord, depth = world_to_cylindrical(world_coord, tgt_h, tgt_w, tgt_ext)
    
    return pix_coord, depth

def mpi_to_ldp(mpi, depth_arr, disp_delta, src_int, src_ext, tgt_ext, tgt_h, tgt_w, num_planes):
    '''Convert multiplane images to layered depth panorama
    
    Args:
        mpi: multiplane images [#planes, height, width, 4]
        depth_arr: list of depth for each plane [#planes]
        disp_delta: disparity interval size
        src_int: intrinsic of input MPI [3, 3]
        src_ext: extrinsic of input MPI [4, 4]
        tgt_ext: extrinsic of output LDP [4, 4]
        tgt_h: output image height
        tgt_w: output image width
        num_planes: output ldp plane count
    Returns:
        Layered depth panorama [#planes, height, width, 5]
    '''
    _, h, w, _ = mpi.shape
    
    ldp = torch.zeros([num_planes, tgt_h, tgt_w, 5], device=mpi.device)
    
    for mpi_layer, depth in zip(mpi, depth_arr):
        # Project to cylindrical coordinate system
        pix_coord, depth = pinhole_to_cylindrical(h, w, depth, src_int, src_ext,
                               tgt_h, tgt_w, tgt_ext)
        disp = divide_safe(1, depth)
        points = mpi_layer.reshape([h*w, 4]).unsqueeze(0)
        
        # Grouping for each layer into different LDP layers
        for d_idx in range(num_planes):
            start = disp_delta * (d_idx)
            end = disp_delta * (d_idx + 1)
            cond = (disp > start) & (disp <= end)
            selected_points = pix_coord[cond]
            if selected_points.shape[0] != 0:
                # If there are suitable points, project onto pano
                tgt_im = torch.zeros([1, tgt_h, tgt_w, 5], device=mpi.device)
                max_disp = torch.max(disp[cond])
                feature = torch.cat([points[cond], disp[cond].unsqueeze(-1)], -1)
                splatted = splat_image(feature.unsqueeze(0).unsqueeze(2),
                                   disp[cond].unsqueeze(0).unsqueeze(2),
                                   selected_points.unsqueeze(0).unsqueeze(2),
                                   max_disp,
                                   tgt_im=tgt_im,
                                )
                
                # Over operation for collapsing the pano
                # TODO might need to change this
                if torch.sum(ldp[d_idx, ..., 3]) == 0:
                    ldp[d_idx] = splatted[0]
                else:
                    # TODO simplify this
                    ldp[d_idx, ..., :3] = splatted[0, ..., :3] * splatted[0, ..., 3:4] + ldp[d_idx, ..., :3] * ldp[d_idx, ..., 3:4] * (1 - splatted[0, ..., 3:4])
                    ldp[d_idx, ..., 4:5] = splatted[0, ..., 4:5] * splatted[0, ..., 3:4] + ldp[d_idx, ..., 4:5] * ldp[d_idx, ..., 3:4] * (1 - splatted[0, ..., 3:4])
                    ldp[d_idx, ..., 3:4] = splatted[0, ..., 3:4] + ldp[d_idx, ..., 3:4] * (1 - splatted[0, ..., 3:4])
                    ldp[d_idx, ..., :3] = divide_safe(ldp[d_idx, ..., :3], ldp[d_idx, ..., 3:4])
                    ldp[d_idx, ..., 4:5] = divide_safe(ldp[d_idx, ..., 4:5], ldp[d_idx, ..., 3:4])
    
    return ldp

def render_ldp(ldp, ldp_ext, tgt_int, tgt_ext):
    
    d, h, w, c = ldp.shape
    
    tgt_h = tgt_int[1, 2].int().item() * 2
    tgt_w = tgt_int[0, 2].int().item() * 2

    # Calculate target pixel coordinates and depth    
    cam_coord = cylinder_coord(h, w, tgt_int,
                               is_homogenous=False, invert_y=True,
                               device=ldp.device)
    depth_maps = divide_safe(1, ldp[..., -1])
    world_coord = cylindrical_to_world(cam_coord, depth_maps, torch.inverse(ldp_ext))
    pix_coord, warped_depth = world_to_pix(world_coord, tgt_int, tgt_ext)
    pix_coord = pix_coord.reshape([d, h, w, 2])
    warped_depth = warped_depth.reshape([d, h, w])
    new_disp = divide_safe(1, warped_depth)

    mask = (pix_coord[..., 0] >= 0) & (pix_coord[..., 0] < tgt_w) & \
        (pix_coord[..., 1] >= 0) & (pix_coord[..., 1] < tgt_h)

    new_disp = new_disp * mask

    # Forward splat onto target view
    max_disparity = torch.max(torch.max(new_disp, 1)[0], 1)[0]   
    tgt_im = torch.zeros([d, tgt_h, tgt_w, 4], device=ldp.device)
    splatted = splat_image(ldp[..., :4], new_disp,
                pix_coord, max_disparity, tgt_im=tgt_im, scale=2500) # higher scale for low disp

    rendering = over_composite(splatted.unsqueeze(1))

    return rendering

def render_ldp_pano(ldp, ldp_ext, tgt_ext):
    
    d, h, w, c = ldp.shape

    # Calculate target pixel coordinates and depth    
    cam_coord = cylinder_coord(h, w, None, vfocal=320,
                               is_homogenous=False, invert_y=True,
                               device=ldp.device)
    depth_maps = divide_safe(1, ldp[..., -1])
    world_coord = cylindrical_to_world(cam_coord, depth_maps, torch.inverse(ldp_ext))
    pix_coord, warped_depth = world_to_cylindrical(world_coord, h, w, tgt_ext)
    pix_coord = pix_coord.reshape([d, h, w, 2])
    warped_depth = warped_depth.reshape([d, h, w])
    new_disp = divide_safe(1, warped_depth)

    mask = (pix_coord[..., 0] >= 0) & (pix_coord[..., 0] < w) & \
        (pix_coord[..., 1] >= 0) & (pix_coord[..., 1] < h)

    new_disp = new_disp * mask

    # Forward splat onto target view
    max_disparity = torch.max(torch.max(new_disp, 1)[0], 1)[0]   
    tgt_im = torch.zeros([d, h, w, 4], device=ldp.device)
    splatted = splat_image(ldp[..., :4], new_disp,
                pix_coord, max_disparity, tgt_im=tgt_im, scale=2500) # higher scale for low disp

    rendering = over_composite(splatted.unsqueeze(1))

    return rendering

def cosine_weights(ldps, src_ints, src_exts, tgt_ext):
    '''Calculate cosine distance for weighting

    Args:
        ldps: layered depth panorama [#views, #planes, height, width, 5]
        src_ints: source intrinsics [#views, 4, 4]
        src_exts: source extrinsics [#views, 4, 4]
        tgt_ext: target extrinsic [4, 4]
    Returns:
        Cosine distance weights [#views, height, width]
    '''
    v = src_ints.shape[0]
    h = src_ints[0, 0, -1].int().item() * 2
    w = src_ints[0, 1, -1].int().item() * 2

    _, d, tgt_h, tgt_w, _ = ldps.shape

    pix_coord = meshgrid_pinhole(h, w,
                        device=src_ints.device)
    pix_coord = pix_coord.view([h*w, 3, 1])
    cam_coord = torch.matmul(torch.inverse(src_ints.unsqueeze(1)), pix_coord)
    cam_coord = cam_coord.squeeze(2).view([v, h, w, 3])
    norm = torch.sqrt(torch.sum(cam_coord ** 2, -1))
    cosine_distance = divide_safe(1, norm)

    pix_coords = []
    depths = []
    for src_int, src_ext in zip(src_ints, src_exts):

        pix_coord, depth = pinhole_to_cylindrical(h, w, 1000000, src_int, src_ext,
                        tgt_h, tgt_w, tgt_ext)
        pix_coords.append(pix_coord)
        depths.append(depth)
    pix_coords = torch.cat(pix_coords, 0).reshape([v, h, w, 2])
    depths = torch.cat(depths, 0).reshape([v, h, w])
    tgt_im = torch.zeros([v, tgt_h, tgt_w, 1], device=ldps.device)
    disp = divide_safe(1, depths)
    max_disp = torch.max(disp.view([v, -1]), 1)[0]

    splatted = splat_image(cosine_distance.unsqueeze(-1), disp, pix_coords, max_disp, tgt_im=tgt_im)
    splatted = splatted.squeeze(-1)

    return splatted

def merge_ldp(ldps, weights):
    '''Merge LDP from different views into one LDP

    Args:
        ldps: layered depth panorama [#views, #planes, height, width, 5]
        weights: additional weights [#views, height, width]
    Returns:
        Merged LDP [#planes, height, width, 5]
    '''
    # final_ldp = divide_safe(torch.sum(ldps * ldps[..., 3:4], 0), torch.sum(ldps[..., 3:4], 0))
    if ldps.shape[0] > 8:
        num = torch.zeros_like(ldps[0])
        denom = torch.zeros_like(ldps[0])
        for i in range(ldps.shape[0]):
            num += ldps[i] * ldps[i, ..., 3:4] * weights[i, None, :, :, None]
            denom += ldps[i, ..., 3:4] * weights[i, None, :, :, None]
        final_ldp = divide_safe(num, denom)

    else:
        final_ldp = divide_safe(torch.sum(ldps * ldps[..., 3:4] * weights[:, None, :, :, None], 0),
                torch.sum(ldps[..., 3:4] * weights[:, None, :, :, None], 0))
    return final_ldp