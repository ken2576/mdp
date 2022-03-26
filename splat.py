import torch
import numpy as np

from projector import meshgrid_pinhole, divide_safe, cartesian_to_polar
from homography import transform_points

def pad_intrinsic(intrinsic):
    '''Pad 3x3 intrinsic matrix to 4x4

    Args:
        intrinsic: intrinsic matrix [..., 3, 3]
    Returns:
        Padded intrinsic matrix [..., 4, 4]
    '''
    orig_shape = intrinsic.shape[:-2]
    padded = torch.zeros(orig_shape + (4, 4,)).to(intrinsic.device)
    padded[..., :3, :3] = intrinsic
    padded[..., 3, 3] = 1.0

    return padded

def transform_matrix(src_int, tgt_int, src_ext, tgt_ext):
    '''Compute transformation matrix

    Args:
        src_int: source intrinsic [batch, 3, 3]
        tgt_int: target intrinsic [batch, 3, 3]
        src_ext: source extrinsic [batch, 4, 4]
        tgt_ext: target extrinsic [batch, 4, 4]
    Returns:
        Homography transformation matrix [batch, 4, 4]
    '''
    src_int_prime = pad_intrinsic(src_int)
    tgt_int_prime = pad_intrinsic(tgt_int)
    transform = torch.matmul(tgt_ext, torch.inverse(src_ext))
    return torch.matmul(
        tgt_int_prime, torch.matmul(transform, torch.inverse(src_int_prime))
    )

def batch_scatter_add(init, indices, updates):
    '''Add sparse updates to a tensor

    Args:
        init: initilization of the output tensor [batch, #points]
        indices: a tensor of indices into the first dimension of init [batch, #updates]
        updates: a tensor of values to be added to init [batch, #updates]
    Returns:
        A new tensor updated from init [batch, #points]
    '''
    batch, num_points = init.shape
    _, num_updates = indices.shape
    init = init.view([-1])
    col = torch.linspace(0, batch-1, batch, device=init.device).view([batch, -1]).int()
    col = col * num_points
    offset = col.expand([batch, num_updates])
    new_indices = indices + offset
    new_indices = new_indices.reshape([-1]).long()
    updates = updates.reshape([-1])
    output = init.index_add(0, new_indices, updates)
    return output.view([batch, num_points])
 

def splat(src_im, tgt_coords, init_tgt_im):
    '''Splat pixels from source image to target coordinates on
       the target image
       Implemented as https://github.com/google/layered-scene-inference/blob/59b5d37022f6aaab30dfd4ddcf560923eaf38578/lsi/geometry/sampling.py#L171

    Args:
        src_im: source image [batch, src_height, src_width, #channels]
        tgt_coords: target coordinates [batch, src_height, src_width, 2]
        init_tgt_im: initial target image [batch, tgt_height, tgt_width, #channels]
    Returns:
        Splatted target image
    '''
    coords = tgt_coords - 0.5
    b, src_h, src_w, c = src_im.shape
    _, tgt_h, tgt_w, _ = init_tgt_im.shape

    src_pixel_num = src_h * src_w
    tgt_pixel_num = tgt_h * tgt_w
    x = coords[..., 0]
    y = coords[..., 1]

    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    y_max = tgt_h - 1.0
    x_max = tgt_w - 1.0

    x0_safe = torch.clamp(x0, 0.0, x_max)
    y0_safe = torch.clamp(y0, 0.0, y_max)
    x1_safe = torch.clamp(x1, 0.0, x_max)
    y1_safe = torch.clamp(y1, 0.0, y_max)

    # Compute bilinear splat weights
    wt_x0 = (x1 - x) * torch.eq(x0, x0_safe).float()   
    wt_x1 = (x - x0) * torch.eq(x1, x1_safe).float()
    wt_y0 = (y1 - y) * torch.eq(y0, y0_safe).float()
    wt_y1 = (y - y0) * torch.eq(y1, y1_safe).float()

    wt_tl = wt_x0 * wt_y0
    wt_tr = wt_x1 * wt_y0
    wt_bl = wt_x0 * wt_y1
    wt_br = wt_x1 * wt_y1

    # Clamp small weights as indicated by the original LSI paper
    eps = 1e-3
    wt_tl *= torch.gt(wt_tl, eps).float()
    wt_tr *= torch.gt(wt_tr, eps).float()
    wt_bl *= torch.gt(wt_bl, eps).float()
    wt_br *= torch.gt(wt_br, eps).float()

    val_tl = (src_im * wt_tl[..., None]).view([b, src_pixel_num, c])
    val_tr = (src_im * wt_tr[..., None]).view([b, src_pixel_num, c])
    val_bl = (src_im * wt_bl[..., None]).view([b, src_pixel_num, c])
    val_br = (src_im * wt_br[..., None]).view([b, src_pixel_num, c])

    idx_tl = (x0_safe + y0_safe * tgt_w).view([b, -1]).int()
    idx_tr = (x1_safe + y0_safe * tgt_w).view([b, -1]).int()
    idx_bl = (x0_safe + y1_safe * tgt_w).view([b, -1]).int()
    idx_br = (x1_safe + y1_safe * tgt_w).view([b, -1]).int()

    init_tgt_im = init_tgt_im.view([b, tgt_pixel_num, c])

    rendered = []
    for c_idx in range(c):
        curr_im = init_tgt_im[..., c_idx]
        curr_im = batch_scatter_add(curr_im, idx_tl, val_tl[..., c_idx])
        curr_im = batch_scatter_add(curr_im, idx_tr, val_tr[..., c_idx])
        curr_im = batch_scatter_add(curr_im, idx_bl, val_bl[..., c_idx])
        curr_im = batch_scatter_add(curr_im, idx_br, val_br[..., c_idx])
        rendered.append(curr_im)
    rendered = torch.stack(rendered, 2)
    return rendered.view([b, tgt_h, tgt_w, c])

def project_points(coord, depth_map, inv_src_int, tgt_int, inv_src_ext, tgt_ext):
    '''Project point from source to target
    
    Args:
        coord: pixel coordinates to warp [height, width, 3]
        depth_map: per-pixel depth map [batch, height, width]
        inv_src_int: source intrinsic [3, 3]
        tgt_int: target intrinsic [3, 3]
        src_ext: source extrinsic [4, 4]
        tgt_ext: target extrinsic [4, 4]
    Returns:
        Warped pixel coordinates [batch, height, width, 2]
    '''
    b, h, w = depth_map.shape
    flattened_depth = depth_map.reshape([b, h*w])
    pix_coord = coord.view([h*w, 3, 1])
    cam_coord = torch.matmul(inv_src_int, pix_coord)
    cam_coord = cam_coord * flattened_depth[..., None, None]
    ones = torch.ones_like(flattened_depth).unsqueeze(2).unsqueeze(2)
    cam_coord = torch.cat([cam_coord, ones], 2)
    world_coord = torch.matmul(inv_src_ext, cam_coord)
    new_cam_coord = torch.matmul(tgt_ext, world_coord)
    pix_coord = torch.matmul(tgt_int, new_cam_coord[:, :, :3])
    pix_coord = pix_coord.squeeze(dim=3).view([b, h, w, 3])
    pix_coord = divide_safe(pix_coord[..., :2], pix_coord[..., 2:3])
    return pix_coord

def project_depth(coord, depth_map, inv_src_int, tgt_int, inv_src_ext, tgt_ext):
    '''Project depth from source to target

    Args:
        coord: pixel coordinates to warp [height, width, 3]
        depth_map: per-pixel depth map [batch, height, width]
        inv_src_int: source intrinsic [3, 3]
        inv_src_ext: source extrinsic [4, 4]
        tgt_ext: target extrinsic [4, 4]
    Returns:
        Warped depth [batch, height, width]
    '''

    b, h, w = depth_map.shape
    flattened_depth = depth_map.reshape([b, h*w])
    pix_coord = coord.view([h*w, 3, 1])
    cam_coord = torch.matmul(inv_src_int, pix_coord)
    cam_coord = cam_coord * flattened_depth[..., None, None]
    ones = torch.ones_like(flattened_depth).unsqueeze(2).unsqueeze(2)
    cam_coord = torch.cat([cam_coord, ones], 2)

    world_coord = torch.matmul(inv_src_ext, cam_coord)
    new_cam_coord = torch.matmul(tgt_ext, world_coord)

    pix_coord = torch.matmul(tgt_int, new_cam_coord[:, :, :3])
    pix_coord = pix_coord.squeeze(dim=3).view([b, h, w, 3])
    warped_depth = pix_coord[..., 2]

    return warped_depth

def project_points_spherical(coord, depth_map, inv_src_int, tgt_h, tgt_w, inv_src_ext, tgt_ext):
    '''Project point from source to target on pano
    
    Args:
        coord: pixel coordinates to warp [height, width, 3]
        depth_map: per-pixel depth map [batch, height, width]
        inv_src_int: source intrinsic [3, 3]
        tgt_h: target height
        tgt_w: target width
        src_ext: source extrinsic [4, 4]
        tgt_ext: target extrinsic [4, 4]
    Returns:
        Warped pixel coordinates [batch, height, width, 2]
    '''
    b, h, w = depth_map.shape
    flattened_depth = depth_map.reshape([b, h*w])
    pix_coord = coord.view([h*w, 3, 1])
    cam_coord = torch.matmul(inv_src_int, pix_coord)
    cam_coord = cam_coord * flattened_depth[..., None, None]
    ones = torch.ones_like(flattened_depth).unsqueeze(2).unsqueeze(2)
    cam_coord = torch.cat([cam_coord, ones], 2)
    world_coord = torch.matmul(inv_src_ext, cam_coord)
    new_cam_coord = torch.matmul(tgt_ext, world_coord)
    
    pix_coord = cartesian_to_polar(new_cam_coord.squeeze(dim=3))
    pix_coord[..., 0] = (pix_coord[..., 0] + 1.) / 2. * tgt_w
    pix_coord[..., 1] = (pix_coord[..., 1] + 1.) / 2. * tgt_h
    pix_coord = pix_coord.view([b, h, w, 2])
    return pix_coord

def project_depth_spherical(coord, depth_map, inv_src_int, inv_src_ext, tgt_ext):
    '''Project depth from source to target on pano

    Args:
        coord: pixel coordinates to warp [height, width, 3]
        depth_map: per-pixel depth map [batch, height, width]
        inv_src_int: source intrinsic [3, 3]
        inv_src_ext: source extrinsic [4, 4]
        tgt_ext: target extrinsic [4, 4]
    Returns:
        Warped depth [batch, height, width]
    '''

    b, h, w = depth_map.shape
    flattened_depth = depth_map.reshape([b, h*w])
    pix_coord = coord.view([h*w, 3, 1])
    cam_coord = torch.matmul(inv_src_int, pix_coord)
    cam_coord = cam_coord * flattened_depth[..., None, None]
    ones = torch.ones_like(flattened_depth).unsqueeze(2).unsqueeze(2)
    cam_coord = torch.cat([cam_coord, ones], 2)

    world_coord = torch.matmul(inv_src_ext, cam_coord)
    new_cam_coord = torch.matmul(tgt_ext, world_coord)
    warped_depth = torch.sqrt(torch.sum(new_cam_coord[..., :3, :] ** 2, 2))
    warped_depth = warped_depth.view([b, h, w])

    return warped_depth

def pano_to_pinhole_points(cam_coord, depth_map, tgt_int, inv_src_ext, tgt_ext):
    '''Project point from source to target on pano
    
    Args:
        cam_coord: spherical pixel coordinates to warp [height, width, 3]
        depth_map: per-pixel depth map [batch, height, width]
        tgt_int: target intrinsic [3, 3]
        inv_src_ext: inverse source extrinsic [4, 4]
        tgt_ext: target extrinsic [4, 4]
    Returns:
        Warped pixel coordinates [batch, height, width, 2]
    '''
    b, h, w = depth_map.shape
    cam_coord = cam_coord.view([h*w, 3, 1])
    flattened_depth = depth_map.reshape([b, h*w])
    cam_coord = cam_coord * flattened_depth[..., None, None]
    ones = torch.ones_like(flattened_depth).unsqueeze(2).unsqueeze(2)
    cam_coord = torch.cat([cam_coord, ones], 2)
    world_coord = torch.matmul(inv_src_ext, cam_coord)
    new_cam_coord = torch.matmul(tgt_ext, world_coord)

    pix_coord = torch.matmul(tgt_int, new_cam_coord[:, :, :3])
    pix_coord = pix_coord.squeeze(dim=3).view([b, h, w, 3])
    zeros = torch.zeros_like(pix_coord[..., 2:3])
    pix_coord[..., 2:3] = torch.where(pix_coord[..., 2:3] > 0, pix_coord[..., 2:3], zeros)
    pix_coord = divide_safe(pix_coord[..., :2], pix_coord[..., 2:3])

    return pix_coord

def pano_to_pinhole_depth(cam_coord, depth_map, inv_src_ext, tgt_ext):
    '''Project depth from source to target on pano

    Args:
        cam_coord: pixel coordinates to warp [height, width, 3]
        depth_map: per-pixel depth map [batch, height, width]
        inv_src_ext: source extrinsic [4, 4]
        tgt_ext: target extrinsic [4, 4]
    Returns:
        Warped depth [batch, height, width]
    '''

    b, h, w = depth_map.shape
    cam_coord = cam_coord.view([h*w, 3, 1])
    flattened_depth = depth_map.reshape([b, h*w])
    cam_coord = cam_coord * flattened_depth[..., None, None]
    ones = torch.ones_like(flattened_depth).unsqueeze(2).unsqueeze(2)
    cam_coord = torch.cat([cam_coord, ones], 2)
    world_coord = torch.matmul(inv_src_ext, cam_coord)
    new_cam_coord = torch.matmul(tgt_ext, world_coord)

    warped_depth = new_cam_coord[..., 2, 0].view([b, h, w])

    return warped_depth

def splat_image(im, disparity, transformed, max_disparity, tgt_im=None, scale=25):
    """Splat an image to the transformed coordinate and do soft z-buffering

    Args:
        im: image to be splatted [batch, height, width, #channels]
        disparity: disparity map for splatting [batch, height, width]
        transformed: transformed coordinates [batch, height, width, 2]
    Returns:
        Splatted image viewed from the novel viewpoint
    """
    
    # Splat RGB
    if max_disparity.ndim == 1:
        normalized_disp = disparity - max_disparity[:, None, None]
    else:
        normalized_disp = disparity - max_disparity
    z_buffer_weights = torch.exp(normalized_disp * scale)[..., None]
    w_im = im * z_buffer_weights
    if tgt_im is None:
        tgt_im = torch.zeros_like(w_im)
    splatted = splat(w_im, transformed, tgt_im)
    
    weights = torch.ones_like(w_im) * z_buffer_weights
    splatted_weights = torch.zeros_like(tgt_im)
    splatted_weights = splat(weights, transformed, splatted_weights)
 
    splatted = divide_safe(splatted, splatted_weights)

    return splatted

def splat_images(imgs, disparity, transformed, max_disparity, tgt_im=None, scale=25):
    """Splat multiple images to the transformed coordinate and do soft z-buffering

    Args:
        imgs: images to be splatted [#views, #planes, height, width, #channels]
        disparity: disparity maps for splatting [#views, #planes, height, width]
        transformed: transformed coordinates [#views, #planes, height, width, 2]
    Returns:
        Splatted image viewed from the novel viewpoint [#channels, height, width]
    """
    # Splat RGB
    z_buffer_weights = torch.exp((disparity - max_disparity) * scale)[..., None]
    w_imgs = imgs * z_buffer_weights
    if tgt_im is None:
        tgt_im = torch.zeros_like(w_imgs[0])

    for idx, (w_im, new_coord) in enumerate(zip(w_imgs, transformed)):
        if idx == 0:
            splatted = splat(w_im, new_coord, tgt_im)
        else:
            splatted = splat(w_im, new_coord, splatted)

    weights = torch.ones_like(w_imgs) * z_buffer_weights
    splatted_weights = torch.zeros_like(tgt_im)
    for w, new_coord in zip(weights, transformed):
        splatted_weights = splat(w, new_coord, splatted_weights)

    return divide_safe(splatted, splatted_weights)

def splat_image_weights(im, disparity, transformed, max_disparity, tgt_im=None, tgt_wim=None, scale=25):
    """Splat an image to the transformed coordinate and do soft z-buffering

    Args:
        im: image to be splatted [batch, height, width, #channels]
        disparity: disparity map for splatting [batch, height, width]
        transformed: transformed coordinates [batch, height, width, 2]
    Returns:
        Splatted RGB and weights
    """
    
    # Splat RGB
    z_buffer_weights = torch.exp((disparity - max_disparity) * scale)[..., None]
    w_im = im * z_buffer_weights
    if tgt_im is None:
        tgt_im = torch.zeros_like(w_im)
    splatted = splat(w_im, transformed, tgt_im)
    
    # Splat weights
    weights = torch.ones_like(w_im) * z_buffer_weights
    if tgt_wim is None:
        tgt_wim = torch.zeros_like(tgt_im)
    splatted_weights = splat(weights, transformed, tgt_wim)

    return splatted, splatted_weights