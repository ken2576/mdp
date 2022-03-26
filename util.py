import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

def create_folder(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            print("Directory creation failed at %s" % path)
        else:
            print("Directory created for %s" % path)

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, default=default)

def save_images(volume, folder):
    '''Save volume (B, C, H, W) to images
    '''
    for idx, im in enumerate(volume):
        tmp = im.permute([1, 2, 0]) * 255.
        tmp = tmp.cpu().detach().numpy().astype(np.uint8)
        plt.imsave(folder + str(idx) + '.png', tmp)

def save_rgba(volume, folder):
    '''Save rgba volume to images
    (D, B, H, W, C)
    '''
    for idx, im in enumerate(volume):
        rgb = im[..., :3].permute([0, 3, 1, 2]) * 255.
        rgb = rgb.permute([0, 2, 3, 1])[0]
        rgb = rgb.cpu().detach().numpy().astype(np.uint8)
        plt.imsave(folder + 'rgb_' + str(idx) + '.png', rgb)
        alpha = im[..., 3].squeeze()
        alpha = alpha.cpu().detach().numpy()
        plt.imsave(folder + 'alpha_' + str(idx) + '.png', alpha)

def save_psv(volume, folder, n_views):
    '''Save plane sweep volume to images
    '''
    b, c, d, h, w = volume.shape
    psv = volume.permute([2, 0, 1, 3, 4])
    for idx, im in enumerate(psv):
        for v in range(n_views):
            tmp = im[:, 3*v:3*(v+1)] * 255.
            tmp = tmp.permute([0, 2, 3, 1])[0]
            tmp = tmp.cpu().detach().numpy().astype(np.uint8)
            plt.imsave(folder + str(v) + '_' + str(idx) + '.png', tmp)
        
        combine = im.view([b, n_views, int(c/n_views), h, w])
        tmp = torch.mean(combine, 1) * 255.
        tmp = tmp.permute([0, 2, 3, 1])[0]
        tmp = tmp.cpu().detach().numpy().astype(np.uint8)
        plt.imsave(folder + 'combined_' + str(idx) + '.png', tmp)

def save_image(im, path):
    '''Save image (1 channel)
    '''
    tmp = im.cpu().numpy()
    plt.imsave(path, tmp)

def save_disp(rgba_layers, disp, path):
    '''Save alpha-composited depth
    '''
    d = rgba_layers.shape[0]
    for idx in range(d):
        alpha = rgba_layers[d-1-idx, 0, ..., 3]
        if idx == 0:
            output = disp[d-1-idx]
        else:
            disp_by_alpha = disp[d-1-idx] * alpha
            output = disp_by_alpha + output * (1.0 - alpha)
    plt.imsave(path + 'disp.png', output.detach().cpu().numpy())

def denormalize(y, std=[0.229, 0.224, 0.225],
    mean=[0.485, 0.456, 0.406]):
    x = y.new(*y.size())
    x[:, 0, :, :] = y[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = y[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = y[:, 2, :, :] * std[2] + mean[2]
    return x

def bilinear_wrapper(imgs, coords):
    '''Wrapper around bilinear sampling function
    
    Args:
        imgs: images to sample [..., height_s, width_s, #channels]
        coords: pixel location to sample from [..., height_t, width_t, 2]
    Returns:
        Sampled images [..., height_t, width_t, #channels]
    '''
    init_dims = imgs.shape[:-3:]
    end_dims_img = imgs.shape[-3::]
    end_dims_coords = coords.shape[-3::]
    prod_init_dims = init_dims[0]
    for i in range(1, len(init_dims)):
        prod_init_dims *= init_dims[i]
    src = imgs.contiguous().view((prod_init_dims,) + end_dims_img)
    coords = coords.view((prod_init_dims,) + end_dims_coords)
    src = src.permute([0, 3, 1, 2])
    tgt = F.grid_sample(src, coords)
    tgt = tgt.permute([0, 2, 3, 1])
    tgt = tgt.view(init_dims + tgt.shape[-3::])
    return tgt