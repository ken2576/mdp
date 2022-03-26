from util import read_json
import numpy as np
from numpy import sin, cos

def rot2pyr(rot):
    corr = np.array([
        [0, 0, -1],
        [1, 0, 0],
        [0, 1, 0.],
    ])
    
    rot = np.matmul(rot, np.linalg.inv(corr))

    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    
    yaw = yaw / np.pi * 180 % 360
    pitch = pitch / np.pi * 180 % 360
    roll = roll / np.pi * 180 % 360
     
    return np.array([pitch, yaw, roll])

def llff2unreal(pose, fix_data=False):
    
    if fix_data:
        hwf = pose[:, 4] / 2. # FIXED YOUR GODDAMN DATA
    else:
        hwf = pose[:, 4]
    pose = np.concatenate([pose[:, 1:2], -pose[:, :1], pose[:, 2:]], 1)
    pose = pose[:, :-1]
    pose[2, :] *= -1
    rot = pose[:, :-1]
    trans = pose[:, -1]
    
    pyr = rot2pyr(rot)
    
    return pyr, trans, hwf  

def pyr2rotmat(p, y, r):
    p = np.array([
        [np.cos(p), 0, -np.sin(p)],
        [        0, 1,          0],
        [np.sin(p), 0,  np.cos(p)],
    ])
    y = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y),  np.cos(y), 0],
        [        0,          0, 1],
    ])
    r = np.array([
        [1,          0,         0],
        [0,  np.cos(r), np.sin(r)],
        [0, -np.sin(r), np.cos(r)],
    ])
    R = np.matmul(r, np.matmul(y, p))
   
    # curr is x y z but we want y z x

    corr = np.array([
        [0, 0, -1],
        [1, 0, 0],
        [0, 1, 0.],
    ])
   
    return np.matmul(R, corr)

def unreal2sidewinder(pyr, trans, hwf):

    def pitch_mat(pitch):
        b = pitch / 180 * np.pi
        return np.array([[cos(b), 0, -sin(b)],
                         [0,      1,       0],
                         [sin(b), 0,  cos(b)]])
        
    def yaw_mat(yaw):
        b = yaw / 180 * np.pi
        return np.array([[cos(b),  sin(b), 0],
                         [-sin(b), cos(b), 0],
                         [     0,       0, 1]])
    
    def roll_mat(roll):
        b = roll / 180 * np.pi
        return np.array([[1,      0,       0],
                         [0, cos(b),  sin(b)],
                         [0,-sin(b),  cos(b)]])
    
    def rpy2rot(pitch, yaw, roll):
        rotation = np.matmul(pitch_mat(pitch), yaw_mat(yaw))
        rotation = np.matmul(roll_mat(roll), rotation)
        return rotation
    
    # c2w rotation
    rot = rpy2rot(*pyr)

    # c2w matrix
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = - np.matmul(rot, trans)

    fix = np.array([[0, 1, 0, 0],
                    [0, 0, -1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])

    mat = fix.dot(mat)

    mat = mat[:3, :]

    mat = np.concatenate([mat, hwf.reshape([3, 1])], 1)

    return mat

def llff2sidewinder(pose, fix_data):
    pyr = llff2unreal(pose, fix_data)
    new_pose = unreal2sidewinder(*pyr)
    return new_pose