import numpy as np
import cv2
import torch
import os
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import json
import trimesh
import glob
import PIL
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

image_size = 384
trans_totensor = transforms.Compose([
    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR)
])

out_path_prefix = 'data/scannet'
data_root = '/root/picasso/yxj/datasets/scannet/'
scenes = ['scene0710_00']
out_names = ['scan2']

for scene, out_name in zip(scenes, out_names):
    out_path = os.path.join(out_path_prefix, out_name)
    os.makedirs(out_path, exist_ok=True)
    print(out_path)

    folders = ["image", "mask"]
    for folder in folders:
        out_folder = os.path.join(out_path, folder)
        os.makedirs(out_folder, exist_ok=True)

    with open(os.path.join(data_root, scene, 'transforms_test.json'), 'r') as f:
        data_dict = json.load(f)
    frames = data_dict['frames']

    color_paths = [os.path.join(data_root, scene, frame['file_path']) for frame in frames]
    print(color_paths)

    intrinsics = []
    poses = []
    for frame in frames:
        intrinsic = np.eye(4)
        intrinsic[0, 0] = frame['fx']
        intrinsic[1, 1] = frame['fy']
        intrinsic[0, 2] = frame['cx']
        intrinsic[1, 2] = frame['cy']
        pose = frame['transform_matrix']

        intrinsics.append(intrinsic[:3])
        poses.append(pose)

    # deal with invalid poses
    poses = np.array(poses)
    min_vertices = poses[:, :3, 3].min(axis=0)
    max_vertices = poses[:, :3, 3].max(axis=0)
 
    center = (min_vertices + max_vertices) / 2.
    scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
    print(center, scale)

    # we should normalized to unit cube
    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -center
    scale_mat[:3 ] *= scale 
    scale_mat = np.linalg.inv(scale_mat)

    # copy image
    cameras = {}
    pcds = []
    H, W = 468, 624

    camera_intrinsic = intrinsics[0]
    # resize
    resize_scale = image_size / H
    camera_intrinsic[0] *= resize_scale
    camera_intrinsic[1] *= resize_scale
    H, W = H * resize_scale, W * resize_scale
    # # center crop by image_size
    # offset_x = (W - image_size) * 0.5
    # offset_y = (H - image_size) * 0.5
    # camera_intrinsic[0, 2] -= offset_x
    # camera_intrinsic[1, 2] -= offset_y
    
    K = camera_intrinsic
    print(K)
    
    for idx, (pose, image_path) in enumerate(zip(poses, color_paths)):        
        out_index = os.path.splitext(os.path.basename(image_path))[0]
        target_image = os.path.join(out_path, "image/%s_rgb.png"%(out_index))
        print(target_image)
        img = Image.open(image_path)
        img_tensor = trans_totensor(img)
        img_tensor.save(target_image)

        mask = (np.ones((image_size, image_size, 3)) * 255.).astype(np.uint8)

        target_image = os.path.join(out_path, "mask/%s_mask.png"%(out_index))
        cv2.imwrite(target_image, mask)
        
        # save pose
        pcds.append(pose[:3, 3])
        pose = K @ np.linalg.inv(pose)
        
        #cameras["scale_mat_%d"%(out_index)] = np.eye(4).astype(np.float32)
        cameras["scale_mat_%s"%(out_index)] = scale_mat
        cameras["world_mat_%s"%(out_index)] = pose

    #np.savez(os.path.join(out_path, "cameras_sphere.npz"), **cameras)
    np.savez(os.path.join(out_path, "cameras.npz"), **cameras)
