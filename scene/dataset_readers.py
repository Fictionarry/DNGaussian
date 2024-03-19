#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

import torch

class CameraInfo(NamedTuple):
    uid: str
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth_mono: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    eval_cameras: list
    nerf_normalization: dict
    ply_path: str


def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported but found {}!".format(intr.model)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # monocular depth
        ## (fix path)
        depth_mono_path = os.path.join('/'.join(images_folder.split("/")[:-1]), 'depth_maps', 'depth_' + os.path.basename(extr.name).split(".")[0] + '.png')
        depth_mono = Image.open(depth_mono_path)

        cam_info = CameraInfo(uid=image_path, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth_mono=depth_mono,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, dataset, eval, rand_pcd, mvs_pcd, llffhold=8, N_sparse=-1):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    # print(cam_intrinsics)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        print("Dataset Type: ", dataset)
        if dataset == "LLFF":
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            eval_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
            if N_sparse > 0:
                idx = list(range(len(train_cam_infos)))
                idx_train = np.linspace(0, len(train_cam_infos) - 1, N_sparse)
                idx_train = [round(i) for i in idx_train]
                idx_test = [i for i in idx if i not in idx_train] 
                train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_train]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in idx_test] + eval_cam_infos

                # print('train', idx_train)
                # print('test', train_cam_infos)
            # else:
            #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        elif dataset == "DTU":
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
            test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
            if N_sparse > 0:
                train_idx = train_idx[:N_sparse]
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]
            eval_cam_infos = test_cam_infos
        else:
            raise NotImplementedError
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        eval_cam_infos = []

    print('train', [info.image_path for info in train_cam_infos])
    print('eval', [info.image_path for info in eval_cam_infos])

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if rand_pcd and mvs_pcd:
        print("[warning] Both --rand_pcd and --mvs_pcd are detected, use --mvs_pcd.")
        rand_pcd = False

    if rand_pcd:
        print('Init random point cloud.')
        ply_path = os.path.join(path, "sparse/0/points3D_random.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")

        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        print(xyz.max(0), xyz.min(0))


        if dataset == "LLFF":
            pcd_shape = (topk_(xyz, 1, 0)[-1] + topk_(-xyz, 1, 0)[-1])
            num_pts = int(pcd_shape.max() * 50)
            xyz = np.random.random((num_pts, 3)) * pcd_shape * 1.3 - topk_(-xyz, 20, 0)[-1]
        elif dataset == "DTU":
            pcd_shape = (topk_(xyz, 100, 0)[-1] + topk_(-xyz, 100, 0)[-1])
            num_pts = 10_00
            xyz = np.random.random((num_pts, 3)) * pcd_shape * 1.3 - topk_(-xyz, 100, 0)[-1] # - 0.15 * pcd_shape
        print(pcd_shape)
        print(f"Generating random point cloud ({num_pts})...")

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    elif mvs_pcd:
        ply_path = os.path.join(path, "3_views/dense/fused.ply")
        assert os.path.exists(ply_path)
        pcd = fetchPly(ply_path)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           eval_cameras=eval_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            # monocular depth
            ## (fix path)
            depth_mono_path = os.path.join('/'.join(image_path.split("/")[:-1]), 'depth_maps', 'depth_' + image_name + '.png')
            depth_mono = Image.open(depth_mono_path)


            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth_mono=depth_mono,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, rand_pcd, llffhold=8, N_sparse=-1, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if eval:
        if N_sparse > 0:
            train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in [2, 16, 26, 55, 73, 76, 86, 93]]
        eval_cam_infos = [c for idx, c in enumerate(test_cam_infos) if idx % llffhold == 0]
        test_cam_infos = test_cam_infos
    else:
        test_cam_infos = []
        eval_cam_infos = []


    print('train', [info.image_path for info in train_cam_infos])
    print('eval', [info.image_path for info in eval_cam_infos])

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if rand_pcd:
        print('Init random point cloud.')
    if rand_pcd or not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2 - 1
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           eval_cameras=eval_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info




def generateLLFFCameras(poses):
    cam_infos = []
    Rs, tvecs, height, width, focal_length_x = pose_utils.convert_poses(poses) 
    # print(Rs, tvecs, height, width, focal_length_x)
    for idx, _ in enumerate(Rs):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(Rs)))
        sys.stdout.flush()

        uid = idx
        R = np.transpose(Rs[idx])
        T = tvecs[idx]

        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None, depth_mono=None, 
                              image_path=None, image_name=None, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


from utils import pose_utils


def CreateLLFFSpiral(basedir):

    # Load poses and bounds.
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses_o = poses_arr[:, :-2].reshape([-1, 3, 5])
    bounds = poses_arr[:, -2:]
    
    # Pull out focal length before processing poses.
    # Correct rotation matrix ordering (and drop 5th column of poses).
    fix_rotation = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
                            dtype=np.float32)
    inv_rotation = np.linalg.inv(fix_rotation)
    poses = poses_o[:, :3, :4] @ fix_rotation

    # Rescale according to a default bd factor.
    # scale = 1. / (bounds.min() * .75)
    # poses[:, :3, 3] *= scale
    # bounds *= scale

    # Recenter poses.
    render_poses = pose_utils.recenter_poses(poses)

    # Separate out 360 versus forward facing scenes.
    render_poses = pose_utils.generate_spiral_path(
          render_poses, bounds, n_frames=180)
    render_poses = pose_utils.backcenter_poses(render_poses, poses)
    render_poses = render_poses @ inv_rotation
    render_poses = np.concatenate([render_poses, np.tile(poses_o[:1, :3, 4:], (render_poses.shape[0], 1, 1))], -1)

    render_cam_infos = generateLLFFCameras(render_poses.transpose([1,2,0]))

    nerf_normalization = getNerfppNorm(render_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=None,
                           test_cameras=render_cam_infos,
                           eval_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=None)
    return scene_info



def CreateDTUSpiral(basedir):

    # Load poses and bounds.
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses_o = poses_arr[:, :-2].reshape([-1, 3, 5])
    bounds = poses_arr[:, -2:]
    
    # Pull out focal length before processing poses.
    # Correct rotation matrix ordering (and drop 5th column of poses).
    fix_rotation = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
                            dtype=np.float32)
    inv_rotation = np.linalg.inv(fix_rotation)
    poses = poses_o[:, :3, :4] @ fix_rotation

    # for i in range(len(poses)):
    #     poses[i][:3, :3] = poses[i][:3, :3].transpose()

    render_poses = pose_utils.recenter_poses(poses)

    s = np.max(np.abs(render_poses[:, :3, -1]))
    render_poses[:, :3, -1] /= s

    # Separate out 360 versus forward facing scenes.
    render_poses = pose_utils.generate_spiral_path_dtu(
          render_poses, n_frames=180)
    
    render_poses[:, :3, -1] *= s
    render_poses = pose_utils.backcenter_poses(render_poses, poses)
    # for i in range(len(render_poses)):
    #     render_poses[i][:3, :3] = render_poses[i][:3, :3].transpose()

    render_poses = render_poses @ inv_rotation
    render_poses = np.concatenate([render_poses, np.tile(poses_o[:1, :3, 4:], (render_poses.shape[0], 1, 1))], -1)

    render_cam_infos = generateLLFFCameras(render_poses.transpose([1,2,0]))

    nerf_normalization = getNerfppNorm(render_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=None,
                           test_cameras=render_cam_infos,
                           eval_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=None)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    # "LLFF" : readLLFFInfo,
    "Spiral" : CreateLLFFSpiral,
    "SpiralDTU" : CreateDTUSpiral,
}