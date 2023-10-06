import json
import math
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import plotly.express as px
import torch
from PIL import Image


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    return K / K[2, 2], R, (t[:3] / t[3])[:, 0]


def run():
    camera_dict = np.load("cameras_sphere.npz")

    world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(49)]
    scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(49)]

    intrinsics_all = []
    pose_all = []

    for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, R, T = load_K_Rt_from_P(None, P)
        intrinsics_all.append(intrinsics)
        pose_all.append((R, T))

    print(intrinsics_all[0])

    points = []
    for r, t in pose_all:
        R = r
        T = t

        R = R[:, [2, 1, 0]]
        T[0], T[2] = T[2], T[0]

        vec = np.array([0, 0, 2])
        rot_vec = np.matmul(vec, R)

        points.append(T)

        num_points = 100
        for _ in range(num_points):
            points.append(points[-1] + rot_vec / num_points)

    points = np.array(points)
    fig = px.scatter_3d(x=points[:, 0], y=points[:, 1], z=points[:, 2])
    fig.update_traces(marker_size=1)
    fig.show()


def run2():
    cam_infos = []
    path = os.path.join(os.path.abspath(""), "../data", "nerf_synthetic", "lego_exr_2")
    with open(os.path.join(path, "transforms_train.json")) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + ".png")

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append({"R": R, "T": T, "FovY": FovY, "FovX": FovX, "width": image.size[0], "height": image.size[1]})

    points = []
    for cam in cam_infos:
        R = cam["R"].T
        T = cam["T"]

        mat = np.zeros((4, 4))
        mat[:3, :3] = R
        mat[:3, 3] = T
        mat[3, 3] = 1.0
        mat = np.linalg.inv(mat)
        mat[:3, :3] = mat[:3, :3].transpose()

        R = mat[:3, :3]
        T = mat[:3, 3]

        mat = np.zeros((4, 4))
        mat[3, 3] = 1.0
        mat[:3, :3] = R.transpose()
        mat[:3, 3] = T
        mat = np.linalg.inv(mat)
        assert np.allclose(mat[:3, :3].T, cam["R"])
        assert np.allclose(mat[:3, 3], cam["T"])

        vec = np.array([0, 0, 3])
        rot_vec = np.matmul(vec, R)

        points.append(T)

        num_points = 100
        for _ in range(num_points):
            points.append(points[-1] + rot_vec / num_points)

    points = np.array(points)
    fig = px.scatter_3d(x=points[:, 0], y=points[:, 1], z=points[:, 2])
    fig.update_traces(marker_size=1)
    fig.update_layout(title="JSON")
    fig.show()


if __name__ == "__main__":
    run()
    run2()
