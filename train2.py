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

import math
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
from matplotlib import pyplot as plt
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, bce_loss, depth_smoothness_loss, dbscan_loss, dbscan_loss2
from utils.sh_utils import RGB2SH
from NeuS.fields import SDFNetwork, SingleVarianceNetwork
from NeuS.dataset import Dataset

from simple_knn._C import distCUDA2

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


reso, n_samples = 400, 64

def depth2pcd(depth_image, intrinsic_matrix, R, T):
    fx, fy, cx, cy = intrinsic_matrix

    rows, cols = torch.tensor(range(reso), device='cuda').unsqueeze(-1).expand(reso, reso), torch.tensor(range(reso), device='cuda').expand(reso, reso)

    zs = depth_image
    xs = (cols - cx) * zs / fx
    ys = (rows - cy) * zs / fy
    pcd2 = torch.cat((xs.unsqueeze(-1), ys.unsqueeze(-1), zs.unsqueeze(-1)), dim=-1)

    return torch.matmul(pcd2, torch.tensor(R, device='cuda').float()) + torch.tensor(T, device='cuda')


# Get camera intrinsics (shared)
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def neus_opac(pts, sdf_network, deviation_network):
    batch_size = pts.size(0)

    pts = pts.requires_grad_(True)
    sdf = sdf_network.sdf(pts)

    inv_s = deviation_network().clip(1e-6, 1e6).expand(batch_size, 1)
    sig = (1 + torch.exp(-inv_s * sdf)).pow(-1) + 1e-7
    ldd = (inv_s * torch.exp(-inv_s * sdf)) / ((1 + torch.exp(-inv_s * sdf)).pow(2))
    alpha = F.relu(ldd / sig / inv_s).reshape(batch_size, 1)
    # torch.nan_to_num(alpha)
    # alpha[alpha.isnan()] = 0.0
    # alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)

    # Eikonal loss
    gradients = sdf_network.gradient(pts).squeeze()
    gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, 3), ord=2, dim=-1) - 1.0) ** 2
    pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size)
    relax_inside_sphere = (pts_norm < 1.2).float()
    gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

    return alpha, gradient_error


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device='cuda')
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def up_sample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
    """
    Up sampling give a fixed inv_s
    """
    batch_size, n_samples = z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
    radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
    inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
    sdf = sdf.reshape(batch_size, n_samples)
    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

    # ----------------------------------------------------------------------------------------------------------
    # Use min value of [ cos, prev_cos ]
    # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
    # robust when meeting situations like below:
    #
    # SDF
    # ^
    # |\          -----x----...
    # | \        /
    # |  x      x
    # |---\----/-------------> 0 level
    # |    \  /
    # |     \/
    # |
    # ----------------------------------------------------------------------------------------------------------
    prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device='cuda'), cos_val[:, :-1]], dim=-1)
    cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
    cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
    cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

    dist = (next_z_vals - prev_z_vals)
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([batch_size, 1], device='cuda'), 1. - alpha + 1e-7], -1), -1)[:, :-1]

    z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
    return z_samples

def cat_z_vals(sdf_network, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
    batch_size, n_samples = z_vals.shape
    _, n_importance = new_z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
    z_vals, index = torch.sort(z_vals, dim=-1)

    if not last:
        new_sdf = sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
        sdf = torch.cat([sdf, new_sdf], dim=-1)
        xx = torch.arange(batch_size, device='cuda')[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
        index = index.reshape(-1)
        sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

    return z_vals, sdf

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.load_ply("data/dtu/dtu_scan24/init.ply")
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    sdf_network = SDFNetwork(d_out=257, d_in=3, d_hidden=256, n_layers=8, skip_in=[4], multires=6, bias=0.5, scale=1.0, geometric_init=True, weight_norm=True, ).to("cuda")
    deviation_network = SingleVarianceNetwork(init_val=0.4).to("cuda")
    # sdf_network.eval()
    # deviation_network.eval()

    checkpoint = torch.load(os.path.join(os.path.abspath(""), 'NeuS/checkpoints/ckpt_300000.pth'), map_location="cuda")
    sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
    # deviation_network.load_state_dict(checkpoint['variance_network_fine'])
    neus = True
    if neus:
        optimizer_neus = torch.optim.Adam([*sdf_network.parameters(), *deviation_network.parameters()])

    # neus_dataset = Dataset(data_dir="data/dtu/dtu_scan24/", render_cameras_name="cameras_sphere.npz", object_cameras_name="cameras_sphere.npz")

    # plt.rcParams.update({'font.size': 24})

    dense_voxel_centers = None
    add_cnt = 0

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, deviation_network().clip(1e-6, 1e6), scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            # print("Num cams:", len(viewpoint_stack))

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_mask = (viewpoint_cam.depth > 0.01).to(torch.float32)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, deviation_network().clip(1e-6, 1e6))
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        mask = render_pkg["mask"]

##########################################################################################################################################
        pass

        from scipy.io import savemat

        xyz = gaussians.get_xyz
        scales = gaussians.get_scaling
        rots = gaussians.get_rotation
        sdfs = sdf_network.sdf(xyz).squeeze()
        savemat("gaussian_means_scales_sdf_pruned_1119.mat", {"means": xyz.detach().cpu().numpy(), "scales": scales.detach().cpu().numpy(), "rotations": rots.detach().cpu().numpy(), "sdf": sdfs.detach().cpu().numpy()})

        return

        # if iteration % 700 == 0:
        #     xyz = gaussians.get_xyz
        #     # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.detach().cpu().numpy()))
        #     # o3d.visualization.draw_geometries([pcd])
        #     dists = sdf_network.sdf(xyz)
        #     mask1 = dists.squeeze().abs() > 0.1
        #     gaussians.prune_points(mask1)


            # minn1, idxs = np.ones((400, 400)) * 1000, np.ones((400, 400), dtype=int) * -1
            # depth3 = render_pkg["depth3"].detach().cpu().numpy().squeeze()
            # dists = depth3[1:] - depth3[:-1]
            # for i in range(63-16):
            #     avg_dist = dists[i:i+16].mean(axis=0)
            #     mask1 = np.logical_and(dists[i+16] > 0.00001, avg_dist < minn1)
            #     minn1[mask1] = avg_dist[mask1]
            #     idxs[mask1] = i
            #
            #     # for j in range(i, i+8):
            #     #     avg_dist2 = dists[j:j+8].mean(axis=0)
            #     #     mask2 = np.logical_and(mask1, avg_dist2 < minn2)
            #     #     minn2[mask2] = avg_dist2[mask2]
            # mask3 = np.logical_and(5 > minn1, minn1 > 0.00001)
            # print(minn1[mask3].mean())
            #
            # fx, fy, cx, cy = fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width), fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height), viewpoint_cam.image_width / 2, viewpoint_cam.image_height / 2
            # points = []
            # points2 = []
            # points3 = []
            # for i in tqdm(range(400)):
            #     for j in range(400):
            #         if mask3[i, j] and 3 > depth3[idxs[i, j], i, j] > 2.0 and 4 > depth3[idxs[i, j] + 15, i, j]:
            #             temp = []
            #             points3.append([i, j])
            #             for k in range(idxs[i, j], idxs[i, j] + 16):
            #                 z = depth3[k, i, j]
            #                 x = (j - cx) * z / fx
            #                 y = (i - cy) * z / fy
            #                 points.append([x, y, z])
            #                 temp.append(z)
            #             points2.append(temp)
            # points = np.array(points)
            # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            # # o3d.visualization.draw_geometries([pcd])
            # points2 = np.array(points2)
            # for i in range(10):
            #     ray = np.random.randint(0, points2.shape[0])
            #     print((points2[ray][-1] - points2[ray][0])/16)
            #     plt.scatter(depth3[:, points3[ray][0], points3[ray][1]], np.zeros(64), c='red')
            #     plt.scatter(points2[ray], np.zeros_like(points2[ray]), c='blue')
            #     plt.show()
        if neus:
            # if iteration % 5_000 == 1:
            #     minn1, minn2 = np.ones((400, 400)) * 1000, np.ones((400, 400)) * 1000
            #     depth3 = render_pkg["depth3"].detach().cpu().numpy().squeeze()
            #     dists = depth3[1:] - depth3[:-1]
            #     for i in range(63-16):
            #         avg_dist = dists[i:i+16].mean(axis=0)
            #         mask1 = np.logical_and(dists[i+16] > 0.00001, avg_dist < minn1)
            #         minn1[mask1] = avg_dist[mask1]
            #
            #         # for j in range(i, i+8):
            #         #     avg_dist2 = dists[j:j+8].mean(axis=0)
            #         #     mask2 = np.logical_and(mask1, avg_dist2 < minn2)
            #         #     minn2[mask2] = avg_dist2[mask2]
            #     mask3 = np.logical_and(5 > minn1, minn1 > 0.00001)
            #     print(minn1[mask3].mean())

            # if iteration == 10_000:
            #     points, intr, Rs, Ts = [], [], [], []
            #     all_cams = scene.getTrainCameras().copy()
            #     all_cams.sort(key=lambda x: int(x.image_name))
            #     for viewpoint_cam in all_cams:
            #         render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            #         depth = render_pkg["depth3"].detach().cpu().numpy()
            #         intrinsic_matrix = np.array([fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width), fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height), viewpoint_cam.image_width / 2, viewpoint_cam.image_height / 2])
            #         mat = np.identity(4)
            #         mat[:3, :3] = viewpoint_cam.R.transpose()
            #         mat[:3, 3] = viewpoint_cam.T
            #         mat = np.linalg.inv(mat)
            #
            #         points.append(depth)
            #         intr.append(intrinsic_matrix)
            #         Rs.append(mat[:3, :3].transpose())
            #         Ts.append(mat[:3, 3])
            #
            #     points, intr, Rs, Ts = np.stack(points), np.stack(intr), np.stack(Rs), np.stack(Ts)
            #     np.savez("all_views_gaussian_intersection_points.npz", points=points, intr=intr, Rs=Rs, Ts=Ts)

            if iteration % 1000 == 0:
                n, n2 = 256, 64
                queries = np.concatenate([arr.reshape(-1, 1) for arr in np.meshgrid(*[np.linspace(-1, 1, num=n)] * 3, indexing="ij")], axis=1)
                occ = np.zeros(0)
                for i in range(0, n**3, n2**3):
                    occ_batch = -sdf_network.sdf(torch.tensor(queries[i:i+n2**3], device='cuda').to(torch.float32)).detach().cpu().numpy()
                    occ = np.concatenate((occ, occ_batch.squeeze()), axis=0)
                occ = occ.reshape(n, n, n)
                # occ = -sdf_network.sdf(torch.tensor(queries, device='cuda').to(torch.float32)).detach().cpu().numpy().reshape(n, n, n)
                import mcubes
                vertices, triangles = mcubes.marching_cubes(occ, 0.0)
                vertices = vertices / (n - 1) * 2 - 1
                mesh = trimesh.Trimesh(vertices, triangles)
                mesh.show()
            #     # mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices), triangles=o3d.utility.Vector3iVector(triangles))
            #     # mesh.compute_vertex_normals()
            #
            #     def depth_image_to_point_cloud(depth_image, color_image, intrinsic_matrix, R, T):
            #         fx, fy, cx, cy = intrinsic_matrix
            #         height, width = depth_image.shape
            #
            #         pcd, colors = [], []
            #         for i in range(height):
            #             for j in range(width):
            #                 z = depth_image[i][j]
            #                 if z > 0:
            #                     x = (j - cx) * z / fx
            #                     y = (i - cy) * z / fy
            #                     pcd.append([x, y, z])
            #                     colors.append(color_image[i][j])
            #
            #         pcd = np.array(pcd)
            #         pcd = np.matmul(pcd, R) + T
            #
            #         return pcd, np.array(colors)
            #
            #     intrinsic_matrix = np.array([fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width), fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height), viewpoint_cam.image_width / 2, viewpoint_cam.image_height / 2])
            #
            #     points = np.empty((0, 3))
            #     colors = np.empty((0, 3))
            #     normals = np.empty((0, 3))
            #     all_cams = scene.getTrainCameras().copy()
            #     for viewpoint_cam in tqdm(all_cams, desc="PCD Creation Progress", leave=False):
            #         # Get data
            #         render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            #         color = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
            #         depth = render_pkg["depth2"]
            #         remove_mask = render_pkg["mask"] < 0.95
            #         depth[remove_mask] = 0
            #         depth = depth.squeeze(0).detach().cpu().numpy()
            #
            #         # Get camera extrinsics
            #         mat = np.zeros((4, 4))
            #         mat[:3, :3] = viewpoint_cam.R.transpose()
            #         mat[:3, 3] = viewpoint_cam.T
            #         mat[3, 3] = 1.0
            #         mat = np.linalg.inv(mat)
            #         mat[:3, :3] = mat[:3, :3].transpose()
            #
            #         # Get point cloud from depth and corresponding colors
            #         p, c = depth_image_to_point_cloud(depth, color, intrinsic_matrix, mat[:3, :3], mat[:3, 3])
            #         pcd = o3d.geometry.PointCloud()
            #         pcd.points = o3d.utility.Vector3dVector(p)
            #         pcd.colors = o3d.utility.Vector3dVector(c)
            #         pcd, removed = pcd.remove_statistical_outlier(25, 1.0)
            #
            #         # Correct normals (angle between normal and vector towards camera should be < 90 deg)
            #         pcd.estimate_normals()
            #         norms = np.asarray(pcd.normals)
            #         points_np = np.asarray(pcd.points)
            #         c_vecs = mat[:3, 3] - points_np
            #         c_mags = np.linalg.norm(c_vecs, axis=1)
            #         n_mags = np.linalg.norm(norms, axis=1)
            #         angles = np.degrees(np.arccos((c_vecs * norms).sum(axis=1) / (c_mags * n_mags)))
            #         norms[angles > 90] *= (-1)
            #
            #         # Save data
            #         points = np.concatenate((points, points_np), axis=0)
            #         normals = np.concatenate((normals, norms), axis=0)
            #         colors = np.concatenate((colors, np.asarray(pcd.colors)), axis=0)
            #
            #     # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            #     # labels = np.array(pcd.cluster_dbscan(eps=0.15, min_points=25))
            #     # unique_labels, label_counts = np.unique(labels[labels >= 0], return_counts=True)
            #     # most_common_label = unique_labels[np.argmax(label_counts)]
            #     # mask = (labels == most_common_label)
            #
            #     # Display pcd
            #     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            #     pcd.normals = o3d.utility.Vector3dVector(normals)
            #     pcd.colors = o3d.utility.Vector3dVector(colors)
            #     pcd, _ = pcd.remove_statistical_outlier(100, 1.25)
            #
            #     o3d.visualization.draw_geometries([mesh, pcd])

            if iteration > 1:
                pts = gaussians.get_xyz
                batch_size = pts.size(0)

                sdf = sdf_network.sdf(pts)
                optimizable_tensors = gaussians.replace_tensor_to_optimizer(gaussians.inverse_opacity_activation(sdf), "opacity")
                gaussians._opacity = optimizable_tensors["opacity"]

                gradients = sdf_network.gradient(pts).squeeze()
                gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, 3), ord=2, dim=-1) - 1.0) ** 2
                pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size)
                relax_inside_sphere = (pts_norm < 1.2).float()
                eikonal_loss = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

                render_pkg = render(viewpoint_cam, gaussians, pipe, background, deviation_network().clip(1e-6, 1e6))
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                mask = render_pkg["mask"]

                # def sigmoid_s(x, s):
                #     return 1. / (1 + torch.exp(-x * s))
                #
                # def ldd_s(x, s):
                #     sig = sigmoid_s(x, s)
                #     return s * sig * (1-sig)
                #
                # s = deviation_network()
                #
                # # depth = render_pkg["depth3"][:, 143, 15]
                # # fx, fy, cx, cy = fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width), fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height), viewpoint_cam.image_width / 2, viewpoint_cam.image_height / 2
                # # z = depth
                # # x = (15 - cx) * z / fx
                # # y = (143 - cy) * z / fy
                # # points = torch.stack((x, y, z), dim=-1)
                # #
                # # sdf = sdf_network.sdf(points)
                # #
                # # np.savez("jerry_debug/cuda_data2.npz", sdf=sdf)
                # #
                # # pass
                #
                # density_ray = (1 - sigmoid_s(render_pkg["opac2"][1:], s) / sigmoid_s(render_pkg["opac2"][:-1], s))
                # density_ray[density_ray<0] = 0
                # # density_ray[density_ray < 0] = 1
                # # density_ray[density_ray.argmin(dim=0)-1] = 0
                # density_cuda = render_pkg["depth3"][:-1]
                #
                # # temp = render_pkg["opac2"]
                # # temp[temp==0] = 100
                # # temp = temp.abs()
                # # print((temp[:, (temp<100).sum(dim=0)>10]<0.1).sum(dim=0).to(torch.float32).mean())
                #
                # for i in range(10):
                #     idx2 = np.random.randint(0, 100000)
                #     idx = render_pkg["opac2"].argmin(dim=0).reshape(-1).sort().indices[-idx2]
                #     y = idx // 400
                #     x = idx % 400
                #     # x, y = 15, 143
                #     cont_opac = ldd_s(render_pkg["opac2"][:, y, x], s) / sigmoid_s(render_pkg["opac2"][:, y, x], s) / s
                #
                #     density_cuda[density_cuda<0] = 0
                #
                #     plt.scatter(range(399), density_cuda[:, y, x].detach().cpu().numpy(), c="blue")
                #     plt.scatter(range(399), render_pkg["opac2"][:-1, y, x].detach().cpu().numpy(), c="red")
                #     plt.show()
                #     plt.scatter(render_pkg["opac2"][:-1, y, x].detach().cpu().numpy(), density_cuda[:, y, x].detach().cpu().numpy(), c="blue")
                #     plt.scatter(render_pkg["opac2"][:-1, y, x].detach().cpu().numpy(), cont_opac[:-1].detach().cpu().numpy(), c='red')
                #     plt.xlim([-0.1, 0.1])
                #     plt.show()
                #
                #     pass

            # if iteration % 3000 == 1:
            #     intrinsic_matrix = np.array([fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width), fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height), viewpoint_cam.image_width / 2, viewpoint_cam.image_height / 2])
            #
            #     # Get camera extrinsics
            #     mat = np.zeros((4, 4))
            #     mat[:3, :3] = viewpoint_cam.R.transpose()
            #     mat[:3, 3] = viewpoint_cam.T
            #     mat[3, 3] = 1.0
            #     mat = np.linalg.inv(mat)
            #     mat[:3, :3] = mat[:3, :3].transpose()
            #
            #     depth3 = render_pkg["depth3"]
            #
            #     points = depth2pcd(depth3, intrinsic_matrix, mat[:3, :3], mat[:3, 3]).float()
            #     sdf = sdf_network.sdf(points[:, 200, 200])
            #
            #     plt.scatter(depth3[:, 200, 200].detach().cpu().numpy(), sdf.squeeze().detach().cpu().numpy())
            #     plt.xlabel("Depth")
            #     plt.ylabel("SDF")
            #     plt.show()
##########################################################################################################################################

        # if iteration % 5000 == 1:
        #     render_pkg2 = render(all_cams[0], gaussians, pipe, background)
        #     # render_pkg2 = render_pkg
        #
        #     print(f"Num Gaussians Iter {iteration}: {gaussians.get_xyz.shape[0]}")
        #     from scipy.io import savemat
        #     savemat("jerry_out/render_data.mat", {"points": render_pkg2["depth3"].detach().cpu().numpy(), "probs": render_pkg2["probs2"].detach().cpu().numpy(), "opacities": render_pkg2["opac2"].detach().cpu().numpy(), "color": render_pkg2["color2"].detach().cpu().numpy(), "gt": all_cams[0].original_image.detach().cpu().numpy()})
        #     print("Wrote data to .mat")
        #
        #     np.savez("jerry_out/render_data.npz", points=render_pkg2["depth3"].detach().cpu().numpy(), probs=render_pkg2["probs2"].detach().cpu().numpy(), opacity=render_pkg2["opac2"].detach().cpu().numpy(), color=render_pkg2["color2"].detach().cpu().numpy(), gt=all_cams[0].original_image.detach().cpu().numpy())
        #     print("Wrote data to .npz")
        #     return

        # if iteration == 2:
        #     all_cams = scene.getTrainCameras().copy()
        #     fovxs, fovys, imgs, poses = [], [], [], []
        #     for cam in all_cams:
        #         mat = np.zeros((4, 4))
        #         mat[:3, :3] = viewpoint_cam.R
        #         mat[:3, 3] = viewpoint_cam.T
        #         mat[3, 3] = 1.0
        #         poses.append(mat)
        #         imgs.append(cam.original_image.cpu().numpy())
        #         fovxs.append(cam.FoVx)
        #         fovys.append(cam.FoVy)
        #
        #     np.savez("dtu_scan24.npz", fovx=fovxs, fovy=fovys, images=imgs, poses=poses)
        #     return

        if iteration % 100 == 2:
            # depth_img = render_pkg["depth2"].detach().cpu().numpy()[0].astype(np.float32)
            mask_img = mask.detach().cpu().numpy()[0].astype(np.float32)
            color_img = image.detach().cpu().numpy().astype(np.float32).clip(0., 1.).transpose(1, 2, 0)
            # color_img = image.T.reshape(100, 100, 3).detach().cpu().numpy().astype(np.float32).clip(0., 1.)
            # depth_gt = (viewpoint_cam.depth.detach().cpu().numpy()).astype(np.float32)
            mask_gt = gt_mask.detach().cpu().numpy().astype(np.float32)
            color_gt = viewpoint_cam.original_image.detach().cpu().numpy().astype(np.float32).transpose(1, 2, 0)

            # plt.imsave("jerry_out/est_depth.png", depth_img, cmap='gray')
            # plt.imsave("jerry_out/est_mask.png", mask_img, cmap='gray')
            plt.imsave("jerry_out/est_color.png", color_img)

            fig, axs = plt.subplots(2, 2, figsize=(10, 12))
            axs[0, 0].imshow(mask_gt, cmap='gray')
            axs[0, 0].set_title('GT Mask')
            axs[0, 0].axis('off')
            axs[0, 1].imshow(mask_img, cmap='gray')
            axs[0, 1].set_title('Estimated Mask')
            axs[0, 1].axis('off')
            axs[1, 0].imshow(color_gt)
            axs[1, 0].set_title('GT Color')
            axs[1, 0].axis('off')
            axs[1, 1].imshow(color_img)
            axs[1, 1].set_title('Estimated Color')
            axs[1, 1].axis('off')
            plt.tight_layout()
            plt.show()

            # fig, axs = plt.subplots(2, 1, figsize=(5, 11))
            # axs[0].imshow(color_gt)
            # axs[0].set_title('GT Color')
            # axs[0].axis('off')
            # axs[1].imshow(color_img)
            # axs[1].set_title('Estimated Color')
            # axs[1].axis('off')
            # plt.tight_layout()
            # plt.show()

            torch.cuda.empty_cache()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        mask_loss = bce_loss(mask.squeeze(), gt_mask)
        beta_1 = 0.1
        loss = 1 * loss + beta_1 * mask_loss
        if neus and iteration > 1:
            loss = loss + eikonal_loss * 0.1

        if neus and iteration % 100 == 0:
            print("Loss:", loss)
            print("S:", deviation_network().clip(1e-6, 1e6))


        # color_error = (image - gt_image) * rays_mask
        # loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
        # print(loss)

        image.retain_grad()
        loss.backward()
        # print("After", gaussians._xyz.grad.max().item())

        # if iteration % 1000 == 0:
        #     print(f"Num Gaussians Iter {iteration}: {gaussians.get_xyz.shape[0]}")
        #     # print(f"Max L2 Size: {torch.linalg.vector_norm(gaussians.get_scaling, dim=1).max()}")

        if neus and iteration > 1:
            # alpha.grad = gaussians._opacity.grad
            sdf.backward(gaussians._opacity.grad)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, radii)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)  # gaussians.optimizer.zero_grad(set_to_none=True)

                if iteration > 1000 and iteration % 100 == 0:
                    # Clustering
                    means = gaussians.get_xyz.detach().cpu().numpy()
                    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(means))

                    labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=25))
                    unique_labels, label_counts = np.unique(labels[labels >= 0], return_counts=True)
                    most_common_label = unique_labels[np.argmax(label_counts)]
                    gaussians.prune_points(labels != most_common_label)
                    # print(f"DBSCAN Pruned {(labels != most_common_label).sum().item()} Gaussians")

                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

                # if iteration == 3000:
                #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
            # if iteration < 2:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                if neus and iteration > 1:
                    deviation_network.variance.grad /= gaussians.get_xyz.size(0)
                    optimizer_neus.step()
                    optimizer_neus.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if iteration % 15_000 == 0:
            print("Draw Means")
            means = gaussians.get_xyz.detach().cpu().numpy()
            colors = gaussians.get_features[:, 0].detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(means))
            pcd.colors = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(colors))
            o3d.visualization.draw_geometries([pcd])

            # from scipy.spatial import Delaunay
            # tri = Delaunay(means)
            #
            # max_triangle_size = 0.0001
            # max_side_length = 0.1
            #
            # def tetrahedron_volume(tetra):
            #     a, b, c, d = tetra
            #     return np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0
            #
            # from scipy.spatial.distance import euclidean
            #
            # def tetrahedron_side_lengths(tetra):
            #     a, b, c, d = tetra
            #     return [euclidean(a, b), euclidean(a, c), euclidean(a, d), euclidean(b, c), euclidean(b, d),
            #             euclidean(c, d)]
            #
            # tetrahedron_volumes = np.array([tetrahedron_volume(means[simplex]) for simplex in tri.simplices])
            # tetrahedron_edges = np.array([tetrahedron_side_lengths(means[simplex]) for simplex in tri.simplices])
            #
            # mesh = trimesh.Trimesh(vertices=means, faces=tri.simplices[(tetrahedron_volumes <= max_triangle_size) & (np.max(tetrahedron_edges, axis=1) <= max_side_length)])
            # mesh.show()
            #
            # print("Begin DBSCAN")
            # labels = np.array(pcd.cluster_dbscan(eps=0.15, min_points=25, print_progress=True))
            # unique_labels, label_counts = np.unique(labels, return_counts=True)
            # most_common_index = np.argmax(label_counts)
            # most_common_label = unique_labels[most_common_index]
            # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(means[labels == most_common_label]))
            # pcd.colors = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector(colors[labels == most_common_label]))
            # o3d.visualization.draw_geometries([pcd])

        # if iteration % 15_000 == 0:
        #     print("Begin Depth to PCD to Mesh")
        #
        #     # gaussians.extract_mesh(resolution=256)
        #
        #     def depth_image_to_point_cloud(depth_image, color_image, intrinsic_matrix, R, T):
        #         fx, fy, cx, cy = intrinsic_matrix
        #         height, width = depth_image.shape
        #
        #         pcd, colors = [], []
        #         for i in range(height):
        #             for j in range(width):
        #                 z = depth_image[i][j]
        #                 if z > 0:
        #                     x = (j - cx) * z / fx
        #                     y = (i - cy) * z / fy
        #                     pcd.append([x, y, z])
        #                     colors.append(color_image[i][j])
        #
        #         pcd = np.array(pcd)
        #         pcd = np.matmul(pcd, R) + T
        #
        #         return pcd, np.array(colors)
        #
        #     intrinsic_matrix = np.array([fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width), fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height), viewpoint_cam.image_width / 2, viewpoint_cam.image_height / 2])
        #
        #     points = np.empty((0, 3))
        #     colors = np.empty((0, 3))
        #     normals = np.empty((0, 3))
        #     all_cams = scene.getTrainCameras().copy()
        #     for viewpoint_cam in tqdm(all_cams, desc="PCD Creation Progress", leave=False):
        #         # Get data
        #         render_pkg = render(viewpoint_cam, gaussians, pipe, background, deviation_network().clip(1e-6, 1e6).item())
        #         color = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
        #         depth = render_pkg["depth2"]
        #         remove_mask = render_pkg["mask"] < 0.95
        #         depth[remove_mask] = 0
        #         depth = depth.squeeze(0).detach().cpu().numpy()
        #
        #         # Get camera extrinsics
        #         mat = np.zeros((4, 4))
        #         mat[:3, :3] = viewpoint_cam.R.transpose()
        #         mat[:3, 3] = viewpoint_cam.T
        #         mat[3, 3] = 1.0
        #         mat = np.linalg.inv(mat)
        #         mat[:3, :3] = mat[:3, :3].transpose()
        #
        #         # Get point cloud from depth and corresponding colors
        #         p, c = depth_image_to_point_cloud(depth, color, intrinsic_matrix, mat[:3, :3], mat[:3, 3])
        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(p)
        #         pcd.colors = o3d.utility.Vector3dVector(c)
        #         pcd, removed = pcd.remove_statistical_outlier(25, 1.0)
        #
        #         # Correct normals (angle between normal and vector towards camera should be < 90 deg)
        #         pcd.estimate_normals()
        #         norms = np.asarray(pcd.normals)
        #         points_np = np.asarray(pcd.points)
        #         c_vecs = mat[:3, 3] - points_np
        #         c_mags = np.linalg.norm(c_vecs, axis=1)
        #         n_mags = np.linalg.norm(norms, axis=1)
        #         angles = np.degrees(np.arccos((c_vecs * norms).sum(axis=1) / (c_mags * n_mags)))
        #         norms[angles > 90] *= (-1)
        #
        #         # Save data
        #         points = np.concatenate((points, points_np), axis=0)
        #         normals = np.concatenate((normals, norms), axis=0)
        #         colors = np.concatenate((colors, np.asarray(pcd.colors)), axis=0)
        #
        #     # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        #     # labels = np.array(pcd.cluster_dbscan(eps=0.15, min_points=25))
        #     # unique_labels, label_counts = np.unique(labels[labels >= 0], return_counts=True)
        #     # most_common_label = unique_labels[np.argmax(label_counts)]
        #     # mask = (labels == most_common_label)
        #
        #     # Display pcd
        #     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        #     pcd.normals = o3d.utility.Vector3dVector(normals)
        #     pcd.colors = o3d.utility.Vector3dVector(colors)
        #     pcd, _ = pcd.remove_statistical_outlier(100, 1.25)
        #     o3d.visualization.draw_geometries([pcd])
        #     o3d.io.write_point_cloud("jerry_out/pcd.ply", pcd)
        #
        #     # To mesh (SPSR)
        #     # pcd.orient_normals_consistent_tangent_plane(15)
        #     # o3d.io.write_point_cloud("jerry_out/pcd_with_normals.ply", pcd)
        #     # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        #     mesh.compute_vertex_normals()
        #     # mesh = mesh.filter_smooth_simple(number_of_iterations=3)
        #     o3d.visualization.draw_geometries([mesh])
        #     o3d.io.write_triangle_mesh("jerry_out/spsr_mesh.ply", mesh)
        #
        #     # # Marching Cubes
        #     # def qvec2rotmat(qvec):
        #     #     return np.array([
        #     #         [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
        #     #          2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
        #     #          2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        #     #         [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
        #     #          1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
        #     #          2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        #     #         [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
        #     #          2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
        #     #          1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])
        #     #
        #     # def pdf_3d_gaussian(p, mean, inv, det):
        #     #     d = p - mean
        #     #     return det * np.exp(-0.5 * (d.T @ inv @ d))
        #     #
        #     #
        #     # means = gaussians.get_xyz.detach().cpu().numpy()
        #     # alphas = gaussians.get_opacity.detach().cpu().numpy()
        #     # rots = gaussians.get_rotation.detach().cpu().numpy()
        #     # rotation_matrices = np.array([qvec2rotmat(qvec) for qvec in rots])
        #     # scaling = gaussians.get_scaling.detach().cpu().numpy()
        #     # scaling_matrices = np.array([np.diag(scale ** 2) for scale in scaling])
        #     # covariance_matrices = []
        #     # for rot_mat, sc_mat in tqdm(zip(rotation_matrices, scaling_matrices)):
        #     #     covariance_matrices.append(rot_mat @ sc_mat @ rot_mat.T)
        #     # covariance_matrices = np.array(covariance_matrices)
        #     # invs = np.array([np.linalg.inv(cov_mat) for cov_mat in covariance_matrices])
        #     # dets = np.array([(1 / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(cov_mat))) for cov_mat in covariance_matrices])
        #     # np.savez("jerry_debug/data_5000.npz", means=means, alphas=alphas, inv=invs, det=dets)
        #     #
        #     # minn, maxx = means.min(axis=0), means.max(axis=0)
        #     # print(minn, maxx)
        #     # dist = 1.1 * (maxx - minn).max()
        #     # corner1, corner2 = (minn + maxx) / 2 - dist / 2, (minn + maxx) / 2 + dist / 2
        #     # n = 128
        #     # queries = np.concatenate([arr.reshape(-1, 1) for arr in np.meshgrid(*[np.linspace(corner1, corner2, num=n)] * 3, indexing="ij")], axis=1)
        #     #
        #     # occ = np.zeros((n**3, 1))
        #     # for i, query in tqdm(enumerate(queries)):
        #     #     for j, mean in enumerate(means):
        #     #         occ[i, 0] += pdf_3d_gaussian(query, mean, invs[j], dets[j]) * alphas[j]
        #     #
        #     # occ = occ.reshape(n, n, n)
        #     # import mcubes
        #     # vertices, triangles = mcubes.marching_cubes(occ, 0.5)
        #     # mesh = trimesh.Trimesh(vertices, triangles)
        #     # mesh.show()
        #     # # mesh.export("mcubes_mesh.ply")


    print("Final # of gaussians:", gaussians._xyz.size(0))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def to_mesh(my_reconstruction):
    # Shorthand
    occ = my_reconstruction

    # Shape of voxel grid
    nx, ny, nz = occ.shape
    # Shape of corresponding occupancy grid
    grid_shape = (nx + 1, ny + 1, nz + 1)

    # Convert values to occupancies
    occ = np.pad(occ, 1, 'constant')

    # Determine if face present
    f1_r = (occ[:-1, 1:-1, 1:-1] & ~occ[1:, 1:-1, 1:-1])
    f2_r = (occ[1:-1, :-1, 1:-1] & ~occ[1:-1, 1:, 1:-1])
    f3_r = (occ[1:-1, 1:-1, :-1] & ~occ[1:-1, 1:-1, 1:])

    f1_l = (~occ[:-1, 1:-1, 1:-1] & occ[1:, 1:-1, 1:-1])
    f2_l = (~occ[1:-1, :-1, 1:-1] & occ[1:-1, 1:, 1:-1])
    f3_l = (~occ[1:-1, 1:-1, :-1] & occ[1:-1, 1:-1, 1:])

    f1 = f1_r | f1_l
    f2 = f2_r | f2_l
    f3 = f3_r | f3_l

    assert (f1.shape == (nx + 1, ny, nz))
    assert (f2.shape == (nx, ny + 1, nz))
    assert (f3.shape == (nx, ny, nz + 1))

    # Determine if vertex present
    v = np.full(grid_shape, False)

    v[:, :-1, :-1] |= f1
    v[:, :-1, 1:] |= f1
    v[:, 1:, :-1] |= f1
    v[:, 1:, 1:] |= f1

    v[:-1, :, :-1] |= f2
    v[:-1, :, 1:] |= f2
    v[1:, :, :-1] |= f2
    v[1:, :, 1:] |= f2

    v[:-1, :-1, :] |= f3
    v[:-1, 1:, :] |= f3
    v[1:, :-1, :] |= f3
    v[1:, 1:, :] |= f3

    # Calculate indices for vertices
    n_vertices = v.sum()
    v_idx = np.full(grid_shape, -1)
    v_idx[v] = np.arange(n_vertices)

    # Vertices
    v_x, v_y, v_z = np.where(v)
    v_x = v_x / nx - 0.5
    v_y = v_y / ny - 0.5
    v_z = v_z / nz - 0.5
    vertices = np.stack([v_x, v_y, v_z], axis=1)

    # Face indices
    f1_l_x, f1_l_y, f1_l_z = np.where(f1_l)
    f2_l_x, f2_l_y, f2_l_z = np.where(f2_l)
    f3_l_x, f3_l_y, f3_l_z = np.where(f3_l)

    f1_r_x, f1_r_y, f1_r_z = np.where(f1_r)
    f2_r_x, f2_r_y, f2_r_z = np.where(f2_r)
    f3_r_x, f3_r_y, f3_r_z = np.where(f3_r)

    faces_1_l = np.stack([
        v_idx[f1_l_x, f1_l_y, f1_l_z],
        v_idx[f1_l_x, f1_l_y, f1_l_z + 1],
        v_idx[f1_l_x, f1_l_y + 1, f1_l_z + 1],
        v_idx[f1_l_x, f1_l_y + 1, f1_l_z],
    ], axis=1)

    faces_1_r = np.stack([
        v_idx[f1_r_x, f1_r_y, f1_r_z],
        v_idx[f1_r_x, f1_r_y + 1, f1_r_z],
        v_idx[f1_r_x, f1_r_y + 1, f1_r_z + 1],
        v_idx[f1_r_x, f1_r_y, f1_r_z + 1],
    ], axis=1)

    faces_2_l = np.stack([
        v_idx[f2_l_x, f2_l_y, f2_l_z],
        v_idx[f2_l_x + 1, f2_l_y, f2_l_z],
        v_idx[f2_l_x + 1, f2_l_y, f2_l_z + 1],
        v_idx[f2_l_x, f2_l_y, f2_l_z + 1],
    ], axis=1)

    faces_2_r = np.stack([
        v_idx[f2_r_x, f2_r_y, f2_r_z],
        v_idx[f2_r_x, f2_r_y, f2_r_z + 1],
        v_idx[f2_r_x + 1, f2_r_y, f2_r_z + 1],
        v_idx[f2_r_x + 1, f2_r_y, f2_r_z],
    ], axis=1)

    faces_3_l = np.stack([
        v_idx[f3_l_x, f3_l_y, f3_l_z],
        v_idx[f3_l_x, f3_l_y + 1, f3_l_z],
        v_idx[f3_l_x + 1, f3_l_y + 1, f3_l_z],
        v_idx[f3_l_x + 1, f3_l_y, f3_l_z],
    ], axis=1)

    faces_3_r = np.stack([
        v_idx[f3_r_x, f3_r_y, f3_r_z],
        v_idx[f3_r_x + 1, f3_r_y, f3_r_z],
        v_idx[f3_r_x + 1, f3_r_y + 1, f3_r_z],
        v_idx[f3_r_x, f3_r_y + 1, f3_r_z],
    ], axis=1)

    faces = np.concatenate([
        faces_1_l, faces_1_r,
        faces_2_l, faces_2_r,
        faces_3_l, faces_3_r,
    ], axis=0)

    # vertices = self.loc + self.scale * vertices
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    return mesh

if __name__ == "__main__":
    # torch.set_default_device('cuda')
    # torch.manual_seed(0)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_500, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_500, 10_000, 15_000, 20_000, 25_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[4_000, 7_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
