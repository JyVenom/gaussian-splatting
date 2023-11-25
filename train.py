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


def neus_sigma(xyz, sdf_network, deviation_network):
    batch_size = 10_000
    points = xyz.split(batch_size)

    out_psdfs = []
    a, b = 0, 0

    for pts in points:
        batch_size = pts.shape[0]

        # pts = pts.requires_grad_(True)
        sdf = sdf_network.sdf(pts)

        inv_s = deviation_network().expand(batch_size, 1)
        sig = (1 + torch.exp(-inv_s * sdf)).pow(-1)
        # ldd = (inv_s * torch.exp(-inv_s * sdf)) / ((1 + torch.exp(-inv_s * sdf)).pow(2))
        ldd = inv_s * sig * (1 - sig)
        sigma = (ldd / sig / inv_s * 2).reshape(batch_size, 1)
        # sigma = (1-(torch.exp(-inv_s * sdf)/(1+torch.exp(-inv_s * sdf))))
        # torch.nan_to_num(sigma)
        # sigma[sigma.isnan()] = 0.0
        # sigma = torch.where(torch.isnan(sigma), torch.zeros_like(sigma), sigma)

        # Eikonal loss
        gradients = sdf_network.gradient(pts).squeeze()
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, 3), ord=2, dim=-1) - 1.0) ** 2
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size)
        relax_inside_sphere = (pts_norm < 1.2).float()
        a += (relax_inside_sphere * gradient_error).sum()
        b += relax_inside_sphere.sum()

        out_psdfs.append(sigma)

        del sdf, sig, ldd, sigma, gradients, gradient_error, pts_norm, relax_inside_sphere

    sigma = torch.cat(out_psdfs, dim=0)
    gradient_error = a / (b + 1e-5)

    return sigma, gradient_error

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.load_ply("data/dtu/dtu_scan24/init_3000.ply")
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
    deviation_network = SingleVarianceNetwork(init_val=0.55).to("cuda")
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

    debug_wo_optimize = False

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
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
        if not neus or iteration == 1:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            mask = render_pkg["mask"]

##########################################################################################################################################
        if neus:
            if iteration > 1:
                sigma, eikonal_loss = neus_sigma(gaussians.get_xyz, sdf_network, deviation_network)
                if debug_wo_optimize:
                    gaussians._psdf = sigma
                    # gaussians._psdf = torch.ones_like(gaussians._psdf)
                else:
                    optimizable_tensors = gaussians.replace_tensor_to_optimizer(sigma, "psdf")
                    gaussians._psdf = optimizable_tensors["psdf"]
                    # optimizable_tensors = gaussians.replace_tensor_to_optimizer(torch.ones_like(gaussians._psdf), "psdf")
                    # gaussians._psdf = optimizable_tensors["psdf"]

                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                mask = render_pkg["mask"]

                if iteration % 3000 == 0:
                    depths3 = render_pkg["depth3"].detach().cpu().numpy()
                    psdf = render_pkg["psdf2"].detach().cpu().numpy()
                    opac = render_pkg["opac2"].detach().cpu().numpy()
                    alpha = render_pkg["alpha2"].detach().cpu().numpy()
                    pgaus = render_pkg["probs2"].detach().cpu().numpy()
                    s = deviation_network().item()
                    sdf = -np.log(psdf / (2 - psdf))/s
                    weights = alpha * np.cumprod(np.concatenate([np.ones((1, 400, 400)), 1. - alpha + 1e-7], 0), 0)[:-1]
                    weight_sum = weights.cumsum(axis=0)

                    print("View #", viewpoint_cam.image_name)

                    # rays = np.concatenate([arr.reshape(-1, 1) for arr in np.meshgrid(*[np.arange(196, 205, 2)] * 2, indexing="ij")], axis=1).astype(int)
                    # for ray_num in rays:
                    ray_num = (200, 200)
                    num = depths3[:, ray_num[0], ray_num[1]].argmin()
                    if num == 0 and depths3[0, ray_num[0], ray_num[1]] != 0:
                        num = depths3.shape[0]
                    plt.plot(depths3[:num, ray_num[0], ray_num[1]], opac[:num, ray_num[0], ray_num[1]], color='red')
                    plt.plot(depths3[:num, ray_num[0], ray_num[1]], alpha[:num, ray_num[0], ray_num[1]], 'b*')
                    plt.plot(depths3[:num, ray_num[0], ray_num[1]], psdf[:num, ray_num[0], ray_num[1]], color='green')
                    plt.plot(depths3[:num, ray_num[0], ray_num[1]], pgaus[:num, ray_num[0], ray_num[1]], color='orange')
                    plt.plot(depths3[:num, ray_num[0], ray_num[1]], sdf[:num, ray_num[0], ray_num[1]] * 10, color='violet')
                    plt.plot(depths3[:num, ray_num[0], ray_num[1]], weight_sum[:num, ray_num[0], ray_num[1]], color='indigo')
                    plt.legend(["opac", "alpha", "psdf (normalised eq, /=s)", "pgaus", "sdf x10", "weight sum"])
                    plt.show()

                    # def fov2focal(fov, pixels):
                    #     return pixels / (2 * math.tan(fov / 2))
                    # intrinsic_matrix = np.array([fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width), fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height), viewpoint_cam.image_width / 2, viewpoint_cam.image_height / 2])
                    #
                    # gpsdf = gaussians.get_psdf.detach().cpu().numpy()
                    # gsdf = -np.log(gpsdf / (2 - gpsdf)) / s
                    #
                    # from scipy.io import savemat
                    # savemat("gaus_ray_data.mat", {"mean": gaussians.get_xyz.detach().cpu().numpy(), "scaling": gaussians.get_scaling.detach().cpu().numpy(), "rotation": gaussians.get_rotation.detach().cpu().numpy(), "sdf": gsdf, "intrinsics": intrinsic_matrix})

                    pass

            if iteration % 10000 == 0:
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
##########################################################################################################################################

        if iteration % 100 == 0:
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

            torch.cuda.empty_cache()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # mask_loss = bce_loss(mask.squeeze(), gt_mask)
        # beta_1 = 0.1
        # loss = 1 * loss + beta_1 * mask_loss
        # if neus and iteration > 1:
        #     loss = loss + eikonal_loss * 0.1

        if neus and iteration % 100 == 0:
            print("Loss:", loss)
            print("S:", deviation_network().clip(1e-6, 1e6))
            print("# Gaussians:", gaussians.get_xyz.shape[0])

        if not debug_wo_optimize:
            loss.backward()

        # if neus and iteration > 1:
        #     sigma.backward(gaussians._opacity.grad)

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
                if not debug_wo_optimize:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, radii)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and not debug_wo_optimize:
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

            # Optimizer step
            if (iteration < opt.iterations and not debug_wo_optimize) or iteration == 1:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                # if neus and iteration > 1:
                #     # deviation_network.variance.grad /= gaussians.get_xyz.size(0)
                #     optimizer_neus.step()
                #     optimizer_neus.zero_grad(set_to_none=True)

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

        if iteration % 15_000 == 0:
            print("Begin Depth to PCD to Mesh")

            # gaussians.extract_mesh(resolution=256)

            def depth_image_to_point_cloud(depth_image, color_image, intrinsic_matrix, R, T):
                fx, fy, cx, cy = intrinsic_matrix
                height, width = depth_image.shape

                pcd, colors = [], []
                for i in range(height):
                    for j in range(width):
                        z = depth_image[i][j]
                        if z > 0:
                            x = (j - cx) * z / fx
                            y = (i - cy) * z / fy
                            pcd.append([x, y, z])
                            colors.append(color_image[i][j])

                pcd = np.array(pcd)
                pcd = np.matmul(pcd, R) + T

                return pcd, np.array(colors)

            def fov2focal(fov, pixels):
                return pixels / (2 * math.tan(fov / 2))

            intrinsic_matrix = np.array([fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width), fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height), viewpoint_cam.image_width / 2, viewpoint_cam.image_height / 2])

            points = np.empty((0, 3))
            colors = np.empty((0, 3))
            normals = np.empty((0, 3))
            all_cams = scene.getTrainCameras().copy()
            for viewpoint_cam in tqdm(all_cams, desc="PCD Creation Progress", leave=False):
                # Get data
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                color = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
                depth = render_pkg["depth2"]
                remove_mask = render_pkg["mask"] < 0.95
                depth[remove_mask] = 0
                depth = depth.squeeze(0).detach().cpu().numpy()

                # Get camera extrinsics
                mat = np.zeros((4, 4))
                mat[:3, :3] = viewpoint_cam.R.transpose()
                mat[:3, 3] = viewpoint_cam.T
                mat[3, 3] = 1.0
                mat = np.linalg.inv(mat)
                mat[:3, :3] = mat[:3, :3].transpose()

                # Get point cloud from depth and corresponding colors
                p, c = depth_image_to_point_cloud(depth, color, intrinsic_matrix, mat[:3, :3], mat[:3, 3])
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(p)
                pcd.colors = o3d.utility.Vector3dVector(c)
                pcd, removed = pcd.remove_statistical_outlier(25, 1.0)

                # Correct normals (angle between normal and vector towards camera should be < 90 deg)
                pcd.estimate_normals()
                norms = np.asarray(pcd.normals)
                points_np = np.asarray(pcd.points)
                c_vecs = mat[:3, 3] - points_np
                c_mags = np.linalg.norm(c_vecs, axis=1)
                n_mags = np.linalg.norm(norms, axis=1)
                angles = np.degrees(np.arccos((c_vecs * norms).sum(axis=1) / (c_mags * n_mags)))
                norms[angles > 90] *= (-1)

                # Save data
                points = np.concatenate((points, points_np), axis=0)
                normals = np.concatenate((normals, norms), axis=0)
                colors = np.concatenate((colors, np.asarray(pcd.colors)), axis=0)

            # Display pcd
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            pcd.normals = o3d.utility.Vector3dVector(normals)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd, _ = pcd.remove_statistical_outlier(100, 1.25)
            o3d.visualization.draw_geometries([pcd])
            o3d.io.write_point_cloud("jerry_out/pcd.ply", pcd)

            # To mesh (SPSR)
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
            mesh.compute_vertex_normals()
            # mesh = mesh.filter_smooth_simple(number_of_iterations=3)
            o3d.visualization.draw_geometries([mesh])
            o3d.io.write_triangle_mesh("jerry_out/spsr_mesh.ply", mesh)

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
