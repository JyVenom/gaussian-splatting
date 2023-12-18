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
import trimesh
from matplotlib import pyplot as plt
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, cosine_similarity_loss, processDepth, bce_loss, depth_smoothness_loss#, dbscan_loss, dbscan_loss2
from NeuS.fields import SDFNetwork, SingleVarianceNetwork
from utils.sh_utils import SH2RGB

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def quat2mat(q):
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros((q.shape[0], 3, 3), device='cuda')
    R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    R[:, 0, 1] = 2 * (x * y - z * r)
    R[:, 0, 2] = 2 * (x * z + y * r)
    R[:, 1, 0] = 2 * (x * y + z * r)
    R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    R[:, 1, 2] = 2 * (y * z - x * r)
    R[:, 2, 0] = 2 * (x * z - y * r)
    R[:, 2, 1] = 2 * (y * z + x * r)
    R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)

    return R


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
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

    sdf_network = SDFNetwork(d_out=257, d_in=3, d_hidden=256, n_layers=8, skip_in=[4], multires=6, bias=0.5, scale=1.0, geometric_init=False, weight_norm=True).to("cuda")
    checkpoint = torch.load(os.path.join(os.path.abspath(""), 'NeuS/checkpoints/ckpt_300000.pth'), map_location="cuda")
    sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
    neus_optim = torch.optim.Adam(sdf_network.parameters(), lr=1e-3)

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
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_mask = (viewpoint_cam.depth > 0.01).to(torch.float32)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth, mask = render_pkg["depth"], render_pkg["mask"]

        if iteration % 100 == 0:
            depth_img = depth.detach().cpu().numpy()[0].astype(np.float32)
            mask_img = mask.detach().cpu().numpy()[0].astype(np.float32)
            color_img = image.detach().cpu().numpy().astype(np.float32).clip(0., 1.).transpose(1, 2, 0)
            depth_gt = (viewpoint_cam.depth.detach().cpu().numpy()).astype(np.float32)
            mask_gt = gt_mask.detach().cpu().numpy().astype(np.float32)
            color_gt = viewpoint_cam.original_image.detach().cpu().numpy().astype(np.float32).transpose(1, 2, 0)

            plt.imsave("jerry_out/est_depth.png", depth_img, cmap='gray')
            plt.imsave("jerry_out/est_mask.png", mask_img, cmap='gray')
            plt.imsave("jerry_out/est_color.png", color_img)

            fig, axs = plt.subplots(3, 2, figsize=(10, 17))
            axs[0, 0].imshow(depth_gt, cmap='gray')
            axs[0, 0].set_title('GT Depth')
            axs[0, 0].axis('off')
            axs[0, 1].imshow(depth_img, cmap='gray')
            axs[0, 1].set_title('Estimated Depth')
            axs[0, 1].axis('off')
            axs[1, 0].imshow(mask_gt, cmap='gray')
            axs[1, 0].set_title('GT Mask')
            axs[1, 0].axis('off')
            axs[1, 1].imshow(mask_img, cmap='gray')
            axs[1, 1].set_title('Estimated Mask')
            axs[1, 1].axis('off')
            axs[2, 0].imshow(color_gt)
            axs[2, 0].set_title('GT Color')
            axs[2, 0].axis('off')
            axs[2, 1].imshow(color_img)
            axs[2, 1].set_title('Estimated Color')
            axs[2, 1].axis('off')
            plt.tight_layout()
            plt.show()

            torch.cuda.empty_cache()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        color_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = 0
        loss += color_loss
        # if iteration > 1500:
        #     sdf = sdf_network.sdf(gaussians.get_xyz).squeeze()
        #     sdf_loss = torch.nn.functional.l1_loss(sdf, torch.zeros_like(sdf)) * 10
        #     loss += sdf_loss
        #
        #     normals = torch.zeros_like(gaussians.get_xyz).cuda()
        #     mask = gaussians.get_scaling.argmin(dim=1)
        #     normals[torch.arange(mask.shape[0]), mask] = 1
        #     normals.requires_grad_(True)
        #     rot_mats = quat2mat(gaussians.get_rotation)
        #     gaus_norms = torch.einsum('ijk,ik->ij', rot_mats, normals)
        #     sdf_norms = sdf_network.gradient(gaussians.get_xyz).squeeze().detach()
        #     norms_loss = torch.nn.functional.cosine_embedding_loss(gaus_norms, sdf_norms, torch.ones(gaus_norms.shape[0]).cuda())
        #     loss += norms_loss
        loss.backward()

        iter_end.record()

        if iteration % 1000 == 0:
            # from scipy.io import savemat
            # savemat("gaussian_data_12_8.mat", {"means": gaussians.get_xyz.detach().cpu().numpy(), "scaling": gaussians.get_scaling.detach().cpu().numpy(), "rotation": gaussians.get_rotation.detach().cpu().numpy(), "opacity": gaussians.get_opacity.detach().cpu().numpy(), "sdf": sdf_network.sdf(gaussians.get_xyz).detach().cpu().numpy()})

            sdf = sdf_network.sdf(gaussians.get_xyz).detach().cpu().numpy()
            plt.hist(sdf, bins=np.linspace(sdf.min(), sdf.max(), 1000))
            std = np.sqrt(((sdf-sdf.mean())**2).mean())
            plt.title(f"STD: {std}")
            plt.show()

        if iteration % 100 == 0:
            print("\nColor Loss:", color_loss)
            # if iteration > 1500:
            #     print("SDF Loss:", sdf_loss)
            #     print("Normal Loss:", norms_loss)
            #     print("Loss:", loss)
            print("# Gaussians:", gaussians.get_xyz.shape[0])
            sdf = sdf_network.sdf(gaussians.get_xyz.detach()).squeeze()
            sdf = sdf[~sdf.isnan()].abs()
            print("Avg SDF", sdf.mean().item())
            opac_thresh = (-math.cos(math.pi * iteration / 5000) + 1) * 0.45
            print("Opacity Threshold", opac_thresh)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, radii)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    opac_thresh = (-math.cos(math.pi*iteration/5000)+1)*0.45
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opac_thresh, scene.cameras_extent, size_threshold)

                # if iteration >= 1000 and iteration % opt.densification_interval == 0:
                #     sdf = sdf_network.sdf(gaussians.get_xyz).squeeze()
                #     if iteration < 6000:
                #         sdf_thresh = (math.cos(math.pi*(iteration-1000)/5000)+1)/2*(1-0.01)+0.01
                #     else:
                #         sdf_thresh = 0.01
                #     gaussians.prune_points(sdf.abs() > sdf_thresh)
                #
                # if iteration == 1000 and iteration % opt.densification_interval == 0:
                #     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gaussians.get_xyz.detach().cpu().numpy()))
                #     labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=25))
                #     unique_labels, label_counts = np.unique(labels[labels >= 0], return_counts=True)
                #     most_common_label = unique_labels[np.argmax(label_counts)]
                #     gaussians.prune_points(labels != most_common_label)
                #     print(f"DBSCAN Pruned {(labels != most_common_label).sum().item()} Gaussians")

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                # # Don't let opacity go below thresh
                # opac_thresh = (-math.cos(math.pi*iteration/5000)+1)*0.45
                # opacities_new = gaussians.inverse_opacity_activation(torch.max(gaussians.get_opacity, torch.ones_like(gaussians.get_opacity) * opac_thresh))
                # optimizable_tensors = gaussians.replace_tensor_to_optimizer(opacities_new, "opacity")
                # gaussians._opacity = optimizable_tensors["opacity"]

                # # Don't let scaling go above thresh
                # scale_thresh = (math.cos(math.pi*iteration/5000)+1)*0.5*(0.5-0.01)+0.01
                # scales_new = gaussians.scaling_inverse_activation(torch.min(gaussians.get_scaling, torch.ones_like(gaussians.get_scaling) * scale_thresh))
                # optimizable_tensors = gaussians.replace_tensor_to_optimizer(scales_new, "scaling")
                # gaussians._scaling = optimizable_tensors["scaling"]

                # Limit z-axis scaling to 0.001
                scales_new = gaussians.get_scaling
                scales_new[:, 2] = torch.min(scales_new[:, 2], torch.ones_like(scales_new[:, 2]) * 0.001)
                scales_new = gaussians.scaling_inverse_activation(scales_new)
                optimizable_tensors = gaussians.replace_tensor_to_optimizer(scales_new, "scaling")
                gaussians._scaling = optimizable_tensors["scaling"]

                if iteration > 1000:
                    # Don't let opacity go below 0.7
                    opacities_new = gaussians.inverse_opacity_activation(torch.max(gaussians.get_opacity, torch.ones_like(gaussians.get_opacity) * 0.7))
                    optimizable_tensors = gaussians.replace_tensor_to_optimizer(opacities_new, "opacity")
                    gaussians._opacity = optimizable_tensors["opacity"]

                    # Don't let scaling go above 0.02
                    scales_new = gaussians.scaling_inverse_activation(torch.min(gaussians.get_scaling, torch.ones_like(gaussians.get_scaling) * 0.02))
                    optimizable_tensors = gaussians.replace_tensor_to_optimizer(scales_new, "scaling")
                    gaussians._scaling = optimizable_tensors["scaling"]

                neus_optim.zero_grad()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if iteration % 2_000 == 0:
            print("Draw Means")
            means = gaussians.get_xyz.detach().cpu().numpy()
            colors = SH2RGB(gaussians.get_features[:, 0].detach().cpu().numpy())

            normals = np.zeros_like(means)
            normals[:, 2] = 1
            rot_mats = quat2mat(gaussians.get_rotation.detach()).cpu().numpy()
            normals = np.einsum('ijk,ik->ij', rot_mats, normals)

            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(means))
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd.normals = o3d.utility.Vector3dVector(normals)

            o3d.visualization.draw_geometries([pcd])

            pcd.paint_uniform_color([0.8, 0.8, 0.8])
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh])

        if iteration % 50_000 == 0:
            print("Begin Depth to PCD to Mesh")

            def depth_image_to_point_cloud(depth_image, color_image, intrinsic_matrix, R, T):
                fx, fy, cx, cy = intrinsic_matrix
                height, width = depth_image.shape

                rows, cols = torch.tensor(range(height), device='cuda').unsqueeze(-1).expand(height, width), torch.tensor(range(width), device='cuda').expand(height, width)

                zs = depth_image
                xs = (cols - cx) * zs / fx
                ys = (rows - cy) * zs / fy
                mask = zs > 0
                pcd2 = torch.cat((xs[mask].unsqueeze(-1), ys[mask].unsqueeze(-1), zs[mask].unsqueeze(-1)), dim=-1)

                return np.matmul(pcd2.detach().cpu().numpy(), R) + T, color_image[mask]

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
                color = render_pkg["render"].permute(1, 2, 0)
                depth = render_pkg["depth"]
                remove_mask = render_pkg["mask"] < 0.95
                depth[remove_mask] = 0
                depth = depth.squeeze(0)

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
                pcd.colors = o3d.utility.Vector3dVector(c.detach().cpu().numpy().clip(0, 0.999))
                o3d.io.write_point_cloud(f"jerry_out/all_views_pcds/{int(viewpoint_cam.image_name)}.ply", pcd)
                # pcd, _ = pcd.remove_statistical_outlier(25, 0.5)

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
                points = np.concatenate((points, np.asarray(pcd.points)), axis=0)
                normals = np.concatenate((normals, norms), axis=0)
                colors = np.concatenate((colors, np.asarray(pcd.colors)), axis=0)

            # Display pcd
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            pcd.normals = o3d.utility.Vector3dVector(normals)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            # pcd, _ = pcd.remove_statistical_outlier(50, 1.0)
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
