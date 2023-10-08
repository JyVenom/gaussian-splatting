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
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import numpy as np
import torch
import trimesh
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import open3d as o3d

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, bce_loss

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
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

    # cur_xyz_lr = opt.position_lr_init * scene.cameras_extent
    # xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init * scene.cameras_extent, lr_final=opt.position_lr_final * scene.cameras_extent, lr_delay_mult=opt.position_lr_delay_mult, max_steps=opt.position_lr_max_steps)
    plt.rcParams.update({'font.size': 24})

    color_losses, mask_losses, net_losses = [], [], []

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
        # cur_xyz_lr = xyz_scheduler_args(iteration)
        # # for param_group in optimizer.param_groups:
        # #     if param_group["name"] == "xyz":
        # #         cur_xyz_lr = xyz_scheduler_args(iteration)
        # #         param_group['lr'] = cur_xyz_lr

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            # print("Num cams:", len(viewpoint_stack))

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_mask = (viewpoint_cam.depth > 0.01).to(torch.float32)
        # gt_depth = viewpoint_cam.depth

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        mask = render_pkg["mask"]
        # depth = render_pkg["mask"]

        # if iteration % 5000 == 0:
        #     print(f"Num Gaussians Iter {iteration}: {gaussians.get_xyz.shape[0]}")
        #     from scipy.io import savemat
        #     savemat("jerry_out/gaussian_data.mat", {"means": gaussians.get_xyz.detach().cpu().numpy(), "scaling": gaussians.get_scaling.detach().cpu().numpy(), "rotation": gaussians.get_rotation.detach().cpu().numpy(), "opacity": gaussians.get_opacity.detach().cpu().numpy()})
        #     print("Wrote data to .mat")

        # if iteration % 2000 == 0:
        #     # render_pkg2 = render_pkg
        #     render_pkg2 = render(scene.getTrainCameras().copy()[3], gaussians, pipe, background)
        #
        #     from scipy.io import savemat
        #     savemat("jerry_out/depth2_alphas_depths3.mat", {"depth2": render_pkg2["depth2"].detach().cpu().numpy(), "alphas": render_pkg2["alphas"].detach().cpu().numpy(), "depths3": render_pkg2["depths3"].detach().cpu().numpy()})
        #     # savemat("jerry_out/depths_peak_01.mat", {"depth": render_pkg2["depth"].detach().cpu().numpy()})
        #
        #     print("Wrote data to .mat")

        if iteration % 500 == 0:
            depth_img = render_pkg["depth2"].detach().cpu().numpy()[0].astype(np.float32)
            mask_img = mask.detach().cpu().numpy()[0].astype(np.float32)
            color_img = image.detach().cpu().numpy().astype(np.float32).transpose(1, 2, 0).clip(0., 1.)
            depth_gt = (viewpoint_cam.depth.detach().cpu().numpy()).astype(np.float32)
            mask_gt = gt_mask.detach().cpu().numpy().astype(np.float32)
            color_gt = viewpoint_cam.original_image.detach().cpu().numpy().astype(np.float32).transpose(1, 2, 0)

            # plt.imsave("jerry_out/est_mask.png", mask_img, cmap='gray')

            fig, axs = plt.subplots(3, 2, figsize=(10, 16))
            axs[0, 0].imshow(mask_gt, cmap='gray')
            axs[0, 0].set_title('GT Mask')
            axs[0, 0].axis('off')
            axs[0, 1].imshow(mask_img, cmap='gray')
            axs[0, 1].set_title('Estimated Mask')
            axs[0, 1].axis('off')
            axs[1, 0].imshow(depth_gt, cmap='gray')
            axs[1, 0].set_title('GT Depth')
            axs[1, 0].axis('off')
            axs[1, 1].imshow(depth_img, cmap='gray')
            axs[1, 1].set_title('Estimated Depth')
            axs[1, 1].axis('off')
            axs[2, 0].imshow(color_gt)
            axs[2, 0].set_title('GT Color')
            axs[2, 0].axis('off')
            axs[2, 1].imshow(color_img)
            axs[2, 1].set_title('Estimated Color')
            axs[2, 1].axis('off')
            plt.tight_layout()
            plt.show()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # color_losses.append(loss.detach().cpu().to(torch.float32).item())
        mask_loss = bce_loss(mask.squeeze(0), gt_mask)
        # mask_losses.append(mask_loss.detach().cpu().to(torch.float32).item())
        beta = 0.9  # NOTE TO JERRY: THIS IS THE COLOR LOSS WEIGHT, SO KEEP IT HIGH!!!
        loss = beta * loss + (1 - beta) * mask_loss
        # net_losses.append(loss.detach().cpu().to(torch.float32).item())
        loss.backward()

        if iteration % 1000 == 0:
        #     import plotly.express as px
        #     import pandas as pd
        #     df = pd.DataFrame({"Color Loss": color_losses, "Mask Loss": mask_losses, "Net Loss": net_losses})
        #     fig = px.scatter(df, y=["Color Loss", "Mask Loss", "Net Loss"])
        #     fig.show()
            print(f"Loss Iter {iteration}: {loss}")

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
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
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

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()  # gaussians.optimizer.zero_grad(set_to_none=True)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if iteration % 30_000 == 0:
            print("Begin Depth to PCD to Mesh")
            def depth_image_to_point_cloud(depth_image, color_image, intrinsic_matrix, R, T):
                fx, fy, cx, cy = intrinsic_matrix
                height, width = depth_image.shape

                pcd, colors = [], []
                for i in range(height):
                    for j in range(width):
                        z = depth_image[i][j]
                        if z > 2:
                            x = (j - cx) * z / fx
                            y = (i - cy) * z / fy
                            pcd.append([x, y, z])
                            colors.append(color_image[i][j])

                pcd = np.array(pcd)
                pcd = np.matmul(pcd, R) + T

                return pcd, np.array(colors)

            # Get camera intrinsics (shared)
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
            pcd, _ = pcd.remove_statistical_outlier(50, 1.5)
            o3d.visualization.draw_geometries([pcd])
            o3d.io.write_point_cloud("jerry_out/pcd.ply", pcd)

            # To mesh (SPSR)
            # pcd.orient_normals_consistent_tangent_plane(15)
            # o3d.io.write_point_cloud("jerry_out/pcd_with_normals.ply", pcd)
            # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            mesh = mesh.filter_smooth_simple(number_of_iterations=3)
            mesh.compute_vertex_normals()
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_500, 5_000, 7_500, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_500, 5_000, 7_500, 10_000, 15_000, 20_000, 25_000, 30_000])
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
