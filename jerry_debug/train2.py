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
from matplotlib import pyplot as plt
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

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

    cur_xyz_lr = opt.position_lr_init * scene.cameras_extent
    xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init * scene.cameras_extent, lr_final=opt.position_lr_final * scene.cameras_extent, lr_delay_mult=opt.position_lr_delay_mult, max_steps=opt.position_lr_max_steps)
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

            # sum = 0  # cnt = 0  # for cam in scene.getTestCameras().copy():  #     jry_depth = cam.depth  #     mid = (jry_depth[jry_depth > 0].min() + jry_depth[jry_depth > 0].max()) / 2  #     sum += mid  #     cnt += 1  # print("avg", sum / cnt)  #  # sum = 0  # cnt = 0  # for cam in scene.getTrainCameras().copy():  #     jry_depth = cam.depth  #     mid = (jry_depth[jry_depth > 0].min() + jry_depth[jry_depth > 0].max()) / 2  #     sum += mid  #     cnt += 1  # print("avg", sum / cnt)

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_depth = viewpoint_cam.depth.to(torch.float32)

        # mid = ((gt_depth[gt_depth > 0].min() + gt_depth[gt_depth > 0].max()) / 2).to(torch.float32)
        # gt_depth[gt_depth > 0] = 2 * mid - gt_depth[gt_depth > 0]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg["depth"]

        if iteration % 250 == 0:
            depth_img = depth.detach().cpu().numpy()[0].astype(np.float32)
            depth_gt = gt_depth.cpu().numpy().astype(np.float32)

            plt.imsave("../jerry_out/est_depth_loss.png", depth_img, cmap='gray')

            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            axs[0].imshow(depth_img, cmap='gray')
            axs[0].set_title('Estimate')
            axs[0].axis('off')
            axs[1].imshow(depth_gt, cmap='gray')
            axs[1].set_title('GT')
            axs[1].axis('off')
            plt.show()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        beta = 0.95
        depth_loss = l1_loss(depth, gt_depth)
        loss = beta * loss + (1 - beta) * depth_loss
        loss.backward()

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

            # if iteration == 4000:
            #     xyz = gaussians._xyz.data.cpu().numpy()
            #     features_dc = gaussians._features_dc.data.cpu().numpy()
            #     features_rest = gaussians._features_rest.data.cpu().numpy()
            #     scaling = gaussians._scaling.data.cpu().numpy()
            #     rotation = gaussians._rotation.data.cpu().numpy()
            #     opacity = gaussians._opacity.data.cpu().numpy()
            #     max_radii2d = gaussians.max_radii2D.data.cpu().numpy()
            #     xyz_grad_accum = gaussians.xyz_gradient_accum.data.cpu().numpy()
            #     denom = gaussians.denom.data.cpu().numpy()
            #     xyz_grads = gaussians._xyz.grad.cpu().numpy()
            #     numpy.savez("iter_4000.npz", xyz=xyz, features_dc=features_dc, features_rest=features_rest,
            #                 scaling=scaling, rotation=rotation, opacity=opacity, max_radii2d=max_radii2d,
            #                 xyz_grad_accum=xyz_grad_accum, denom=denom, xyz_grads=xyz_grads)
            #     print(gaussians.max_sh_degree)
            #     print(scene.cameras_extent)
            #     print("Num Gaussians:", gaussians.gaussians_count)
            #     break

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                #                                                      radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, radii)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)  # gaussians.optimizer.zero_grad(set_to_none=True)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()  # gaussians.optimizer.zero_grad(set_to_none=True)

            # if iteration % 100 == 0:
            #     print("Jerry testing, gaussian count =", gaussians.gaussians_count)

            # Optimizer step
            if iteration < opt.iterations:
                # gaussians._xyz[gaussians.gaussians_count:].grad = 0
                # gaussians._features_dc[gaussians.gaussians_count:].grad = 0
                # gaussians._features_rest[gaussians.gaussians_count:].grad = 0
                # gaussians._scaling[gaussians.gaussians_count:].grad = 0
                # gaussians._rotation[gaussians.gaussians_count:].grad = 0
                # gaussians._opacity[gaussians.gaussians_count:].grad = 0

                # optimizer = generate_optim(gaussians, opt, cur_xyz_lr, iteration)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if iteration % 2500 == 0:
            import math
            import open3d as o3d
            # from scipy.io import savemat

            def depth_image_to_point_cloud(depth_image, intrinsic_matrix, R, T):
                fx, fy, cx, cy = intrinsic_matrix
                height, width = depth_image.shape

                pcd = []
                for i in range(height):
                    for j in range(width):
                        z = depth_image[i][j]
                        if z > 2:
                            x = (j - cx) * z / fx
                            y = (i - cy) * z / fy
                            pcd.append([x, y, z])

                pcd = np.array(pcd)
                pcd = np.matmul(pcd, R) + T

                # savemat(f"render_depth_{_ + 9}.mat", {"depth_img": depth_img, "intrinsics": intrinsic_matrix, "R": R, "T": T})

                return pcd

            # def depth_image_to_point_cloud(depth_image, intrinsic_matrix, R, T):
            #     height, width = depth_image.shape
            #     u, v = np.meshgrid(np.arange(width), np.arange(height))
            #     u = u.flatten()
            #     v = v.flatten()
            #     depth = depth_image.flatten()
            #
            #     fx, fy, cx, cy = intrinsic_matrix
            #     x = (u - cx) * depth / fx
            #     y = (v - cy) * depth / fy
            #     z = depth
            #
            #     point_cloud_camera = np.column_stack((x, y, z))
            #     point_cloud_world = np.matmul(point_cloud_camera, R) + T
            #
            #     savemat(f"render_depth_{_ + 7}.mat", {"depth_img": depth_img, "intrinsics": intrinsic_matrix, "R": R, "T": T})
            #
            #     return point_cloud_world

            # def depth_image_to_point_cloud(depth_image, intrinsic_matrix, R, T):
            #     height, width = depth_image.shape
            #     u, v = np.meshgrid(np.arange(width), np.arange(height))
            #     u, v = u.flatten(), v.flatten()
            #     depth = depth_image.flatten()
            #
            #     fx, fy, cx, cy = intrinsic_matrix
            #     x = (u - cx) * depth / fx
            #     y = (v - cy) * depth / fy
            #     z = depth
            #
            #     mask = depth > 2
            #     x, y, z = x[mask], y[mask], z[mask]
            #
            #     point_cloud_camera = np.column_stack((x, y, z))
            #     point_cloud_world = np.matmul(point_cloud_camera, R) + T
            #
            #     return point_cloud_world

            def fov2focal(fov, pixels):
                return pixels / (2 * math.tan(fov / 2))

            points = np.empty((0, 3))
            all_cams = scene.getTestCameras().copy()
            for _, viewpoint_cam in enumerate(all_cams[:10]):
            # viewpoint_cam = all_cams[0]
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                depth = render_pkg["depth"].squeeze(0).detach().cpu().numpy()

                # for k in range(4):
                #     mask = np.zeros((800, 800), dtype=bool)
                #     for i in range(1, 799):
                #         for j in range(1, 799):
                #             if depth[i, j] != 0 and (depth[i - 1][j] == 0 or depth[i + 1][j] == 0 or depth[i][j - 1] == 0 or depth[i][j + 1] == 0):
                #                 mask[i, j] = True
                #     depth[mask] = 0

                intrinsic_matrix = np.array([fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width), fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height), viewpoint_cam.image_width / 2, viewpoint_cam.image_height / 2])
                mat = np.zeros((4, 4))
                mat[:3, :3] = viewpoint_cam.R.transpose()
                mat[:3, 3] = viewpoint_cam.T
                mat[3, 3] = 1.0
                mat = np.linalg.inv(mat)
                mat[:3, :3] = mat[:3, :3].transpose()

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(depth_image_to_point_cloud(depth, intrinsic_matrix, mat[:3, :3], mat[:3, 3]))
                pcd, _ = pcd.remove_statistical_outlier(20, 0.5)
                points = np.concatenate((points, np.asarray(pcd.points)), axis=0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # pcd, _ = pcd.remove_statistical_outlier(20, 1.0)
            o3d.visualization.draw_geometries([pcd])

            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(20)
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            mesh = o3d.geometry.TriangleMesh(mesh)
            mesh = mesh.filter_smooth_simple(number_of_iterations=5)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([1, 0.706, 0])

            o3d.visualization.draw_geometries([mesh])

            # np.savez("alphas.npz", alpha=gaussians._opacity.detach().cpu().numpy())

            # import plotly.express as px
            # fig = px.scatter_3d(x=points[:, 0], y=points[:, 1], z=points[:, 2])
            # fig.show()

        '''
        if iteration % 500 == 0:
            import math
            import open3d as o3d
            def fov2focal(fov, pixels):
                return pixels / (2 * math.tan(fov / 2))

            points = np.empty((0, 3))
            all_cams = scene.getTestCameras().copy()
            for i, viewpoint_cam in enumerate(all_cams[:10]):
                # viewpoint_cam = all_cams[0]
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                depth = render_pkg["depth"]
                depth[depth < 10] = 0
                # depth = viewpoint_cam.depth.to(torch.uint8)
                depth = o3d.geometry.Image(depth.squeeze(0).unsqueeze(-1).detach().cpu().numpy())
                intrinsics = o3d.camera.PinholeCameraIntrinsic(viewpoint_cam.image_width, viewpoint_cam.image_height, fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width), fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height), viewpoint_cam.image_width / 2, viewpoint_cam.image_height / 2)
                extrinsics = viewpoint_cam.world_view_transform.detach().cpu().numpy().Ty
                pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics, extrinsics)
                tmp = np.asarray(pcd.points)
                points = np.concatenate((points, tmp), axis=0)  # for point in tmp:    #     points.append(point)
            # points = np.array(points)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            # pcd.remove_radius_outlier(16, 0.05)
            o3d.visualization.draw_geometries([pcd])
        '''

    print("Final # of gaussians:", gaussians._xyz.size(0))  # for i in range(10):  #     print(gaussians._xyz[i])
    np.savez("alphas.npz", alpha=gaussians._opacity)


# def generate_optim(gaussians, opt, cur_xyz_lr, t):
#     l = [
#         {'params': [gaussians._xyz], 'lr': cur_xyz_lr, "name": "xyz"},
#         {'params': [gaussians._features_dc], 'lr': opt.feature_lr, "name": "f_dc"},
#         {'params': [gaussians._features_rest], 'lr': opt.feature_lr / 20.0, "name": "f_rest"},
#         {'params': [gaussians._opacity], 'lr': opt.opacity_lr, "name": "opacity"},
#         {'params': [gaussians._scaling], 'lr': opt.scaling_lr, "name": "scaling"},
#         {'params': [gaussians._rotation], 'lr': opt.rotation_lr, "name": "rotation"}
#     ]
#     return CustomAdam(l, t, lr=0.0, epsilon=1e-15)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("../output/", unique_str[0:10])

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_500, 5_000, 7_500, 10_000, 15_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_500, 5_000, 7_500, 10_000])  # don't save basically
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
