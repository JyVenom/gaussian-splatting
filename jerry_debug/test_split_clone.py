import sys
from argparse import ArgumentParser

import numpy as np
import torch

from arguments import OptimizationParams
from scene import GaussianModel


def run():
    torch.manual_seed(0)

    gaussians = GaussianModel(3)
    parser = ArgumentParser(description="Training script parameters")
    op = OptimizationParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[600, 3_000, 5_000, 10_000, 15_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[500, 7_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # gaussians.load_ply("./output/284456ea-a/point_cloud/iteration_4000/point_cloud.ply")

    data = np.load("iter_4000.npz")
    xyz, features_dc, features_rest, scaling, rotation, opacity, max_radii2d, xyz_grad_accum, denom, xyz_grads = data[
        "xyz"], data["features_dc"], data["features_rest"], data["scaling"], data["rotation"], data["opacity"], data[
        "max_radii2d"], data["xyz_grad_accum"], data["denom"], data["xyz_grads"]

    gaussians._xyz.data = torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
    gaussians._features_dc.data = torch.tensor(features_dc, dtype=torch.float, device="cuda").requires_grad_(True)
    gaussians._features_rest.data = torch.tensor(features_rest, dtype=torch.float, device="cuda").requires_grad_(True)
    gaussians._opacity.data = torch.tensor(opacity, dtype=torch.float, device="cuda").requires_grad_(True)
    gaussians._scaling.data = torch.tensor(scaling, dtype=torch.float, device="cuda").requires_grad_(True)
    gaussians._rotation.data = torch.tensor(rotation, dtype=torch.float, device="cuda").requires_grad_(True)
    gaussians.max_radii2D = torch.tensor(max_radii2d, dtype=torch.float, device="cuda")
    gaussians.xyz_gradient_accum = torch.tensor(xyz_grad_accum, dtype=torch.float, device="cuda")
    gaussians.denom = torch.tensor(denom, dtype=torch.float, device="cuda")
    gaussians._xyz.grads = torch.tensor(xyz_grads, dtype=torch.float, device="cuda")
    gaussians.active_sh_degree = 3

    gaussians.training_setup(op.extract(args))

    gaussians.gaussians_count = 406987
    # print("Org # of gaussians:", gaussians.gaussians_count)
    gaussians.densify_and_prune(0.0002, 0.005, 7.451763153076173, 20)
    print("Step")
    # gaussians.optimizer.step()
    print("Final # of gaussians:", gaussians.gaussians_count)
    for i in range(10):
        print(gaussians._xyz[i])


if __name__ == "__main__":
    run()
