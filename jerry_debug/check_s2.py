import os

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from NeuS.fields import SDFNetwork


def run():
    data = np.load("gaussian_means_pruned_1118.npz")
    xyz, scales = data["xyz"], data["scale"][:, [1, 2]].max(axis=1) * 3

    xyz = torch.tensor(xyz, device='cuda')
    scales = torch.tensor(scales, device='cuda')

    # sdf_network = SDFNetwork(d_out=257, d_in=3, d_hidden=256, n_layers=8, skip_in=[4], multires=6, bias=0.5, scale=1.0, geometric_init=True, weight_norm=True, ).to("cuda")
    # checkpoint = torch.load(os.path.join(os.path.abspath(""), '../NeuS/checkpoints/ckpt_300000.pth'), map_location="cuda")
    # sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
    # sdfs = sdf_network.sdf(xyz).squeeze()
    #
    # from scipy.io import savemat
    # savemat("gaussian_means_scales_sdf_pruned_1119.mat", {"means": xyz.detach().cpu().numpy(), "scales": scales.detach().cpu().numpy(), "sdf": sdfs.detach().cpu().numpy()})
    #
    # return

    order = xyz[:, 0].sort(descending=True).indices
    xyz = xyz[order]
    scales = scales[order]

    reso = 400
    rows, cols = torch.tensor(range(reso), device='cuda').unsqueeze(-1).expand(reso, reso), torch.tensor(range(reso), device='cuda').expand(reso, reso)
    rows = rows / reso * 2 - 1
    cols = cols / reso * 2 - 1
    dists = (xyz[:, 1].unsqueeze(-1) - cols.reshape(-1)) ** 2 + (xyz[:, 2].unsqueeze(-1) - rows.reshape(-1)) ** 2
    mask = dists < scales.unsqueeze(-1) ** 2
    # points = torch.stack((rows.reshape(-1).expand(mask.shape)[mask], cols.reshape(-1).expand(mask.shape)[mask], xyz[:, 1].unsqueeze(-1).expand(mask.shape)[mask]), dim=-1).cpu().numpy()
    #
    # chunk_size = 1000000
    # for i in tqdm(range(0, points.shape[0], chunk_size)):
    #     chunk = points[i:i + chunk_size, :]
    #     pcd_chunk = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(chunk))
    #     o3d.io.write_point_cloud(f"large_point_cloud/large_point_cloud_chunk_{i // chunk_size}.ply", pcd_chunk)

    sdf_network = SDFNetwork(d_out=257, d_in=3, d_hidden=256, n_layers=8, skip_in=[4], multires=6, bias=0.5, scale=1.0, geometric_init=True, weight_norm=True, ).to("cuda")
    checkpoint = torch.load(os.path.join(os.path.abspath(""), '../NeuS/checkpoints/ckpt_300000.pth'), map_location="cuda")
    sdf_network.load_state_dict(checkpoint['sdf_network_fine'])

    mask = mask.reshape(-1, 400, 400)
    a, b, c, d = 0, 0, 0, 0
    sdfs = sdf_network.sdf(xyz).squeeze()
    for i in tqdm(range(reso)):
        for j in range(reso):
            mask2 = mask[:, i, j]
            sdf_ray = sdfs[mask2]
            mask3 = sdf_ray.abs() < 0.1
            points = xyz[mask2]

            for k in range(1, mask3.shape[0]):
                if sdf_ray[k-1] > 0 and sdf_ray[k] < 0:
                    end = (points[k-1, 0] + points[k, 0]) - points[0, 0]
                    mask3 = torch.logical_and(mask3, points[:, 0] > end)
                    break

            ray = xyz[mask2][mask3, 0]
            n = ray.shape[0]
            if n > 10:
                dist = (ray[-1] - ray[0]) / n
                a += dist.item()
                b += 1
            if n > 8:
                dists = ray[:-1] - ray[1:]
                minn = dists.unfold(0, 8, 1).mean(dim=1).min().item()
                c += minn
                d += 1
    print(a / b)
    print(c / d)


if __name__ == '__main__':
    run()
