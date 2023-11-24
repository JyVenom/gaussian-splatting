import torch
from tqdm import tqdm
import numpy as np
import open3d as o3d


def run():
    data = np.load("gaussian_means_pruned_1118.npz")
    xyz, scales = data["xyz"], data["scale"].max(axis=1) * 3

    rays, points = [], np.zeros((0, 3))
    xs = np.linspace(-1, 1, 400)
    ys = np.linspace(-1, 1, 400)
    for y in tqdm(ys):
        for x in xs:
            dists = (xyz[:, 0] - x)**2 + (xyz[:, 2] - y)**2
            mask = dists < scales**2
            if mask.sum() > 0:
                rays.append(xyz[mask, 1])
                points = np.concatenate((points, xyz[mask]))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.visualization.draw_geometries([pcd])

    sum, cnt = 0, 0
    for ray in rays:
        temp = np.array(ray)
        if temp.shape[0] > 16:
            minn = 1000
            for i in range(16, temp.shape[0]):
                minn = min(minn, temp[i] - temp[i - 16])
            sum += minn
            cnt += 1
    print(sum / cnt)


if __name__ == '__main__':
    run()
