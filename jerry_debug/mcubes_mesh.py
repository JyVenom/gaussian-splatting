import mcubes
import numpy as np
import torch
import trimesh
from tqdm import tqdm

n = 64


def pdf_3d_gaussian(p, mean, inv, det):
    d = p - mean
    return det * torch.exp(-0.5 * torch.einsum('ij,ij->i', torch.matmul(d, inv), d))


def run():
    data = np.load("data_5000.npz")
    means, alphas, invs, dets = data["means"], data["alphas"], data["inv"], data["det"]

    mask = (alphas > 0.005).squeeze(-1)
    means = means[mask]
    alphas = alphas[mask]
    invs = invs[mask]
    dets = dets[mask]

    minn, maxx = means.min(axis=0), means.max(axis=0)
    print(minn, maxx)
    dist = 1.2 * (maxx - minn).max()
    queries = np.concatenate([arr.reshape(-1, 1) for arr in np.meshgrid(*[np.linspace(-dist / 2, dist / 2, num=n)] * 3, indexing="ij")], axis=1) + (minn + maxx) / 2

    means = torch.from_numpy(means).cuda()
    alphas = torch.from_numpy(alphas).cuda()
    invs = torch.from_numpy(invs).cuda()
    dets = torch.from_numpy(dets).cuda()
    queries = torch.from_numpy(queries).cuda()

    occ = torch.zeros((n ** 3)).cuda()
    for i in tqdm(range(means.shape[0])):
        occ += pdf_3d_gaussian(queries, means[i], invs[i], dets[i]) * alphas[i]
    occ = occ.cpu().numpy().reshape(n, n, n)

    np.savez("occ.npz", occ=occ)
    # data = np.load("occ.npz")
    # occ = data["occ"]

    vertices, triangles = mcubes.marching_cubes(occ, 0.5)
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.show()


if __name__ == '__main__':
    run()
