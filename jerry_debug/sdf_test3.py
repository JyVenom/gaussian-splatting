import math
from math import exp

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import mcubes
import open3d as o3d


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {'include_input': True, 'input_dims': input_dims, 'max_freq_log2': multires - 1,
                    'num_freqs': multires, 'log_sampling': True, 'periodic_fns': [torch.sin, torch.cos], }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, n_layers, skip_in=(4,), multires=0, bias=0.5, scale=1.0,
                 geometric_init=True, weight_norm=True, inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = \
            torch.autograd.grad(outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True,
                                only_inputs=True)[0]
        return gradients.unsqueeze(1)


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def depth_image_to_point_cloud(depth_image, intrinsic_matrix, R, T):
    fx, fy, cx, cy = intrinsic_matrix

    pcd = np.zeros((200, int(cy * 2), int(cx * 2), 3))
    for i in tqdm(range(pcd.shape[1])):
        for j in range(pcd.shape[2]):
            zs = depth_image[:, i, j]
            xs = (j - cx) * zs / fx
            ys = (i - cy) * zs / fy
            pcd[:, i, j] = np.concatenate((np.array(xs[:, np.newaxis]), np.array(ys[:, np.newaxis]), np.array(zs[:, np.newaxis])), axis=-1)
            pcd[:, i, j] = np.matmul(pcd[:, i, j], R) + T

    return pcd


# Get camera intrinsics (shared)
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


class ToOpacity(nn.Module):
    def __init__(self):
        super(ToOpacity, self).__init__()

        self.s = torch.tensor(4.0)

    def forward(self, x):
        sig = (1 + torch.exp(-self.s * x)).pow(-1)
        ldd = (self.s * torch.exp(-self.s * x)) / ((1 + torch.exp(-self.s * x)).pow(2))
        return F.relu(ldd / sig)


# def sig(x, s):
#     return (1+torch.exp(-s*x))**-1
#
#
# def ldd(x, s):
#     return (s*torch.exp(-s*x))/((1+torch.exp(-s*x))**2)
#
# def to_opacity(x, s=1):
#     return ldd(x, s) / sig(x, s)

def run():
    # Load Gaussian Data
    data = np.load("../jerry_out/render_data.npz")
    points, probs, colors, opacs, gt = data["points"], data["probs"], data["color"], data["opacity"], data["gt"]

    intrinsic_matrix = np.array([724.428155, 965.904207, 200, 200])
    R = np.array([[0.94776466, -0.10209334, 0.30219049],
                  [0.03839772, 0.97702145, 0.20965379],
                  [-0.31665084, -0.18709902, 0.92990655]])
    T = np.array([0.72900188, 2.79314918, -0.93300664])
    points = depth_image_to_point_cloud(points, intrinsic_matrix, R, T)
    # # import open3d as o3d
    # # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.reshape(-1, 3)))
    # # margin = 0.25
    # # minn, maxx = pcd.get_min_bound(), pcd.get_max_bound()
    # # center = (minn + maxx) / 2
    # # diag = (maxx - minn).max() * (1 + margin) / 2
    # # print(center - diag, center + diag)
    # #
    # # print(pcd.get_min_bound(), pcd.get_max_bound())
    # # o3d.visualization.draw_geometries([pcd])
    #
    # #
    # #
    # # print(points)
    # # img = np.zeros((800, 800, 3))
    # # trans = np.ones((800, 800, 3))
    # # colors_t = np.concatenate((colors[0][..., np.newaxis], colors[1][..., np.newaxis], colors[2][..., np.newaxis]), axis=-1)
    # # for i in tqdm(range(200)):
    # #     alpha = (opacs[i] * probs[i]).clip(0, 0.99)
    # #     alpha = np.concatenate((alpha[..., np.newaxis], alpha[..., np.newaxis], alpha[..., np.newaxis]), axis=-1)
    # #     img += colors_t[i] * trans * alpha
    # #     trans = trans * (1 - alpha)
    # # for y in tqdm(range(res)):
    # #     Y = int(y / res * 800)
    # #     for x in range(res):
    # #         trans = 1.0
    # #         X = int(x / res * 800)
    # #         for i in range(200):
    # #             alpha = min(0.99, opacs[i, Y, X] * probs[i, Y, X])
    # #             for c in range(3):
    # #                 img[y, x, c] += colors[c, i, Y, X] * trans * alpha
    # #             trans = trans * (1 - alpha)
    # #
    # # import matplotlib.pyplot as plt
    # # plt.imshow(img)
    # # plt.axis('off')
    # # plt.tight_layout()
    # # plt.show()
    #
    # res = 400
    # gt = (gt*255).astype(np.uint8).transpose(1, 2, 0)
    # img = PIL.Image.fromarray(gt)
    # mask = PIL.Image.open("../data/dtu/dtu_scan24_2/masks/043.png")
    # img = img.resize((res, res))
    # mask = mask.resize((res, res))
    # mask = (np.array(mask) == 0)
    # img = np.array(img)
    # img[mask] = 0
    # gt_scaled = img.astype(np.float32) / 255.0
    # # gt_scaled = np.zeros((res, res, 3))
    # # for y in range(res):
    # #     Y = int(y / res * 800)
    # #     for x in range(res):
    # #         X = int(x / res * 800)
    # #         gt_scaled[y, x] = gt[:, Y, X]
    # # plt.imshow(gt_scaled)
    # # plt.axis('off')
    # # plt.show()
    #
    device = 'cuda'
    points = torch.tensor(points).to(torch.float32).to(device)
    probs = torch.tensor(probs).to(torch.float32).to(device)
    # # colors = torch.from_numpy(colors).to(torch.float32).to('cuda')
    colors_t = np.concatenate((colors[0][..., np.newaxis], colors[1][..., np.newaxis], colors[2][..., np.newaxis]), axis=-1)
    colors_t = torch.tensor(colors_t).to(torch.float32).to(device)
    # gt_scaled = torch.tensor(gt_scaled).to(device)
    #
    # sdf = SDFNetwork(d_out=257, d_in=3, d_hidden=256, n_layers=8, skip_in=[4], multires=6, bias=0.5, scale=1.0, geometric_init=True, weight_norm=True).to(device)
    to_opacity = ToOpacity().to(device)
    #
    # criterion = nn.MSELoss()
    # optim = torch.optim.Adam(sdf.parameters())
    #
    # num_iterations = 3125
    # for epoch in tqdm(range(num_iterations)):
    #     optim.zero_grad()
    #
    #     num_rays = 512
    #     rays = torch.randint(0, res, (num_rays, 2)).to(device)
    #     img = torch.zeros((num_rays, 3)).to(device)
    #     trans = torch.ones((num_rays, 3)).to(device)
    #     for i in range(60):
    #         sds = sdf.sdf(points[i, rays[:, 0], rays[:, 1]]).squeeze(-1)
    #         alpha = (to_opacity(sds) * probs[i, rays[:, 0], rays[:, 1]]).clip(0, 0.99)
    #         alpha = torch.cat((alpha.unsqueeze(-1), alpha.unsqueeze(-1), alpha.unsqueeze(-1)), dim=-1)
    #         img += colors_t[i, rays[:, 0], rays[:, 1]] * trans * alpha
    #         trans = trans * (1 - alpha)
    #
    #     loss = criterion(img, gt_scaled[rays[:, 0], rays[:, 1]])
    #     loss.backward()
    #     optim.step()  # lambda_dssim = 0.2  # Ll1 = l1_loss(img, gt)  # loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(img, gt))
    #
    #     if (epoch + 1) % 50 == 0:
    #         print(f'Epoch [{epoch + 1}/{num_iterations}], Loss: {loss.detach().cpu().item()}')
    #         imgf = np.zeros((res, res, 3))
    #         raysf = rays.detach().cpu().numpy()
    #         imgf[raysf[:, 0], raysf[:, 1]] = img.detach().cpu().numpy()
    #         plt.imshow(imgf)
    #         plt.axis('off')
    #         plt.show()
    #
    # # batch_size = 50
    # # img = torch.zeros((400, 400, 3), device=device)
    # # trans = torch.ones((400, 400, 3), device=device)
    # # for i in tqdm(range(29)):
    # #     for y in range(0, 400, batch_size):
    # #         for x in range(0, 400, batch_size):
    # #             sds = sdf.sdf(points[i, y:y + batch_size, x:x + batch_size].reshape(-1, 3))
    # #             alpha = to_opacity(sds).reshape(batch_size, batch_size)
    # #             alpha = alpha * probs[i, y:y + batch_size, x:x + batch_size]
    # #             alpha = alpha.clip(0, 0.99)
    # #             alpha = torch.cat((alpha.unsqueeze(-1), alpha.unsqueeze(-1), alpha.unsqueeze(-1)), dim=-1)
    # #             img[y:y + batch_size, x:x + batch_size] += colors_t[i, y:y + batch_size, x:x + batch_size] * trans[y:y + batch_size, x:x + batch_size] * alpha
    # #             trans[y:y + batch_size, x:x + batch_size] = trans[y:y + batch_size, x:x + batch_size] * (1 - alpha)
    # # plt.imshow(img)
    # # plt.axis('off')
    # # plt.show()
    #
    # torch.save(sdf.state_dict(), "sdf.pt")
    # print("Saved")

    sdf = SDFNetwork(d_out=257, d_in=3, d_hidden=256, n_layers=8, skip_in=[4], multires=6, bias=0.5, scale=1.0, geometric_init=True, weight_norm=True).cuda()
    sdf.load_state_dict(torch.load("sdf.pt"))
    sdf.eval()
    print("Loaded")

    img = torch.zeros((100, 100, 3), device=device)
    trans = torch.ones((100, 100, 3), device=device)
    for i in tqdm(range(60)):
        for y in range(0, 100, 1):
            for x in range(0, 100, 1):
                sds = sdf.sdf(points[i, y*4, x*4].reshape(-1, 3))
                alpha = to_opacity(sds).reshape(1, 1)
                alpha = alpha * probs[i, y*4, x*4]
                alpha = alpha.clip(0, 0.99)
                alpha = torch.cat((alpha, alpha, alpha), dim=-1).squeeze(0)
                img[y, x] += colors_t[i, y*4, x*4] * trans[y, x] * alpha
                trans[y, x] = trans[y, x] * (1 - alpha)
    plt.imshow(img.detach().cpu().numpy())
    plt.axis('off')
    plt.show()

    n = 128
    X, Y, Z = np.meshgrid(np.linspace(-6.61978651, 3.41620085, n)[:, np.newaxis], np.linspace(-3.76717679, 6.26881057, n)[:, np.newaxis], np.linspace(-1.93660538, 8.09938198, n)[:, np.newaxis])
    points = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), axis=-1)

    with torch.no_grad():
        dists = sdf.sdf(torch.tensor(points, dtype=torch.float32, device='cuda')).detach().cpu().numpy()
        dists = dists.reshape(n, n, n)
        # print(dists.min(), dists.max())
        # print(dists.reshape(-1, 1))
        # plt.scatter(x=range(n**3), y=dists.reshape(-1, 1))
        # plt.show()
        vertices, triangles = mcubes.marching_cubes(dists, 0.0)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
        o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    run()
