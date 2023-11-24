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
from multiprocessing import Pool


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


def depth_image_to_point_cloud(depth_image, intrinsic_matrix):
    fx, fy, cx, cy = intrinsic_matrix

    pcd = np.zeros((200, 400, 400, 3))
    # for i in tqdm(range(200)):
    #     for j in range(800):
    #         for k in range(800):
    #             z = depth_image[i, j, k]
    #             x = (k - cx) * z / fx
    #             y = (j - cy) * z / fy
    #             pcd[i, j, k] = np.array([x, y, z])
    for i in tqdm(range(400)):
        for j in range(400):
            zs = depth_image[:, i, j]
            xs = (j - cx) * zs / fx
            ys = (i - cy) * zs / fy
            pcd[:, i, j] = np.concatenate((np.array(xs[:, np.newaxis]), np.array(ys[:, np.newaxis]), np.array(zs[:, np.newaxis])), axis=-1)

    return pcd


# Get camera intrinsics (shared)
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


# def process(x, y, points, probs, colors):
#     color = torch.zeros(3)
#     trans = 1.0
#     for i in range(200):
#         alpha = min(0.99, sdf.sdf(points[i]) * probs[i])
#         for c in range(3):
#             color[c] += colors[c, i] * trans * alpha
#         trans = trans * (1 - alpha)
#
#     return x, y, color


class ToOpacity(nn.Module):
    def __init__(self):
        super(ToOpacity, self).__init__()

        self.s = torch.tensor(4.0)

    def forward(self, x):
        sig = (1 + torch.exp(-self.s * x)).pow(-1)
        ldd = (self.s * torch.exp(-self.s * x)) / ((1 + torch.exp(-self.s * x)).pow(2))
        return F.relu(ldd / sig)


def sig(x, s):
    return (1+torch.exp(-s*x))**-1


def ldd(x, s):
    return (s*torch.exp(-s*x))/((1+torch.exp(-s*x))**2)

def to_opacity(x, s=1):
    return ldd(x, s) / sig(x, s)

def run():
    # Load Gaussian Data
    data = np.load("../jerry_out/render_data.npz")
    points, probs, colors, opacs, gt = data["points"], data["probs"], data["color"], data["opacity"], data["gt"]

    # to_opacity = ToOpacity()
    #
    # sds = 7.5 - points[:, 400, 400]
    # sds = sds.astype(np.float64)
    # sds[20:120] = sds[0:100]
    # sds[0:20] = np.linspace(2.0, 0.7, 20)
    # sds[89:109] = np.linspace(-0.5, -5.0, 20)
    # opacity = np.zeros(199)
    # s = 4
    # for i in range(199):
    #     opacity[i] = to_opacity(sds[i])
    # #     opacity[i] = ((sigmoid(sds[i], s) - sigmoid(sds[i+1], s)) / (sigmoid(sds[i], s) * (sds[i] - sds[i+1])))
    # plt.scatter(x=sds[:108], y=(opacity/s).tolist()[:108])
    # plt.show()
    #
    # return

    intrinsic_matrix = np.array([1448.8563098982593, 1931.8084131976789, 200, 200])

    points = depth_image_to_point_cloud(points, intrinsic_matrix)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.reshape(-1, 3)))
    o3d.visualization.draw_geometries([pcd])

    # img = np.zeros((800, 800, 3))
    # trans = np.ones((800, 800, 3))
    # colors_t = np.concatenate((colors[0][..., np.newaxis], colors[1][..., np.newaxis], colors[2][..., np.newaxis]), axis=-1)
    # for i in tqdm(range(200)):
    #     alpha = (opacs[i] * probs[i]).clip(0, 0.99)
    #     alpha = np.concatenate((alpha[..., np.newaxis], alpha[..., np.newaxis], alpha[..., np.newaxis]), axis=-1)
    #     img += colors_t[i] * trans * alpha
    #     trans = trans * (1 - alpha)
    # for y in tqdm(range(res)):
    #     Y = int(y / res * 800)
    #     for x in range(res):
    #         trans = 1.0
    #         X = int(x / res * 800)
    #         for i in range(200):
    #             alpha = min(0.99, opacs[i, Y, X] * probs[i, Y, X])
    #             for c in range(3):
    #                 img[y, x, c] += colors[c, i, Y, X] * trans * alpha
    #             trans = trans * (1 - alpha)
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    res = 400
    gt = (gt*255).astype(np.uint8).transpose(1, 2, 0)
    img = PIL.Image.fromarray(gt)
    img = img.resize((res, res))
    gt_scaled = np.array(img)
    # gt_scaled = np.zeros((res, res, 3))
    # for y in range(res):
    #     Y = int(y / res * 800)
    #     for x in range(res):
    #         X = int(x / res * 800)
    #         gt_scaled[y, x] = gt[:, Y, X]
    plt.imshow(gt_scaled)
    plt.axis('off')
    plt.show()

    device = 'cuda'
    points = torch.from_numpy(points).to(torch.float32).to(device)
    probs = torch.from_numpy(probs).to(torch.float32).to(device)
    # colors = torch.from_numpy(colors).to(torch.float32).to('cuda')
    colors_t = np.concatenate((colors[0][..., np.newaxis], colors[1][..., np.newaxis], colors[2][..., np.newaxis]), axis=-1)
    colors_t = torch.from_numpy(colors_t).to(torch.float32).to(device)
    gt_scaled = torch.from_numpy(gt_scaled).to(device)

    sdf = SDFNetwork(d_out=257, d_in=3, d_hidden=256, n_layers=8, skip_in=[4], multires=6, bias=0.5, scale=1.0, geometric_init=True, weight_norm=True).to(device)
    # to_opacity = ToOpacity().to(device)

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(sdf.parameters())

    num_iterations = 3125
    for epoch in (range(num_iterations)):
        optim.zero_grad()

        # dists = sdf.sdf(points)
        # opacity = to_opacity(dists).reshape(200, 800, 800)

        # queries = []
        # for y in range(res):
        #     Y = int(y / res * 800)
        #     for x in range(res):
        #         X = int(x / res * 800)
        #         queries.append((x, y, points[:, Y, X], probs[:, Y, X], colors[..., Y, X]))
        # img = torch.zeros((800, 800, 3))
        # with Pool(10) as p:
        #     res = p.starmap(process, queries)
        #
        #     for x, y, color in res:
        #         for c in range(3):
        #             img[y, x, c] = color[c]

        num_rays = 512
        rays = np.random.randint(0, 800, (num_rays, 2))
        img = torch.zeros((num_rays, 3))
        trans = torch.ones((num_rays, 3))
        for i in tqdm(range(200)):
            sds = sdf.sdf(points[i, rays[:, 0], rays[:, 1]])
            alpha = to_opacity(sds)
            alpha = alpha * probs[i, rays[:, 0], rays[:, 1]]
            alpha = alpha.clip(0, 0.99)
            alpha = torch.cat((alpha.unsqueeze(-1), alpha.unsqueeze(-1), alpha.unsqueeze(-1)), dim=-1)
            img += colors_t[i, rays[:, 0], rays[:, 1]] * trans * alpha
            trans = trans * (1 - alpha)



        # batch_size = 200
        # img = torch.zeros((800, 800, 3), device=device)
        # trans = torch.ones((800, 800, 3), device=device)
        # for i in tqdm(range(200)):
        #     for y in range(0, 200, batch_size):
        #         for x in range(0, 200, batch_size):
        #             sds = sdf.sdf(points[i, y:y + batch_size, x:x + batch_size].reshape(-1, 3))
        #             alpha = to_opacity(sds).reshape(batch_size, batch_size)
        #             alpha = alpha * probs[i, y:y + batch_size, x:x + batch_size]
        #             alpha = alpha.clip(0, 0.99)
        #             # alpha = (to_opacity(sdf.sdf(points[i, y:y + batch_size, x:x + batch_size].reshape(-1, 3))).reshape(batch_size, batch_size) * probs[i, y:y + batch_size, x:x + batch_size]).clip(0, 0.99)
        #             alpha = torch.cat((alpha.unsqueeze(-1), alpha.unsqueeze(-1), alpha.unsqueeze(-1)), dim=-1)
        #             img[y:y + batch_size, x:x + batch_size] += colors_t[i, y:y + batch_size, x:x + batch_size] * trans[y:y + batch_size, x:x + batch_size] * alpha
        #             trans[y:y + batch_size, x:x + batch_size] = trans[y:y + batch_size, x:x + batch_size] * (1 - alpha)
        #
        #             del sds
        #             torch.cuda.empty_cache()


        # for y in tqdm(range(res)):
        #     Y = int(y / res * 800)
        #     for x in range(res):
        #         trans = 1.0
        #         X = int(x / res * 800)
        #         for i in range(200):
        #             alpha = (to_opacity(sdf.sdf(points[i, Y, X].unsqueeze(0))) * probs[i, Y, X]).clamp(0, 0.99)
        #             for c in range(3):
        #                 img[y, x, c] += colors[c, i, Y, X] * trans * alpha
        #             trans = trans * (1 - alpha)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        loss = criterion(img, gt_scaled[rays[:, 0], rays[:, 1]])
        loss.backward()
        optim.step()  # lambda_dssim = 0.2  # Ll1 = l1_loss(img, gt)  # loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(img, gt))

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_iterations}], Loss: {loss.detach().cpu().item()}')


if __name__ == '__main__':
    run()
