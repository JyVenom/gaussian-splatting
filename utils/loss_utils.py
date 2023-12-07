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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from sklearn.cluster import DBSCAN

# from sklearnex import patch_sklearn, config_context
# patch_sklearn()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(0)
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

def bce_loss(network_output, gt):
    bce = torch.nn.BCELoss()
    return bce(network_output, gt)
    # epsilon = 1e-15  # Small constant to avoid division by zero
    # loss = - (gt * torch.log(network_output + epsilon) + (1 - gt) * torch.log(1 - network_output + epsilon))
    # return torch.mean(loss)

def depth_smoothness_loss(depth_map, image):
    dx = torch.abs(depth_map[:, :, :-1] - depth_map[:, :, 1:])
    dy = torch.abs(depth_map[:, :-1, :] - depth_map[:, 1:, :])

    image_dx = torch.abs(image[:, :, :-1] - image[:, :, 1:])
    image_dy = torch.abs(image[:, :-1, :] - image[:, 1:, :])

    return torch.mean(dx * image_dx) + torch.mean(dy * image_dy)

# def dbscan_loss(penalty, data):
#     with config_context(target_offload="gpu:0"):
#         labels = DBSCAN(eps=0.15, min_samples=25, n_jobs=-1).fit_predict(data.detach().cpu().numpy())
#
#         unique_labels, counts = np.unique(labels, return_counts=True)
#         largest_cluster_label = unique_labels[np.argmax(counts)]
#
#         cluster_penalty = (labels != largest_cluster_label).astype(np.float32)
#         cluster_penalty = torch.from_numpy(cluster_penalty).cuda()
#
#         return penalty * torch.mean(cluster_penalty)
#
# def dbscan_loss2(data):
#     data = data.detach().cpu().numpy()
#     with config_context(target_offload="gpu:0"):
#         labels = DBSCAN(eps=0.15, min_samples=25, n_jobs=-1).fit_predict(data)
#
#         unique_labels, counts = np.unique(labels, return_counts=True)
#         largest_cluster_label = unique_labels[np.argmax(counts)]
#
#         T = torch.tensor(data[labels == largest_cluster_label], device="cuda")
#         S = torch.tensor(data, device="cuda")
#
#         return torch.cdist(S, T, p=2).min(dim=1).values.mean()

def depth2pcd(depth_image, intrinsic_matrix, R, T):
    fx, fy, cx, cy = intrinsic_matrix
    height, width = depth_image.shape

    rows, cols = torch.tensor(range(height), device='cuda').unsqueeze(-1).expand(height, width), torch.tensor(range(width), device='cuda').expand(height, width)

    zs = depth_image
    xs = (cols - cx) * zs / fx
    ys = (rows - cy) * zs / fy
    mask = zs > 0
    pcd2 = torch.cat((xs[mask].unsqueeze(-1), ys[mask].unsqueeze(-1), zs[mask].unsqueeze(-1)), dim=-1)

    return torch.matmul(pcd2, R) + T

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
def processDepth(depth, viewpoint_cam):
    intrinsic_matrix = torch.tensor([fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width), fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height), viewpoint_cam.image_width / 2, viewpoint_cam.image_height / 2])
    mat = torch.eye(4, device='cuda')
    mat[:3, :3] = torch.tensor(viewpoint_cam.R, device='cuda').T
    mat[:3, 3] = torch.tensor(viewpoint_cam.T, device='cuda')
    mat = torch.linalg.inv(mat)
    pcd = depth2pcd(depth.squeeze(), intrinsic_matrix, mat[:3, :3].T, mat[:3, 3])
    torch.cuda.empty_cache()
    return pcd


def cosine_similarity_loss(predicted_normals, ground_truth_normals):
    # Normalize the vectors to unit length
    predicted_normals = F.normalize(predicted_normals, p=2, dim=1)
    ground_truth_normals = F.normalize(ground_truth_normals, p=2, dim=1)

    # Compute the cosine similarity
    similarity = torch.sum(predicted_normals * ground_truth_normals, dim=1)

    # Use the negative cosine similarity as the loss (to minimize it)
    return -torch.mean(similarity)
