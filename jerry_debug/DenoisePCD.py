import numpy as np
import open3d as o3d


def run():
    all_points, all_colors = [], []
    for view in range(49):
        pcd = o3d.io.read_point_cloud(f"../jerry_out/all_views_pcds/{view}.ply")
        pcd, _ = pcd.remove_statistical_outlier(25, 2.0)
        # o3d.visualization.draw_geometries([pcd])
        all_points.append(np.asarray(pcd.points))
        all_colors.append(np.asarray(pcd.colors))

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.visualization.draw_geometries([pcd])
    pcd, _ = pcd.remove_statistical_outlier(50, 2.0)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    run()
