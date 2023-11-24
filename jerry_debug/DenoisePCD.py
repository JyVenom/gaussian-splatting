import numpy as np
import open3d as o3d


def run():
    pcd = o3d.io.read_point_cloud("points3D.ply")
    o3d.visualization.draw_geometries([pcd])

    labels = np.array(pcd.cluster_dbscan(eps=0.15, min_points=25))
    unique_labels, label_counts = np.unique(labels[labels >= 0], return_counts=True)
    most_common_label = unique_labels[np.argmax(label_counts)]
    mask = (labels == most_common_label)

    points = np.asarray(pcd.points)[mask]
    colors = np.asarray(pcd.colors)[mask]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("points3D_3.ply", pcd)


if __name__ == '__main__':
    run()
