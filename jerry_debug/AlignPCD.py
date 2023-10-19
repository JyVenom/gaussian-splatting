import numpy as np
import open3d as o3d


def run():
    gt = o3d.io.read_point_cloud("points3D_4.ply")
    colmap = o3d.io.read_point_cloud("points3D.ply")

    points = np.asarray(colmap.points)
    colors = np.asarray(colmap.colors)
    labels = np.array(colmap.cluster_dbscan(eps=0.11, min_points=25))
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    most_common_label = unique_labels[np.argmax(label_counts)]
    colmap = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[labels == most_common_label]))
    colmap.colors = o3d.utility.Vector3dVector(colors[labels == most_common_label])
    o3d.visualization.draw_geometries([colmap, gt])

    # ab_dists = gt.compute_point_cloud_distance(colmap)
    # ab_dists = np.asarray(ab_dists)
    # print(ab_dists.mean())
    # ba_dists = colmap.compute_point_cloud_distance(gt)
    # ba_dists = np.asarray(ba_dists)
    # print(ba_dists.mean())
    # print(ab_dists.mean()+ba_dists.mean())

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source=gt,
        target=colmap,
        max_correspondence_distance=0.02,
        # estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    # print(reg_p2p)
    # print(reg_p2p.transformation)
    gt.transform(reg_p2p.transformation)
    o3d.visualization.draw_geometries([colmap, gt])

    ab_dists = gt.compute_point_cloud_distance(colmap)
    ab_dists = np.asarray(ab_dists)
    print(ab_dists.mean())
    ba_dists = colmap.compute_point_cloud_distance(gt)
    ba_dists = np.asarray(ba_dists)
    print(ba_dists.mean())
    print(ab_dists.mean()+ba_dists.mean())

    # o3d.io.write_point_cloud("points3D_4.ply")

if __name__ == "__main__":
    run()
