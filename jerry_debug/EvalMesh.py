import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


def run():
    neus = o3d.io.read_triangle_mesh("neus_bird.ply")
    mine = o3d.io.read_triangle_mesh("my_bird.ply")
    gt = o3d.io.read_point_cloud("bird_gt.ply")

    num_points = 40_000
    neus = neus.sample_points_poisson_disk(number_of_points=num_points)
    mine = mine.sample_points_poisson_disk(number_of_points=num_points)
    mine.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source=neus,
        target=gt,
        max_correspondence_distance=0.02,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    neus.transform(reg_p2p.transformation)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source=mine,
        target=gt,
        max_correspondence_distance=0.02,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    mine.transform(reg_p2p.transformation)

    # o3d.visualization.draw_geometries([neus])
    # o3d.visualization.draw_geometries([mine])
    o3d.visualization.draw_geometries([neus, mine, gt])

    thresh = 0.5
    gt = np.asarray(gt.points)
    neus = np.asarray(neus.points)
    mine = np.asarray(mine.points)

    print("------NeuS------")
    ab, _ = KDTree(gt).query(neus, k=1, workers=-1)
    ab = ab[ab < thresh]
    print("1-sided CD:", np.mean(np.square(ab)))
    ba, _ = KDTree(neus).query(gt, k=1, workers=-1)
    ba = ba[ba < thresh]
    print("CD:", np.mean(np.square(ab)) + np.mean(np.square(ba)))

    print("------Mine------")
    ab, _ = KDTree(gt).query(mine, k=1, workers=-1)
    ab = ab[ab < thresh]
    print("1-sided CD:", np.mean(np.square(ab)))
    ba, _ = KDTree(mine).query(gt, k=1, workers=-1)
    ba = ba[ba < thresh]
    print("CD:", np.mean(np.square(ab)) + np.mean(np.square(ba)))


if __name__ == "__main__":
    run()
