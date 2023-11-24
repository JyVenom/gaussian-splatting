import open3d as o3d
import numpy as np


def run():
    target = o3d.io.read_point_cloud("../data/dtu/dtu_scan24/points3d_2.ply")
    source = o3d.io.read_point_cloud("../data/dtu/dtu_scan24/points3d.ply")

    target.colors = o3d.utility.Vector3dVector(np.asarray(source.colors))
    o3d.visualization.draw_geometries([target])
    o3d.io.write_point_cloud("points3d.ply", target)


if __name__ == '__main__':
    run()
