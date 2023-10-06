import open3d as o3d


def run():
    pcd = o3d.io.read_point_cloud("../output/4692a701-9/point_cloud/iteration_10000/point_cloud.ply")
    points = pcd.points

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(50)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh = mesh.filter_smooth_simple(number_of_iterations=5)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1, 0.706, 0])

    o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    run()
