import open3d as o3d


PLY_PATH = "room_point_cloud.ply"


print("Loading point cloud...")

pcd = o3d.io.read_point_cloud(
    PLY_PATH
)


print(
    "Points:",
    len(pcd.points)
)


print(
    "Has colors:",
    pcd.has_colors()
)


print(
    "Bounding box:"
)

print(
    pcd.get_axis_aligned_bounding_box()
)


print("\nOpening 3D viewer...")

o3d.visualization.draw_geometries(
    [pcd],
    window_name="3D Room Point Cloud"
)