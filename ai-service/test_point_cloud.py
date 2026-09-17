import open3d as o3d


PLY_PATH = "room_point_cloud.ply"


print("==============================")
print("3D POINT CLOUD VIEWER")
print("==============================")


# -------------------------------------------------
# Load point cloud
# -------------------------------------------------

print("\nLoading point cloud...")

pcd = o3d.io.read_point_cloud(
    PLY_PATH
)


if pcd.is_empty():

    raise RuntimeError(
        f"Point cloud is empty: {PLY_PATH}"
    )


print(
    "Points:",
    len(pcd.points)
)


# -------------------------------------------------
# Check colors
# -------------------------------------------------

print(
    "Has colors:",
    pcd.has_colors()
)


# -------------------------------------------------
# Bounding box
# -------------------------------------------------

bbox = pcd.get_axis_aligned_bounding_box()

print("\nBounding box:")
print(bbox)


print(
    "\nMin bound:",
    bbox.min_bound
)

print(
    "Max bound:",
    bbox.max_bound
)


# -------------------------------------------------
# Estimate normals
# -------------------------------------------------

print("\nEstimating normals...")

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.05,
        max_nn=30
    )
)


# -------------------------------------------------
# Visualize
# -------------------------------------------------

print("\nOpening 3D viewer...")

o3d.visualization.draw_geometries(
    [pcd],
    window_name="3D Room Point Cloud",
    width=1280,
    height=720
)


print("\nDone.")
