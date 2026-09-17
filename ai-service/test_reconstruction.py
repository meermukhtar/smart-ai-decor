from room_analysis.reconstruction import (
    load_image,
    load_depth,
    create_point_cloud,
    save_point_cloud,
    create_depth_visualization
)


IMAGE_PATH = "rooms.jpeg"
DEPTH_PATH = room12.jpg_depth.png"

POINT_CLOUD_PATH = "room_point_cloud.ply"
DEPTH_VISUAL_PATH = "depth_visualization.jpg"


print("\n==============================")
print("3D ROOM RECONSTRUCTION")
print("==============================")


# -----------------------------------------
# Load image
# -----------------------------------------

image = load_image(
    IMAGE_PATH
)

print(
    "Image:",
    image.shape
)


# -----------------------------------------
# Load depth
# -----------------------------------------

depth = load_depth(
    DEPTH_PATH
)

print(
    "Depth:",
    depth.shape
)

print(
    "Depth min:",
    depth.min()
)

print(
    "Depth max:",
    depth.max()
)


# -----------------------------------------
# Depth visualization
# -----------------------------------------

create_depth_visualization(
    depth,
    DEPTH_VISUAL_PATH
)

print(
    "Saved:",
    DEPTH_VISUAL_PATH
)


# -----------------------------------------
# Create point cloud
# -----------------------------------------

points, colors = create_point_cloud(
    image,
    depth
)

print(
    "Point cloud points:",
    len(points)
)


# -----------------------------------------
# Save point cloud
# -----------------------------------------

save_point_cloud(
    points,
    colors,
    POINT_CLOUD_PATH
)

print(
    "Saved:",
    POINT_CLOUD_PATH
)


print("\n==============================")
print("RECONSTRUCTION COMPLETE")
print("==============================")