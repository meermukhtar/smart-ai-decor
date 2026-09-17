import cv2
import numpy as np
import open3d as o3d


def create_point_cloud(
    image_path,
    depth_path
):
    """
    Create an RGB point cloud from
    an image and its depth map.
    """

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise FileNotFoundError(
            image_path
        )

    depth = cv2.imread(
        depth_path,
        cv2.IMREAD_UNCHANGED
    )

    if depth is None:
        raise FileNotFoundError(
            depth_path
        )

    # Convert BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize depth if required
    if depth.shape[:2] != image.shape[:2]:

        depth = cv2.resize(
            depth,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

    height, width = image.shape[:2]

    # -------------------------------------------------
    # Approximate camera parameters
    # -------------------------------------------------

    fx = width
    fy = width

    cx = width / 2
    cy = height / 2

    # -------------------------------------------------
    # Normalize depth
    # -------------------------------------------------

    depth = depth.astype(
        np.float32
    )

    depth_min = depth.min()
    depth_max = depth.max()

    depth = (
        depth - depth_min
    ) / (
        depth_max - depth_min + 1e-8
    )

    # Give depth an approximate
    # physical range.
    depth = (
        depth * 5.0
        + 0.1
    )

    # -------------------------------------------------
    # Pixel coordinates
    # -------------------------------------------------

    u, v = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )

    # -------------------------------------------------
    # Convert to 3D
    # -------------------------------------------------

    z = depth

    x = (
        (u - cx)
        * z
        / fx
    )

    y = (
        (v - cy)
        * z
        / fy
    )

    points = np.stack(
        [
            x,
            y,
            z
        ],
        axis=-1
    )

    points = points.reshape(
        -1,
        3
    )

    colors = (
        image.reshape(
            -1,
            3
        )
        / 255.0
    )

    # -------------------------------------------------
    # Remove invalid points
    # -------------------------------------------------

    valid = np.isfinite(
        points
    ).all(axis=1)

    points = points[
        valid
    ]

    colors = colors[
        valid
    ]

    # -------------------------------------------------
    # Create Open3D cloud
    # -------------------------------------------------

    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    cloud.colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )

    return cloud