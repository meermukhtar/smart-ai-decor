import os
import cv2
import numpy as np


def load_image(image_path):
    """
    Load an RGB image.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


def load_depth(depth_path):
    """
    Load depth map.

    Supports:
    - PNG
    - JPG
    - NumPy .npy files
    """

    if depth_path.endswith(".npy"):

        depth = np.load(depth_path)

    else:

        depth = cv2.imread(
            depth_path,
            cv2.IMREAD_UNCHANGED
        )

    if depth is None:
        raise FileNotFoundError(
            f"Could not load depth map: {depth_path}"
        )

    depth = depth.astype(np.float32)

    return depth


def normalize_depth(depth):
    """
    Normalize depth to 0-1.
    """

    min_depth = np.min(depth)
    max_depth = np.max(depth)

    if max_depth - min_depth < 1e-8:

        return np.zeros_like(
            depth,
            dtype=np.float32
        )

    normalized = (
        depth - min_depth
    ) / (
        max_depth - min_depth
    )

    return normalized.astype(
        np.float32
    )


def resize_depth(
    depth,
    width,
    height
):
    """
    Resize depth map to match image dimensions.
    """

    return cv2.resize(
        depth,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )


def create_point_cloud_and_mesh(
    image_rgb,
    depth_norm,
    target_width=256,
    max_edge_jump_ratio=0.12,
    d_min=1.2,
    d_max=5.0
):
    """
    Converts RGB image + depth map into a solid 3D surface mesh and point cloud
    with analytical surface normals and depth-discontinuity boundary filtering.

    Returns:
        points: (N, 3) XYZ coordinates (Three.js/OpenGL standard coordinates)
        colors: (N, 3) RGB uint8 colors
        normals: (N, 3) Unit normal vectors (pointing towards camera)
        faces: (M, 3) Triangular polygon indices (0-indexed)
    """
    orig_h, orig_w = image_rgb.shape[:2]
    scale = target_width / float(orig_w)
    target_height = int(orig_h * scale)

    img_small = cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth_norm, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    depth_small = cv2.bilateralFilter(depth_small.astype(np.float32), 5, 0.08, 4)

    h, w = target_height, target_width
    cx = w / 2.0
    cy = h / 2.0
    fx = w * 0.90
    fy = w * 0.90

    # Invert disparity: normalized depth 1 is close (d_min), 0 is far (d_max)
    inv_d_min = 1.0 / d_min
    inv_d_max = 1.0 / d_max
    z_grid = 1.0 / (np.clip(depth_small, 0.001, 1.0) * (inv_d_min - inv_d_max) + inv_d_max)

    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    x_grid = (u_grid - cx) * z_grid / fx
    y_grid = -(v_grid - cy) * z_grid / fy  # Invert Y so up is positive in 3D
    z_grid_3d = -z_grid  # Looking down negative Z (standard OpenGL/Three.js)

    pts_grid = np.stack([x_grid, y_grid, z_grid_3d], axis=-1)

    # 1. Compute Analytical Surface Normals via central differences
    dx = np.zeros_like(pts_grid)
    dy = np.zeros_like(pts_grid)

    dx[:, 1:-1] = pts_grid[:, 2:] - pts_grid[:, :-2]
    dx[:, 0] = pts_grid[:, 1] - pts_grid[:, 0]
    dx[:, -1] = pts_grid[:, -1] - pts_grid[:, -2]

    dy[1:-1, :] = pts_grid[2:, :] - pts_grid[:-2, :]
    dy[0, :] = pts_grid[1, :] - pts_grid[0, :]
    dy[-1, :] = pts_grid[-1, :] - pts_grid[-2, :]

    normals = np.cross(dx, dy)
    norm_len = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.where(norm_len > 1e-6, normals / (norm_len + 1e-8), np.array([0, 0, 1], dtype=np.float32))

    # Ensure normals face the viewer
    flip_mask = (normals[:, :, 2] < 0)
    normals[flip_mask] = -normals[flip_mask]

    # 2. Triangulate Surface Mesh with Depth-Discontinuity Filtering
    vert_indices = np.arange(h * w).reshape(h, w)
    tl = vert_indices[:-1, :-1].ravel()
    bl = vert_indices[1:, :-1].ravel()
    tr = vert_indices[:-1, 1:].ravel()
    br = vert_indices[1:, 1:].ravel()

    z_tl = z_grid[:-1, :-1].ravel()
    z_bl = z_grid[1:, :-1].ravel()
    z_tr = z_grid[:-1, 1:].ravel()
    z_br = z_grid[1:, 1:].ravel()

    max_jump = (d_max - d_min) * max_edge_jump_ratio

    # Triangle 1: (TL, BL, TR)
    jump_t1 = np.maximum(np.abs(z_tl - z_bl), np.maximum(np.abs(z_bl - z_tr), np.abs(z_tr - z_tl)))
    valid_t1 = jump_t1 < max_jump

    # Triangle 2: (TR, BL, BR)
    jump_t2 = np.maximum(np.abs(z_tr - z_bl), np.maximum(np.abs(z_bl - z_br), np.abs(z_br - z_tr)))
    valid_t2 = jump_t2 < max_jump

    faces_t1 = np.stack([tl[valid_t1], bl[valid_t1], tr[valid_t1]], axis=-1)
    faces_t2 = np.stack([tr[valid_t2], bl[valid_t2], br[valid_t2]], axis=-1)
    faces = np.vstack([faces_t1, faces_t2])

    flat_pts = pts_grid.reshape(-1, 3)
    flat_cols = img_small.reshape(-1, 3)
    flat_norms = normals.reshape(-1, 3)

    return flat_pts, flat_cols, flat_norms, faces


def create_point_cloud(image, depth, focal_length=None):
    """
    Backward-compatible basic point cloud helper.
    """
    depth_n = normalize_depth(depth) if depth.max() > 1.0 else depth
    pts, cols, _, _ = create_point_cloud_and_mesh(image, depth_n, target_width=min(image.shape[1], 256))
    return pts, cols


def save_point_cloud(points, colors, output_path, normals=None, faces=None):
    """
    Save point cloud as an enhanced PLY file (with surface normals and faces if available).
    """
    has_normals = normals is not None and len(normals) == len(points)
    has_faces = faces is not None and len(faces) > 0

    with open(output_path, "w", buffering=1048576) as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_normals:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        if has_faces:
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        if has_normals:
            for (px, py, pz), (nx, ny, nz), (r, g, b) in zip(points, normals, colors):
                f.write(f"{px:.4f} {py:.4f} {pz:.4f} {nx:.3f} {ny:.3f} {nz:.3f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for (px, py, pz), (r, g, b) in zip(points, colors):
                f.write(f"{px:.4f} {py:.4f} {pz:.4f} {int(r)} {int(g)} {int(b)}\n")

        if has_faces:
            for f1, f2, f3 in faces:
                f.write(f"3 {f1} {f2} {f3}\n")


def save_mesh_obj(points, colors, output_path, normals=None, faces=None):
    """
    Save 3D room mesh as a standard Wavefront OBJ file for WebGL / Three.js OBJLoader.
    """
    has_normals = normals is not None and len(normals) == len(points)
    has_faces = faces is not None and len(faces) > 0

    with open(output_path, "w", buffering=1048576) as f:
        f.write("# SmartSpace AI Universal 3D Room Mesh\n")
        f.write(f"# Vertices: {len(points)}, Faces: {len(faces) if has_faces else 0}\n")

        for (px, py, pz), (r, g, b) in zip(points, colors):
            f.write(f"v {px:.4f} {py:.4f} {pz:.4f} {r/255.0:.3f} {g/255.0:.3f} {b/255.0:.3f}\n")

        if has_normals:
            for nx, ny, nz in normals:
                f.write(f"vn {nx:.3f} {ny:.3f} {nz:.3f}\n")

        if has_faces:
            if has_normals:
                for f1, f2, f3 in faces:
                    f.write(f"f {f1+1}//{f1+1} {f2+1}//{f2+1} {f3+1}//{f3+1}\n")
            else:
                for f1, f2, f3 in faces:
                    f.write(f"f {f1+1} {f2+1} {f3+1}\n")


def create_depth_visualization(depth, output_path):
    """
    Create a visual depth image using COLORMAP_JET.
    """
    normalized = normalize_depth(depth)
    visual = (normalized * 255).astype(np.uint8)
    visual = cv2.applyColorMap(visual, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, visual)


def render_novel_view(
    image_bgr,
    depth_norm,
    yaw_deg=0.0,
    pitch_deg=0.0,
    tx=0.0,
    ty=0.0,
    tz=0.0,
    fov_scale=1.0,
    d_min=1.2,
    d_max=5.0
):
    """
    Renders a geometrically coherent novel camera viewpoint of the staged room.
    Applies 3D rotation and translation relative to the room center, followed by
    z-buffer projection and seamless edge/disocclusion inpainting.
    """
    h, w = image_bgr.shape[:2]

    # Camera Intrinsics
    cx = w / 2.0
    cy = h / 2.0
    fx = w * 0.90
    fy = w * 0.90

    # Metric depth from inverse disparity
    depth_clamped = np.clip(depth_norm, 0.001, 1.0)
    inv_d_min = 1.0 / d_min
    inv_d_max = 1.0 / d_max
    z_grid = 1.0 / (depth_clamped * (inv_d_min - inv_d_max) + inv_d_max)

    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    x_grid = (u_grid - cx) * z_grid / fx
    y_grid = (v_grid - cy) * z_grid / fy

    # Focus pivot
    z_pivot = float(np.median(z_grid))
    c_pivot = np.array([0.0, 0.0, z_pivot], dtype=np.float32)

    # 3D Rotation matrices
    rad_yaw = np.radians(yaw_deg)
    rad_pitch = np.radians(pitch_deg)

    Ry = np.array([
        [np.cos(rad_yaw), 0, np.sin(rad_yaw)],
        [0, 1, 0],
        [-np.sin(rad_yaw), 0, np.cos(rad_yaw)]
    ], dtype=np.float32)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rad_pitch), -np.sin(rad_pitch)],
        [0, np.sin(rad_pitch), np.cos(rad_pitch)]
    ], dtype=np.float32)

    R = Ry @ Rx

    # Transform 3D coordinates
    pts = np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3)
    colors = image_bgr.reshape(-1, 3)

    pts_rel = pts - c_pivot
    pts_rot = (R @ pts_rel.T).T
    pts_trans = pts_rot + c_pivot + np.array([tx, ty, tz], dtype=np.float32)

    # Project to virtual camera plane
    fx_new = fx * fov_scale
    fy_new = fy * fov_scale

    valid = pts_trans[:, 2] > 0.2
    pts_v = pts_trans[valid]
    cols_v = colors[valid]

    u_proj = (fx_new * pts_v[:, 0] / pts_v[:, 2] + cx)
    v_proj = (fy_new * pts_v[:, 1] / pts_v[:, 2] + cy)
    z_proj = pts_v[:, 2]

    out_img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Painter's order (farthest to nearest)
    sort_idx = np.argsort(-z_proj)
    u_sorted = u_proj[sort_idx]
    v_sorted = v_proj[sort_idx]
    z_sorted = z_proj[sort_idx]
    c_sorted = cols_v[sort_idx]

    # Splat 2x2 footprint to eliminate single-pixel pinholes
    for du in [0, 1]:
        for dv in [0, 1]:
            ui = np.clip(np.round(u_sorted).astype(np.int32) + du, 0, w - 1)
            vi = np.clip(np.round(v_sorted).astype(np.int32) + dv, 0, h - 1)

            in_bounds = (u_sorted >= 0) & (u_sorted < w) & (v_sorted >= 0) & (v_sorted < h)
            ui_b = ui[in_bounds]
            vi_b = vi[in_bounds]
            c_b = c_sorted[in_bounds]

            out_img[vi_b, ui_b] = c_b
            mask[vi_b, ui_b] = 255

    # Seamless inpainting for disocclusion cracks / border gaps
    holes = (mask == 0).astype(np.uint8)
    if np.any(holes):
        holes_dilated = cv2.dilate(holes, np.ones((3, 3), np.uint8))
        out_img = cv2.inpaint(out_img, holes_dilated, 3, cv2.INPAINT_TELEA)

    return out_img


def synthesize_novel_views(image_bgr, depth_norm, output_dir, file_prefix):
    """
    Synthesizes a comprehensive multi-angle view array of the staged room.
    Produces 5 labeled camera viewpoints for interactive frontend rotation:
    - Front View (0°)
    - Left Perspective (-14°)
    - Right Perspective (+14°)
    - Elevated Plan / Bird's Eye View (+10°)
    - Wide Angle Panoramic View

    Returns:
        List of view dictionaries containing angle metadata and file paths.
    """
    h, w = image_bgr.shape[:2]
    depth_smooth = cv2.bilateralFilter(depth_norm.astype(np.float32), 7, 0.1, 5)

    view_configs = [
        {
            "angle": "front",
            "label": "Front View (0°)",
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "fov_scale": 1.0,
        },
        {
            "angle": "left_perspective",
            "label": "Left Perspective (-14°)",
            "yaw_deg": -14.0,
            "pitch_deg": 0.0,
            "tx": -0.15,
            "ty": 0.0,
            "fov_scale": 1.0,
        },
        {
            "angle": "right_perspective",
            "label": "Right Perspective (+14°)",
            "yaw_deg": 14.0,
            "pitch_deg": 0.0,
            "tx": 0.15,
            "ty": 0.0,
            "fov_scale": 1.0,
        },
        {
            "angle": "top_elevated",
            "label": "Elevated Plan View (+10°)",
            "yaw_deg": 0.0,
            "pitch_deg": 10.0,
            "tx": 0.0,
            "ty": -0.18,
            "fov_scale": 1.04,
        },
        {
            "angle": "wide_angle",
            "label": "Wide Angle Panoramic View",
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "fov_scale": 1.18,
        },
    ]

    results = []
    for cfg in view_configs:
        file_name = f"{file_prefix}_view_{cfg['angle']}.jpg"
        out_path = os.path.join(output_dir, file_name)

        if cfg["angle"] == "front":
            cv2.imwrite(out_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            rendered = render_novel_view(
                image_bgr,
                depth_smooth,
                yaw_deg=cfg["yaw_deg"],
                pitch_deg=cfg["pitch_deg"],
                tx=cfg["tx"],
                ty=cfg["ty"],
                fov_scale=cfg["fov_scale"]
            )
            cv2.imwrite(out_path, rendered, [cv2.IMWRITE_JPEG_QUALITY, 95])

        results.append({
            "angle": cfg["angle"],
            "label": cfg["label"],
            "yaw_deg": cfg["yaw_deg"],
            "pitch_deg": cfg["pitch_deg"],
            "fov_scale": cfg["fov_scale"],
            "file_name": file_name,
            "file_path": out_path,
        })

    return results