import os
import cv2
import numpy as np
import torch
from PIL import Image

from room_analysis.furniture import FURNITURE
from room_analysis.furniture_renderer import (
    render_furniture,
    load_furniture,
    crop_transparent,
    resize_furniture,
    overlay_image,
    create_shadow,
)
from room_analysis.scale import calculate_scale, meters_to_pixels
from room_analysis.regions import find_free_regions
from room_analysis.placement import check_bounding_box_fit, find_best_position
from room_analysis.depth import estimate_depth
from room_analysis.reconstruction import (
    create_point_cloud,
    create_point_cloud_and_mesh,
    save_point_cloud,
    save_mesh_obj,
    create_depth_visualization,
    synthesize_novel_views,
)
from room_analysis.staging import build_staging_prompt, AIStagingEngine

AI_SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RoomDecorationPipeline:
    """
    End-to-end intelligent room decoration & space suggestion engine.
    """

    _yolo_model = None
    _seg_processor = None
    _seg_model = None

    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    @classmethod
    def get_yolo(cls):
        if cls._yolo_model is None:
            from ultralytics import YOLO
            model_path = os.path.join(AI_SERVICE_DIR, "yolo11n-seg.pt")
            if not os.path.exists(model_path):
                model_path = "yolo11n-seg.pt"
            cls._yolo_model = YOLO(model_path)
        return cls._yolo_model

    @classmethod
    def get_segformer(cls):
        if cls._seg_model is None:
            from transformers import (
                AutoImageProcessor,
                SegformerForSemanticSegmentation,
            )
            seg_model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
            cls._seg_processor = AutoImageProcessor.from_pretrained(seg_model_name)
            cls._seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_model_name)
            if torch.cuda.is_available():
                cls._seg_model = cls._seg_model.to("cuda")
        return cls._seg_processor, cls._seg_model

    def detect_objects(self, image_path, confidence=0.25):
        """Run YOLO segmentation to detect existing furniture and obstacles."""
        yolo = self.get_yolo()
        results = yolo(image_path, conf=confidence)

        detections = []
        occupied_mask = None
        orig_h, orig_w = None, None

        for result in results:
            orig_h, orig_w = result.orig_shape
            occupied_mask = np.zeros((orig_h, orig_w), dtype=bool)

            if result.boxes is None or result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()

            for index, mask in enumerate(masks):
                mask_full = cv2.resize(
                    mask,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST
                ) > 0.5

                class_id = int(result.boxes.cls[index])
                score = float(result.boxes.conf[index])
                class_name = result.names[class_id]

                x1, y1, x2, y2 = result.boxes.xyxy[index].cpu().numpy().astype(int)
                w = int(x2 - x1)
                h = int(y2 - y1)

                detections.append({
                    "name": class_name,
                    "confidence": round(score, 3),
                    "bbox": {"x": int(x1), "y": int(y1), "width": w, "height": h},
                    "mask": mask_full,
                })

                occupied_mask |= mask_full

        if occupied_mask is None:
            img = cv2.imread(image_path)
            orig_h, orig_w = img.shape[:2]
            occupied_mask = np.zeros((orig_h, orig_w), dtype=bool)

        return detections, occupied_mask, (orig_h, orig_w)

    def segment_room_surfaces(self, pil_image, occupied_mask):
        """
        Run Segformer to extract floor, wall, ceiling, and window masks.
        """
        processor, seg_model = self.get_segformer()
        width, height = pil_image.size

        inputs = processor(images=pil_image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = seg_model(**inputs)

        logits = outputs.logits
        logits = torch.nn.functional.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )

        prediction = logits.argmax(dim=1)[0].cpu().numpy()

        # ADE20K classes:
        # 0: wall, 3: floor, 5: ceiling, 8: windowpane
        floor_mask = (prediction == 3)
        wall_mask = (prediction == 0)
        ceiling_mask = (prediction == 5)
        window_mask = (prediction == 8)

        # Fallback if floor mask is too tiny
        if np.count_nonzero(floor_mask) < (width * height * 0.05):
            floor_mask = np.zeros((height, width), dtype=bool)
            floor_mask[int(height * 0.60):, :] = True

        # Clean noise
        kernel = np.ones((5, 5), np.uint8)
        floor_uint8 = (floor_mask.astype(np.uint8) * 255)
        floor_uint8 = cv2.morphologyEx(floor_uint8, cv2.MORPH_OPEN, kernel)
        floor_uint8 = cv2.morphologyEx(floor_uint8, cv2.MORPH_CLOSE, kernel)
        floor_mask = (floor_uint8 > 0)

        # Subtract furniture obstacles from floor
        occupied_floor = floor_mask & occupied_mask
        free_floor = floor_mask & ~occupied_floor

        # Free wall: wall that is not covered by ceiling, floor, or windows
        free_wall = wall_mask & ~ceiling_mask & ~window_mask & ~occupied_mask

        return {
            "floor_mask": floor_mask,
            "free_floor": free_floor,
            "wall_mask": wall_mask,
            "free_wall": free_wall,
            "ceiling_mask": ceiling_mask,
            "window_mask": window_mask,
        }

    def determine_room_type(self, requested_type, detections):
        """Determine room type automatically if requested_type is 'auto'."""
        if requested_type and requested_type != "auto":
            return requested_type.lower()

        detected_names = [d["name"].lower() for d in detections]

        if any("bed" in name for name in detected_names):
            return "bedroom"
        elif any(name in ["couch", "sofa"] for name in detected_names):
            return "living_room"
        elif any(name in ["desk", "office chair", "laptop"] for name in detected_names):
            return "office"
        elif any("dining" in name for name in detected_names):
            return "dining_room"

        return "living_room"

    def generate_recommendations(self, room_type, detections, free_space_pct):
        """
        Generate contextual smart suggestions based on detected items and room type.
        """
        detected_names = [d["name"].lower() for d in detections]
        has_bed = any("bed" in n for n in detected_names)
        has_couch = any(n in ["couch", "sofa"] for n in detected_names)
        has_desk = any("desk" in n for n in detected_names)

        recommendations = []

        if room_type == "bedroom":
            if has_bed:
                recommendations.append({
                    "item_name": "study_desk",
                    "catalog": "bedroom",
                    "title": "Study & Work Desk",
                    "category": "work_study",
                    "reason": "Since your bed is already in place, utilize open floor space for a dedicated study or remote work desk.",
                    "priority": 1,
                    "is_wall_item": False,
                })
                recommendations.append({
                    "item_name": "study_chair",
                    "catalog": "bedroom",
                    "title": "Ergonomic Desk Chair",
                    "category": "work_study",
                    "reason": "Comfortable seating paired with your study table.",
                    "priority": 2,
                    "is_wall_item": False,
                })
                recommendations.append({
                    "item_name": "gaming_tv_unit",
                    "catalog": "bedroom",
                    "title": "TV Unit & PlayStation Gaming Setup",
                    "category": "entertainment",
                    "reason": "Transform the free wall/floor zone into an entertainment & gaming console lounge.",
                    "priority": 3,
                    "is_wall_item": False,
                })
                recommendations.append({
                    "item_name": "lounge_couch",
                    "catalog": "bedroom",
                    "title": "Relaxation Couch / Accent Chair",
                    "category": "seating",
                    "reason": "Cozy single couch positioned near the gaming/TV area for downtime.",
                    "priority": 4,
                    "is_wall_item": False,
                })
                recommendations.append({
                    "item_name": "wall_art",
                    "catalog": "bedroom",
                    "title": "Aesthetic Wall Canvas Art",
                    "category": "wall_decor",
                    "reason": "Add artistic flair to the empty wall above the desk or bedside.",
                    "priority": 5,
                    "is_wall_item": True,
                })
                recommendations.append({
                    "item_name": "wall_shelf",
                    "catalog": "bedroom",
                    "title": "Floating Book & Plant Wall Shelf",
                    "category": "wall_decor",
                    "reason": "Space-saving vertical storage for books and potted greens.",
                    "priority": 6,
                    "is_wall_item": True,
                })
            else:
                recommendations.append({
                    "item_name": "double_bed",
                    "catalog": "bedroom",
                    "title": "Queen / King Master Bed",
                    "category": "bed",
                    "reason": "Primary centerpiece for the bedroom.",
                    "priority": 1,
                    "is_wall_item": False,
                })
                recommendations.append({
                    "item_name": "study_desk",
                    "catalog": "bedroom",
                    "title": "Study Table",
                    "category": "work_study",
                    "reason": "Corner workstation setup.",
                    "priority": 2,
                    "is_wall_item": False,
                })
                recommendations.append({
                    "item_name": "wardrobe",
                    "catalog": "bedroom",
                    "title": "Storage Almari / Wardrobe",
                    "category": "storage",
                    "reason": "Essential clothes and essentials storage.",
                    "priority": 3,
                    "is_wall_item": False,
                })

        elif room_type == "gym":
            recommendations.append({
                "item_name": "dumbbells_rack",
                "catalog": "gym",
                "title": "3-Tier Dumbbell Rack",
                "category": "fitness",
                "reason": "Commercial-grade dumbbell weight station organized neatly on the floor.",
                "priority": 1,
                "is_wall_item": False,
            })
            recommendations.append({
                "item_name": "fitness_mirror",
                "catalog": "gym",
                "title": "Full-Length Gym Mirror",
                "category": "fitness_mirror",
                "reason": "Form check mirror reflecting light and expanding room depth.",
                "priority": 2,
                "is_wall_item": False,
            })
            recommendations.append({
                "item_name": "pull_up_bar",
                "catalog": "gym",
                "title": "Wall-Mounted Multi-Grip Pull-Up Bar",
                "category": "fitness_wall",
                "reason": "Mounted on upper wall for upper-body bodyweight calisthenics.",
                "priority": 3,
                "is_wall_item": True,
            })

        elif room_type == "office":
            recommendations.append({
                "item_name": "office_desk",
                "catalog": "office",
                "title": "Executive Office Desk",
                "category": "work",
                "reason": "Central productive workstation with clean surface.",
                "priority": 1,
                "is_wall_item": False,
            })
            recommendations.append({
                "item_name": "office_chair",
                "catalog": "office",
                "title": "Ergonomic Swivel Chair",
                "category": "work",
                "reason": "High-back lumbar support office chair.",
                "priority": 2,
                "is_wall_item": False,
            })
            recommendations.append({
                "item_name": "bookshelf",
                "catalog": "office",
                "title": "Bookshelf & Document Rack",
                "category": "storage",
                "reason": "Organize books, binders, and decorative pieces.",
                "priority": 3,
                "is_wall_item": False,
            })
            recommendations.append({
                "item_name": "indoor_plant",
                "catalog": "office",
                "title": "Indoor Fiddle-Leaf Fig Plant",
                "category": "decor",
                "reason": "Brings natural warmth and biophilic wellness into office space.",
                "priority": 4,
                "is_wall_item": False,
            })
            recommendations.append({
                "item_name": "wall_art",
                "catalog": "office",
                "title": "Modern Wall Art",
                "category": "wall_decor",
                "reason": "Professional framed art to enhance wall aesthetics.",
                "priority": 5,
                "is_wall_item": True,
            })

        elif room_type == "reading_nook":
            recommendations.append({
                "item_name": "bookshelf",
                "catalog": "reading_nook",
                "title": "Vertical Bookrack",
                "category": "storage",
                "reason": "Fits compactly into tight spaces or corners for a cozy library feel.",
                "priority": 1,
                "is_wall_item": False,
            })
            recommendations.append({
                "item_name": "reading_chair",
                "catalog": "reading_nook",
                "title": "Cozy Armchair Lounger",
                "category": "seating",
                "reason": "Perfect spot to curl up with a book.",
                "priority": 2,
                "is_wall_item": False,
            })
            recommendations.append({
                "item_name": "wall_shelf",
                "catalog": "reading_nook",
                "title": "Floating Plant & Book Shelf",
                "category": "wall_decor",
                "reason": "Decorative vertical display.",
                "priority": 3,
                "is_wall_item": True,
            })
            recommendations.append({
                "item_name": "indoor_plant",
                "catalog": "reading_nook",
                "title": "Indoor Plant",
                "category": "decor",
                "reason": "Greenery accent.",
                "priority": 4,
                "is_wall_item": False,
            })

        else: # living_room default
            recommendations.append({
                "item_name": "2_seater_sofa",
                "catalog": "living_room",
                "title": "Modern 2-Seater Sofa",
                "category": "seating",
                "reason": "Essential comfortable seating for lounging and guests.",
                "priority": 1,
                "is_wall_item": False,
            })
            recommendations.append({
                "item_name": "coffee_table",
                "catalog": "living_room",
                "title": "Minimalist Coffee Table",
                "category": "table",
                "reason": "Centerpiece table placed in front of sofa.",
                "priority": 2,
                "is_wall_item": False,
            })
            recommendations.append({
                "item_name": "gaming_tv_unit",
                "catalog": "living_room",
                "title": "TV Console & PlayStation Gaming Station",
                "category": "entertainment",
                "reason": "Main entertainment wall unit with gaming console setup.",
                "priority": 3,
                "is_wall_item": False,
            })
            recommendations.append({
                "item_name": "indoor_plant",
                "catalog": "living_room",
                "title": "Potted Indoor Plant",
                "category": "decor",
                "reason": "Fresh biophilic accent next to the TV or seating.",
                "priority": 4,
                "is_wall_item": False,
            })
            recommendations.append({
                "item_name": "wall_art",
                "catalog": "living_room",
                "title": "Framed Abstract Wall Canvas",
                "category": "wall_decor",
                "reason": "Adds color and personality to empty walls.",
                "priority": 5,
                "is_wall_item": True,
            })

        return recommendations

    def find_wall_placement(
        self, width, height, item_w_px, item_h_px, placed_boxes,
        wall_mask=None, ceiling_mask=None, window_mask=None
    ):
        """Find a clean, visually appealing upper wall location avoiding ceiling and windows with 100% strict clearance."""
        wall_y_min = int(height * 0.16)
        wall_y_max = int(height * 0.44)

        # Generate dense candidate X positions across the room width
        candidate_xs = []
        for x_ratio in [0.45, 0.40, 0.50, 0.35, 0.55, 0.30, 0.60, 0.25, 0.20, 0.15, 0.70]:
            cx = int(width * x_ratio - item_w_px // 2)
            if 25 <= cx <= (width - item_w_px - 25):
                candidate_xs.append(cx)

        best_cand = None
        best_wall_score = -1

        for cy in range(wall_y_min, max(wall_y_min + 1, wall_y_max - item_h_px), 15):
            for cx in candidate_xs:
                if cx < 25 or (cx + item_w_px) > (width - 25):
                    continue
                if (cy + item_h_px) > (height * 0.50):
                    continue

                box = (cx, cy, cx + item_w_px, cy + item_h_px)

                # Check overlap with any placed item (with 15px clearance)
                overlaps = False
                for p in placed_boxes:
                    px1, py1, px2, py2 = p
                    if not (box[2] <= px1 - 10 or box[0] >= px2 + 10 or box[3] <= py1 - 10 or box[1] >= py2 + 10):
                        overlaps = True
                        break
                if overlaps:
                    continue

                # STRICT WINDOW CHECK: Absolutely 0% overlap allowed with windows!
                if window_mask is not None and window_mask.shape[:2] == (height, width):
                    # Add 15px padding around window check to prevent grazing borders
                    wy1 = max(0, cy - 10)
                    wy2 = min(height, cy + item_h_px + 10)
                    wx1 = max(0, cx - 15)
                    wx2 = min(width, cx + item_w_px + 15)
                    patch_win = window_mask[wy1:wy2, wx1:wx2]
                    if np.sum(patch_win > 0) > 0:
                        continue  # Zero tolerance: do NOT place on or near windows!

                # CEILING CHECK
                if ceiling_mask is not None and ceiling_mask.shape[:2] == (height, width):
                    patch_ceil = ceiling_mask[cy:cy + item_h_px, cx:cx + item_w_px]
                    if np.mean(patch_ceil > 0) > 0.05:
                        continue

                # WALL SCORE
                score = 100
                if wall_mask is not None and wall_mask.shape[:2] == (height, width):
                    patch_wall = wall_mask[cy:cy + item_h_px, cx:cx + item_w_px]
                    wall_ratio = np.mean(patch_wall > 0)
                    if wall_ratio < 0.60:
                        continue
                    # Prefer centered positions on blank wall
                    dist_to_center = abs((cx + item_w_px / 2.0) - (width * 0.45))
                    score = int(wall_ratio * 150) - int(dist_to_center * 0.08)

                if score > best_wall_score:
                    best_wall_score = score
                    best_cand = {"x": cx, "y": cy, "width": item_w_px, "height": item_h_px}

        if best_cand is None:
            # Fallback to safe wall candidates with strict window validation
            for fx_ratio in [0.42, 0.35, 0.25, 0.20, 0.50]:
                fx = int(width * fx_ratio - item_w_px // 2)
                fy = int(height * 0.24)
                if fx + item_w_px <= width - 25 and fy + item_h_px <= int(height * 0.50) and fx >= 25:
                    box = (fx, fy, fx + item_w_px, fy + item_h_px)
                    overlaps = any(
                        not (box[2] <= p[0] - 10 or box[0] >= p[2] + 10 or box[3] <= p[1] - 10 or box[1] >= p[3] + 10)
                        for p in placed_boxes
                    )
                    if not overlaps:
                        # Validate against window_mask even in fallback!
                        if window_mask is not None and window_mask.shape[:2] == (height, width):
                            patch_win = window_mask[fy:fy + item_h_px, fx:fx + item_w_px]
                            if np.sum(patch_win > 0) > 0:
                                continue
                        best_cand = {"x": fx, "y": fy, "width": item_w_px, "height": item_h_px}
                        break

        return best_cand

    def process(
        self,
        image_path,
        room_type="auto",
        room_width_m=4.5,
        output_dir=None,
        file_prefix="room",
        generate_3d=True,
        description=None,
        style="modern_luxury",
    ):
        """
        Execute full pipeline:
        1. Object detection & Floor/Wall segmentation
        2. Free space extraction & region classification
        3. Recommendation generation & AI Staging prompt synthesis
        4. Perspective-aware coordinated placement & rendering
        5. 3D depth & point cloud generation
        """
        if output_dir is None:
            output_dir = os.path.join(AI_SERVICE_DIR, "output")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load image
        original_bgr = cv2.imread(image_path)
        if original_bgr is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        height, width = original_bgr.shape[:2]
        pil_image = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))

        # Scale
        meters_per_pixel = calculate_scale(width, room_width_m)

        # 2. Detect objects & Floor surfaces
        detections, occupied_mask, _ = self.detect_objects(image_path)
        surfaces = self.segment_room_surfaces(pil_image, occupied_mask)
        floor_mask = surfaces["floor_mask"]
        free_floor = surfaces["free_floor"]
        wall_mask = surfaces["wall_mask"]
        ceiling_mask = surfaces["ceiling_mask"]
        window_mask = surfaces["window_mask"]

        # Free floor stats
        floor_pixels = int(floor_mask.sum())
        free_pixels = int(free_floor.sum())
        free_space_pct = round((free_pixels / max(floor_pixels, 1)) * 100, 1)
        est_floor_area_sqm = round(floor_pixels * (meters_per_pixel ** 2), 2)
        est_free_area_sqm = round(free_pixels * (meters_per_pixel ** 2), 2)

        # Save temporary free mask for region detection
        temp_mask_path = os.path.join(output_dir, f"{file_prefix}_free_mask.png")
        cv2.imwrite(temp_mask_path, (free_floor.astype(np.uint8) * 255))

        regions = find_free_regions(temp_mask_path)
        if os.path.exists(temp_mask_path):
            os.remove(temp_mask_path)

        # 3. Determine room type & Generate suggestions + Staging prompt
        resolved_room_type = self.determine_room_type(room_type, detections)
        staging_prompt = build_staging_prompt(
            room_type=resolved_room_type,
            style=style,
            user_description=description,
            detected_objects=detections,
            free_space_pct=free_space_pct
        )

        recommendations = self.generate_recommendations(
            resolved_room_type, detections, free_space_pct
        )

        # 4. Catalog lookup
        catalog_items = {}
        for cat, items in FURNITURE.items():
            for item in items:
                catalog_items[item["name"]] = item

        # 5. Place and Render items
        decorated_bgr = original_bgr.copy()
        annotated_bgr = original_bgr.copy()

        placed_items = []
        placed_boxes = []

        # Current free mask tracked for placement
        current_free_mask = free_floor.copy().astype(np.uint8)

        # Clamp floor mask so items aren't cut off at the bottom edge
        bottom_margin = int(height * 0.06)
        current_free_mask[max(0, height - bottom_margin):, :] = 0

        last_desk_pos = None
        last_seating_pos = None

        for rec in recommendations:
            item_name = rec["item_name"]
            if item_name not in catalog_items:
                rec["placed"] = False
                continue

            item_info = catalog_items[item_name]
            is_wall_item = item_info.get("is_wall_item", False)
            item_file = item_info["file"]

            # Load asset to preserve natural aspect ratio
            resolved_file = item_file
            if not os.path.isabs(resolved_file):
                candidate = os.path.join(AI_SERVICE_DIR, resolved_file)
                if os.path.exists(candidate):
                    resolved_file = candidate

            png_header = cv2.imread(resolved_file, cv2.IMREAD_UNCHANGED)
            base_w_px = meters_to_pixels(item_info["width"], meters_per_pixel)
            if is_wall_item or item_info.get("category") in ["fitness_mirror", "decor"]:
                if png_header is not None:
                    png_h, png_w = png_header.shape[:2]
                    base_h_px = max(25, int(base_w_px * (png_h / float(png_w))))
                else:
                    base_h_px = meters_to_pixels(item_info.get("depth", 0.8), meters_per_pixel)
            else:
                base_h_px = max(25, meters_to_pixels(item_info["depth"], meters_per_pixel))

            placed_pos = None

            if is_wall_item:
                # Wall item scaling
                wall_w = max(50, int(base_w_px * 0.85))
                wall_h = max(40, int(base_h_px * 0.85 if base_h_px > 40 else base_w_px * 0.70))
                placed_pos = self.find_wall_placement(
                    width, height, wall_w, wall_h, placed_boxes,
                    wall_mask=wall_mask, ceiling_mask=ceiling_mask, window_mask=window_mask
                )
                if placed_pos:
                    placed_pos["rotation"] = 0

            elif item_name in ["study_chair", "office_chair"] and last_desk_pos is not None:
                # Place chair realistically in front of or next to the desk
                cw = max(40, int(base_w_px * 0.85))
                ch = max(40, int(base_h_px * 0.85))
                cand_chair_x = last_desk_pos["x"] + last_desk_pos["width"] // 2 - cw // 2
                cand_chair_y = last_desk_pos["y"] + last_desk_pos["height"] - 15

                if cand_chair_y + ch <= height - 15 and cand_chair_x >= 10 and cand_chair_x + cw <= width - 10:
                    placed_pos = {
                        "x": cand_chair_x,
                        "y": cand_chair_y,
                        "width": cw,
                        "height": ch,
                        "rotation": 0
                    }

            elif item_name == "coffee_table" and last_seating_pos is not None:
                # Place coffee table realistically in front of the sofa
                tw = max(50, int(last_seating_pos["width"] * 0.65))
                th = max(35, int(base_h_px * 0.75))
                cand_tx = last_seating_pos["x"] + (last_seating_pos["width"] - tw) // 2
                cand_ty = min(height - th - 15, last_seating_pos["y"] + int(last_seating_pos["height"] * 0.70))
                if cand_ty > last_seating_pos["y"] and cand_tx >= 15 and cand_tx + tw <= width - 15:
                    placed_pos = {
                        "x": cand_tx,
                        "y": cand_ty,
                        "width": tw,
                        "height": th,
                        "rotation": 0
                    }

            elif item_name in ["gaming_tv_unit"]:
                # Place TV console along the left or side wall
                for tv_rx, tv_ry in [(0.06, 0.60), (0.10, 0.55), (0.05, 0.65), (0.50, 0.52)]:
                    tw = max(90, int(base_w_px * 0.75))
                    th = max(45, int(base_h_px * 0.75))
                    cand_tx = int(width * tv_rx)
                    cand_ty = int(height * tv_ry - th // 2)
                    if 10 <= cand_tx <= width - tw - 10 and cand_ty + th <= height - 15 and cand_ty >= 0:
                        box = (cand_tx, cand_ty, cand_tx + tw, cand_ty + th)
                        if not any(not (box[2] <= p[0] - 5 or box[0] >= p[2] + 5 or box[3] <= p[1] - 5 or box[1] >= p[3] + 5) for p in placed_boxes):
                            if window_mask is not None and np.sum(window_mask[cand_ty:cand_ty+th, cand_tx:cand_tx+tw] > 0) > 0:
                                continue
                            placed_pos = {"x": cand_tx, "y": cand_ty, "width": tw, "height": th, "rotation": 0}
                            break

            elif item_name in ["indoor_plant"]:
                # Place plant in empty corner or side space
                for prx, pry in [(0.26, 0.46), (0.20, 0.50), (0.75, 0.52), (0.15, 0.55)]:
                    pw_c = max(35, int(base_w_px * 0.75))
                    ph_c = max(50, int(base_h_px * 0.75))
                    cand_px = int(width * prx - pw_c // 2)
                    cand_py = int(height * pry - ph_c // 2)
                    if 15 <= cand_px <= width - pw_c - 15 and cand_py + ph_c <= height - 15 and cand_py >= 0:
                        box = (cand_px, cand_py, cand_px + pw_c, cand_py + ph_c)
                        if not any(not (box[2] <= p[0] - 5 or box[0] >= p[2] + 5 or box[3] <= p[1] - 5 or box[1] >= p[3] + 5) for p in placed_boxes):
                            if window_mask is not None and np.sum(window_mask[cand_py:cand_py+ph_c, cand_px:cand_px+pw_c] > 0) > 0:
                                continue
                            placed_pos = {"x": cand_px, "y": cand_py, "width": pw_c, "height": ph_c, "rotation": 0}
                            break

            if placed_pos is None and not is_wall_item:
                # Find placement in available regions
                for r_idx, region in enumerate(regions):
                    region_y = region["bbox"]["y"] + region["bbox"]["height"] / 2.0
                    y_ratio = region_y / float(height)
                    persp_scale = float(np.clip(0.65 + (y_ratio - 0.45) * 0.9, 0.60, 1.15))

                    scaled_w = max(40, int(base_w_px * persp_scale))
                    scaled_h = max(35, int(base_h_px * persp_scale))

                    # Perspective foreshortening: floor plane items have foreshortened vertical depth
                    if item_name in ["double_bed", "single_bed", "king_bed", "bed"]:
                        scaled_h = int(scaled_w * 0.55)
                        scaled_w = min(scaled_w, int(width * 0.42))
                        scaled_h = min(scaled_h, int(region["bbox"]["height"] * 0.65))
                    elif item_name in ["wardrobe", "study_desk", "desk"]:
                        scaled_h = int(scaled_w * 0.55)
                        scaled_w = min(scaled_w, int(width * 0.35))
                        scaled_h = min(scaled_h, int(region["bbox"]["height"] * 0.55))

                    can_rotate = item_info.get("can_rotate", item_info.get("category") in ["seating", "table"])
                    loc = find_best_position(
                        current_free_mask,
                        region,
                        scaled_w,
                        scaled_h,
                        step=12,
                        allow_rotation=can_rotate
                    )
                    if loc:
                        # Ensure boundary clearance
                        if (
                            loc["y"] + loc["height"] <= height - 15
                            and loc["x"] + loc["width"] <= width - 20
                            and loc["x"] >= 15
                        ):
                            placed_pos = loc
                            break

            if placed_pos:
                px = placed_pos["x"]
                py = placed_pos["y"]
                pw = placed_pos["width"]
                ph = placed_pos["height"]
                rot = placed_pos.get("rotation", 0)

                # Keep track of key anchors
                if "desk" in item_name:
                    last_desk_pos = placed_pos
                if item_info.get("category") == "seating":
                    last_seating_pos = placed_pos

                # Render on decorated image
                try:
                    render_furniture(
                        decorated_bgr,
                        item_file,
                        px,
                        py,
                        pw,
                        ph,
                        rotation=rot,
                        add_shadow=not is_wall_item
                    )

                    # Update mask & collision box with spacing margin
                    pad = 8
                    x1 = max(0, px - pad)
                    y1 = max(0, py - pad)
                    x2 = min(width, px + pw + pad)
                    y2 = min(height, py + ph + pad)
                    current_free_mask[y1:y2, x1:x2] = 0
                    placed_boxes.append((px, py, px + pw, py + ph))

                    rec["placed"] = True
                    rec["placement"] = {
                        "x": int(px),
                        "y": int(py),
                        "width": int(pw),
                        "height": int(ph),
                        "rotation": int(rot),
                    }
                    placed_items.append({
                        "item_name": item_name,
                        "title": rec["title"],
                        "category": rec["category"],
                        "position": rec["placement"]
                    })
                except Exception as e:
                    print(f"Failed to render {item_name}: {e}")
                    rec["placed"] = False
            else:
                rec["placed"] = False

        # 6. Create Annotated Image with clean UI badges
        annotated_bgr = decorated_bgr.copy()
        badge_palette = {
            "work_study": (245, 130, 32),    # Amber
            "entertainment": (210, 45, 120), # Magenta/Violet
            "seating": (40, 160, 220),       # Sky blue
            "wall_decor": (120, 190, 32),    # Light green
            "fitness": (35, 60, 230),        # Sport red
            "fitness_mirror": (35, 120, 230),# Orange-red
            "fitness_wall": (20, 180, 240),  # Yellow-orange
            "storage": (180, 100, 40),       # Teal
            "decor": (60, 180, 75),          # Green
            "table": (150, 150, 150),        # Gray
        }

        for item in placed_items:
            pos = item["position"]
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            cat = item["category"]
            color = badge_palette.get(cat, (0, 220, 120))

            # Bounding box
            cv2.rectangle(
                annotated_bgr,
                (x, y),
                (x + w, y + h),
                color,
                2,
                cv2.LINE_AA
            )

            # Badge pill
            label = f"+ {item['title']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            badge_y1 = max(0, y - th - 12)
            badge_y2 = badge_y1 + th + 10
            badge_x1 = max(0, x)
            badge_x2 = min(width, x + tw + 16)

            # Semi-transparent pill
            overlay = annotated_bgr.copy()
            cv2.rectangle(
                overlay,
                (badge_x1, badge_y1),
                (badge_x2, badge_y2),
                (20, 20, 20),
                -1
            )
            cv2.addWeighted(overlay, 0.75, annotated_bgr, 0.25, 0, annotated_bgr)

            # Left color bar
            cv2.rectangle(
                annotated_bgr,
                (badge_x1, badge_y1),
                (badge_x1 + 4, badge_y2),
                color,
                -1
            )

            # Text
            cv2.putText(
                annotated_bgr,
                label,
                (badge_x1 + 10, badge_y2 - 6),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        # 7. Check AI Staging Engine (Cloud / Local Diffusion)
        staging_engine = AIStagingEngine()
        ai_staged_path = os.path.join(output_dir, f"{file_prefix}_ai_staged.jpg")
        stage_res = staging_engine.stage_room(
            original_image_path=image_path,
            depth_map_path=None,
            prompt_dict=staging_prompt,
            output_path=ai_staged_path,
            floor_mask=free_floor,
            window_mask=window_mask,
            occupied_mask=occupied_mask,
        )
        if stage_res.get("success") and os.path.exists(ai_staged_path):
            staged_cv = cv2.imread(ai_staged_path)
            if staged_cv is not None:
                decorated_bgr = staged_cv

        # 8. Save Images
        decorated_path = os.path.join(output_dir, f"{file_prefix}_decorated.jpg")
        annotated_path = os.path.join(output_dir, f"{file_prefix}_annotated.jpg")
        cv2.imwrite(decorated_path, decorated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(annotated_path, annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 9. Depth, 3D Mesh/Point Cloud & Multi-Angle View Generation
        depth_vis_path = os.path.join(output_dir, f"{file_prefix}_depth.jpg")
        point_cloud_path = os.path.join(output_dir, f"{file_prefix}_point_cloud.ply")
        mesh_3d_path = os.path.join(output_dir, f"{file_prefix}_mesh.obj")
        multi_angle_views = []

        try:
            # Estimate depth on the STAGED room image so 3D model and multi-angle views
            # capture the real 3D geometry of placed furniture, beds, desks, and decor!
            depth_map = estimate_depth(decorated_path)
            create_depth_visualization(depth_map, depth_vis_path)

            if generate_3d:
                dec_rgb = cv2.cvtColor(decorated_bgr, cv2.COLOR_BGR2RGB)
                pts, cols, norms, faces = create_point_cloud_and_mesh(
                    dec_rgb,
                    depth_map,
                    target_width=256,
                    max_edge_jump_ratio=0.12
                )
                save_point_cloud(pts, cols, point_cloud_path, normals=norms, faces=faces)
                save_mesh_obj(pts, cols, mesh_3d_path, normals=norms, faces=faces)

            # Synthesize novel camera viewpoints for interactive 3D rotation
            multi_angle_views = synthesize_novel_views(
                image_bgr=decorated_bgr,
                depth_norm=depth_map,
                output_dir=output_dir,
                file_prefix=file_prefix
            )
        except Exception as e:
            print(f"Depth / 3D point cloud / multi-angle generation notice: {e}")
            if not os.path.exists(depth_vis_path):
                depth_vis_path = None
            if not os.path.exists(point_cloud_path):
                point_cloud_path = None
            if not os.path.exists(mesh_3d_path):
                mesh_3d_path = None

        return {
            "success": True,
            "room_type": resolved_room_type,
            "applied_style": style,
            "user_description": description or "",
            "staging_prompt": staging_prompt,
            "dimensions_estimate": {
                "width_m": room_width_m,
                "meters_per_pixel": round(meters_per_pixel, 5),
                "image_width": width,
                "image_height": height,
            },
            "detected_existing_objects": [
                {
                    "name": d["name"],
                    "confidence": d["confidence"],
                    "bbox": d["bbox"],
                }
                for d in detections
            ],
            "free_space_summary": {
                "total_floor_area_sqm": est_floor_area_sqm,
                "free_floor_area_sqm": est_free_area_sqm,
                "free_space_percentage": free_space_pct,
                "candidate_regions_count": len(regions),
            },
            "placed_items": placed_items,
            "suggestions": recommendations,
            "multi_angle_views": multi_angle_views,
            "files": {
                "decorated_image": decorated_path,
                "annotated_image": annotated_path,
                "depth_map": depth_vis_path,
                "point_cloud": point_cloud_path,
                "mesh_3d": mesh_3d_path,
                "multi_angle_views": multi_angle_views,
            }
        }
