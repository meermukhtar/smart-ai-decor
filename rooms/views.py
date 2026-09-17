import os
import sys
from pathlib import Path

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status

from .models import Room
from .serializers import RoomSerializer

# Add ai-service to sys.path
AI_SERVICE_PATH = str(settings.BASE_DIR / "ai-service")
if AI_SERVICE_PATH not in sys.path:
    sys.path.insert(0, AI_SERVICE_PATH)

from room_analysis.pipeline import RoomDecorationPipeline

# Cache pipeline instance
_pipeline_instance = None


def get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RoomDecorationPipeline()
    return _pipeline_instance


class RoomUploadView(APIView):
    """Basic image upload view."""

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = RoomSerializer(data=request.data)
        if serializer.is_valid():
            room = serializer.save()
            return Response(
                RoomSerializer(room).data,
                status=status.HTTP_201_CREATED
            )
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )


class AutoDecorateRoomView(APIView):
    """
    Single unified API endpoint:
    Upload room image -> detects existing objects & free floor
    -> automatically recommends additions (study desk, tv/couch, gym, office, nook)
    -> generates decorated 2D image + depth map + 3D point cloud (.ply)
    -> returns URLs and structured suggestions.
    """

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        image_file = request.FILES.get("image")
        if not image_file:
            return Response(
                {
                    "success": False,
                    "error": "No image file provided. Please upload an image with key 'image'."
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        name = request.data.get("name", "Room Decoration")
        room_type = request.data.get("room_type", "auto")
        description = request.data.get("description")
        style = request.data.get("style", "modern_luxury")

        try:
            room_width_m = float(request.data.get("room_width_m", 4.5))
        except (TypeError, ValueError):
            room_width_m = 4.5

        # 1. Save Room record
        room = Room.objects.create(
            name=name,
            image=image_file,
            room_type=room_type if room_type != "auto" else None
        )

        image_path = room.image.path

        # Output directories inside media
        output_dir = os.path.join(settings.MEDIA_ROOT, "decorated")
        os.makedirs(output_dir, exist_ok=True)
        file_prefix = f"room_{room.id}"

        # 2. Run Pipeline
        pipeline = get_pipeline()
        try:
            result = pipeline.process(
                image_path=image_path,
                room_type=room_type,
                room_width_m=room_width_m,
                output_dir=output_dir,
                file_prefix=file_prefix,
                generate_3d=True,
                description=description,
                style=style
            )
        except Exception as e:
            return Response(
                {
                    "success": False,
                    "error": f"AI room decoration failed: {str(e)}"
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 3. Associate generated files
        files = result.get("files", {})

        dec_file = files.get("decorated_image")
        if dec_file and os.path.exists(dec_file):
            room.decorated_image.name = f"decorated/{os.path.basename(dec_file)}"

        ann_file = files.get("annotated_image")
        if ann_file and os.path.exists(ann_file):
            room.annotated_image.name = f"decorated/{os.path.basename(ann_file)}"

        dep_file = files.get("depth_map")
        if dep_file and os.path.exists(dep_file):
            room.depth_map.name = f"decorated/{os.path.basename(dep_file)}"

        pcd_file = files.get("point_cloud")
        if pcd_file and os.path.exists(pcd_file):
            room.point_cloud.name = f"decorated/{os.path.basename(pcd_file)}"

        # 3D Solid Surface Mesh (.obj)
        mesh_file = files.get("mesh_3d")
        mesh_3d_url = None
        if mesh_file and os.path.exists(mesh_file):
            mesh_3d_url = request.build_absolute_uri(settings.MEDIA_URL + f"decorated/{os.path.basename(mesh_file)}")

        # Multi-Angle View Array
        raw_views = result.get("multi_angle_views", [])
        multi_angle_views_response = []
        for v in raw_views:
            img_url = request.build_absolute_uri(settings.MEDIA_URL + f"decorated/{v['file_name']}")
            multi_angle_views_response.append({
                "angle": v.get("angle"),
                "label": v.get("label"),
                "yaw_deg": v.get("yaw_deg"),
                "pitch_deg": v.get("pitch_deg"),
                "fov_scale": v.get("fov_scale", 1.0),
                "image_url": img_url,
            })

        room.room_type = result.get("room_type", room_type)
        room.analysis = {
            "dimensions": result.get("dimensions_estimate"),
            "detected_existing_objects": result.get("detected_existing_objects"),
            "free_space_summary": result.get("free_space_summary"),
            "placed_items": result.get("placed_items"),
            "staging_prompt": result.get("staging_prompt"),
            "applied_style": result.get("applied_style"),
            "user_description": result.get("user_description"),
            "mesh_3d_url": mesh_3d_url,
            "multi_angle_views": multi_angle_views_response,
        }
        room.suggestions = result.get("suggestions")
        room.save()

        # 4. Helper for absolute URLs
        def make_url(field):
            if field and hasattr(field, "url") and field.name:
                return request.build_absolute_uri(field.url)
            return None

        return Response(
            {
                "success": True,
                "room_id": room.id,
                "room_name": room.name,
                "room_type": room.room_type,
                "applied_style": result.get("applied_style", style),
                "user_description": result.get("user_description", description or ""),
                "staging_prompt": result.get("staging_prompt", {}),
                "original_image_url": make_url(room.image),
                "decorated_image_url": make_url(room.decorated_image),
                "annotated_image_url": make_url(room.annotated_image),
                "depth_map_url": make_url(room.depth_map),
                "point_cloud_url": make_url(room.point_cloud),
                "mesh_3d_url": mesh_3d_url,
                "multi_angle_views": multi_angle_views_response,
                "detected_existing_objects": result.get("detected_existing_objects", []),
                "free_space_summary": result.get("free_space_summary", {}),
                "placed_items": result.get("placed_items", []),
                "suggestions": result.get("suggestions", []),
                "message": "Room analyzed, AI staging prompt generated, and decorated design rendered successfully!"
            },
            status=status.HTTP_200_OK
        )