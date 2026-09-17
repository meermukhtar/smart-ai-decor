import os
import base64
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from room_analysis.pipeline import RoomDecorationPipeline

app = FastAPI(
    title="Smart Space AI - Automated Room Decoration & Suggestion API",
    description="End-to-end room analysis, intelligent suggestions, and 2D decorated room generation.",
    version="2.0.0"
)

# Output directory for standalone FastAPI runs
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RoomDecorationPipeline()
    return _pipeline


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Smart Space AI Decorator",
        "version": "2.0.0"
    }


@app.post("/analyze-room/")
async def analyze_room(
    image: UploadFile = File(...),
    room_type: str = Form("auto"),
    room_width_m: float = Form(4.5),
):
    return await decorate_room(image=image, room_type=room_type, room_width_m=room_width_m)


@app.post("/decorate-room/")
@app.post("/api/decorate-room/")
async def decorate_room(
    image: UploadFile = File(...),
    room_type: str = Form("auto"),
    room_width_m: float = Form(4.5),
):
    try:
        suffix = os.path.splitext(image.filename)[1] or ".jpg"
        temp_id = str(uuid.uuid4())[:8]

        temp_img_path = os.path.join(OUTPUT_DIR, f"upload_{temp_id}{suffix}")
        with open(temp_img_path, "wb") as f:
            content = await image.read()
            f.write(content)

        pipeline = get_pipeline()
        result = pipeline.process(
            image_path=temp_img_path,
            room_type=room_type,
            room_width_m=room_width_m,
            output_dir=OUTPUT_DIR,
            file_prefix=f"api_{temp_id}",
            generate_3d=True
        )

        files = result.get("files", {})
        dec_path = files.get("decorated_image")
        dec_b64 = None
        if dec_path and os.path.exists(dec_path):
            with open(dec_path, "rb") as df:
                dec_b64 = base64.b64encode(df.read()).decode("utf-8")

        return {
            "success": True,
            "room_type": result.get("room_type"),
            "dimensions_estimate": result.get("dimensions_estimate"),
            "detected_existing_objects": result.get("detected_existing_objects"),
            "free_space_summary": result.get("free_space_summary"),
            "placed_items": result.get("placed_items"),
            "suggestions": result.get("suggestions"),
            "output_files": {
                "decorated_image_url": f"/output/{os.path.basename(files['decorated_image'])}" if files.get('decorated_image') else None,
                "annotated_image_url": f"/output/{os.path.basename(files['annotated_image'])}" if files.get('annotated_image') else None,
                "depth_map_url": f"/output/{os.path.basename(files['depth_map'])}" if files.get('depth_map') else None,
                "point_cloud_url": f"/output/{os.path.basename(files['point_cloud'])}" if files.get('point_cloud') else None,
            },
            "decorated_image_base64": dec_b64,
            "message": "Room analyzed and decorated successfully!"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )