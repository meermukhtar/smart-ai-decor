import os
import json
import time
from dotenv import load_dotenv
from google import genai


from room_analysis.detection import detect_furniture
from room_analysis.design_actions import match_all_actions
# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from your .env file."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# FURNITURE CATALOG
# ============================================================

FURNITURE_CATALOG = {
    "living_room": [
        "2_seater_sofa",
        "3_seater_sofa",
        "coffee_table",
        "armchair",
        "side_table"
    ],

    "bedroom": [
        "single_bed",
        "double_bed",
        "wardrobe",
        "desk"
    ],

    "office": [
        "office_desk",
        "office_chair",
        "bookshelf"
    ],

    "dining_room": [
        "dining_table",
        "dining_chair"
    ]
}

catalog_text = json.dumps(
    FURNITURE_CATALOG,
    indent=2
)


# ============================================================
# UPLOAD ROOM IMAGE
# ============================================================

print("Uploading room image...")

image = client.files.upload(
    file="room12.jpg"
)

print("Image uploaded successfully.")


# ============================================================
# GEMINI PROMPT
# ============================================================

prompt = f"""
You are an expert interior designer and room layout planner.

Analyze the provided room image carefully.

Your goal is to improve the existing room design while preserving
good existing furniture and maintaining practical walking and
circulation space.

============================================================
VISUAL ANALYSIS RULES
============================================================

Only describe objects and architectural features that are clearly
visible in the image.

DO NOT assume or invent:

- fireplaces
- terraces
- ocean views
- windows
- doors
- furniture
- room dimensions
- architectural features
- decorations
- objects that cannot clearly be seen

If something is uncertain, do not claim that it exists.

============================================================
AVAILABLE FURNITURE CATALOG
============================================================

Our computer vision and rendering system can ONLY work with these
furniture items:

{catalog_text}

IMPORTANT:

When recommending furniture to ADD or REPLACE, you MUST use one
of the exact catalog names above.

DO NOT invent furniture that is not in the catalog.

For example, do NOT recommend:

- rug
- nightstand
- bench
- TV
- cabinet
- plant
- lamp
- artwork

unless it is represented by one of the catalog items above.

============================================================
EXISTING FURNITURE
============================================================

Identify furniture that is actually visible in the image.

For every visible furniture item, decide:

- KEEP
- MOVE
- REPLACE
- REMOVE

Prefer KEEP when existing furniture already works well.

Do not recommend unnecessary changes.

============================================================
MOVE
============================================================

For MOVE actions, describe the desired movement naturally.

Examples:

"Move the sofa closer to the wall."

"Move the coffee table toward the center of the seating area."

"Move the bed slightly away from the wall."

DO NOT provide:

- pixel coordinates
- bounding boxes
- exact coordinates
- exact measurements
- percentages

Our computer vision system will determine the actual physical
position.

============================================================
REPLACE
============================================================

For REPLACE actions:

1. Identify the existing furniture.
2. Specify the replacement using an exact catalog name.
3. Explain why replacement improves the room.

============================================================
ADD
============================================================

For ADD actions:

- Use ONLY furniture from the catalog.
- Do not overcrowd the room.
- Maintain walking space.
- Only add furniture when it provides a meaningful improvement.

============================================================
SYSTEM ARCHITECTURE
============================================================

Gemini is responsible for deciding:

WHAT should change.

Our computer vision system is responsible for deciding:

WHERE the furniture can physically be placed.

Therefore Gemini must NOT calculate:

- pixel positions
- exact coordinates
- exact room measurements
- bounding boxes

For MOVE actions, Gemini should only describe the desired movement
in natural language.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "room_type": "...",

    "existing_furniture": [
        {{
            "name": "...",
            "catalog_name": "...",
            "action": "KEEP",
            "reason": "..."
        }}
    ],

    "actions": [
        {{
            "action": "MOVE",
            "object": "...",
            "catalog_name": "...",
            "instruction": "...",
            "reason": "..."
        }},
        {{
            "action": "REPLACE",
            "object": "...",
            "catalog_name": "...",
            "replacement": "...",
            "replacement_catalog_name": "...",
            "instruction": "...",
            "reason": "..."
        }},
        {{
            "action": "REMOVE",
            "object": "...",
            "catalog_name": "...",
            "reason": "..."
        }},
        {{
            "action": "ADD",
            "object": "...",
            "catalog_name": "...",
            "reason": "..."
        }}
    ],

    "style": "...",

    "colors": [
        "...",
        "..."
    ],

    "materials": [
        "...",
        "..."
    ],

    "overall_recommendation": "..."
}}

============================================================
FINAL RULES
============================================================

1. Only describe furniture actually visible in the image.

2. Do not invent architectural features.

3. Use exact catalog names whenever the computer vision system
   needs to process furniture.

4. Never recommend furniture outside the catalog.

5. Preserve useful existing furniture.

6. Do not overcrowd the room.

7. Maintain practical circulation space.

8. Only recommend meaningful changes.

9. Do not provide coordinates.

10. Do not provide exact measurements.

11. Return ONLY valid JSON.
"""


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):
    """
    Extract JSON from Gemini response.

    Handles cases where Gemini returns:
    ```json
    {...}
    ```

    instead of plain JSON.
    """

    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Try direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start:end + 1]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Gemini response does not contain valid JSON."
    )


# ============================================================
# VALIDATE DESIGN PLAN
# ============================================================

def validate_design_plan(data):
    """
    Basic validation of Gemini's design plan.
    """

    required_fields = [
        "room_type",
        "existing_furniture",
        "actions",
        "style",
        "colors",
        "materials",
        "overall_recommendation"
    ]

    for field in required_fields:
        if field not in data:
            raise ValueError(
                f"Missing required field: {field}"
            )

    if not isinstance(data["existing_furniture"], list):
        raise ValueError(
            "existing_furniture must be a list."
        )

    if not isinstance(data["actions"], list):
        raise ValueError(
            "actions must be a list."
        )

    if not isinstance(data["colors"], list):
        raise ValueError(
            "colors must be a list."
        )

    if not isinstance(data["materials"], list):
        raise ValueError(
            "materials must be a list."
        )

    allowed_actions = {
        "KEEP",
        "MOVE",
        "REPLACE",
        "REMOVE",
        "ADD"
    }

    # Validate existing furniture actions
    for item in data["existing_furniture"]:

        action = item.get("action")

        if action not in allowed_actions:
            raise ValueError(
                f"Invalid existing furniture action: {action}"
            )

    # Validate actions
    for item in data["actions"]:

        action = item.get("action")

        if action not in allowed_actions:
            raise ValueError(
                f"Invalid action: {action}"
            )

    return True


# ============================================================
# CHECK CATALOG
# ============================================================

def get_all_catalog_items():

    items = set()

    for room_items in FURNITURE_CATALOG.values():
        items.update(room_items)

    return items


ALL_CATALOG_ITEMS = get_all_catalog_items()


def check_catalog_usage(data):
    """
    Check whether Gemini recommended furniture that is outside
    our available catalog.
    """

    warnings = []

    # Check existing furniture
    for item in data["existing_furniture"]:

        catalog_name = item.get("catalog_name")

        if catalog_name:
            if catalog_name not in ALL_CATALOG_ITEMS:
                warnings.append(
                    f"Unknown catalog item in existing_furniture: "
                    f"{catalog_name}"
                )

    # Check actions
    for item in data["actions"]:

        action = item.get("action")

        catalog_name = item.get("catalog_name")

        replacement_catalog_name = item.get(
            "replacement_catalog_name"
        )

        if action in {"ADD", "MOVE", "REMOVE"}:

            if catalog_name:
                if catalog_name not in ALL_CATALOG_ITEMS:
                    warnings.append(
                        f"Unknown catalog item: {catalog_name}"
                    )

        if action == "REPLACE":

            if replacement_catalog_name:
                if replacement_catalog_name not in ALL_CATALOG_ITEMS:
                    warnings.append(
                        f"Unknown replacement catalog item: "
                        f"{replacement_catalog_name}"
                    )

    return warnings


# ============================================================
# GEMINI CALL WITH RETRIES + FALLBACK
# ============================================================

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash"
]


def generate_design():

    last_error = None

    for model in MODELS:

        print("\n" + "-" * 60)
        print(f"Trying model: {model}")
        print("-" * 60)

        # Try each model up to 2 times
        for attempt in range(1, 3):

            try:

                print(
                    f"Attempt {attempt}/2..."
                )

                response = client.models.generate_content(
                    model=model,
                    contents=[
                        image,
                        prompt
                    ]
                )

                print(
                    f"SUCCESS: {model}"
                )

                return response

            except Exception as e:

                last_error = e

                print(
                    f"Failed: {model}"
                )

                print(
                    f"Error: {e}"
                )

                # Retry temporary server errors
                if "503" in str(e):

                    if attempt < 2:

                        print(
                            "Temporary 503 error. "
                            "Waiting 3 seconds before retry..."
                        )

                        time.sleep(3)

                    else:

                        print(
                            "Model still unavailable."
                        )

                else:

                    # Don't retry non-temporary errors
                    break

    raise RuntimeError(
        f"All Gemini models failed.\n"
        f"Last error: {last_error}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("GEMINI ROOM DESIGN ANALYSIS")
    print("=" * 60)

    try:

        # Generate Gemini response
        response = generate_design()

        # Get raw response
        raw_text = response.text

        print("\n" + "=" * 60)
        print("RAW GEMINI RESPONSE")
        print("=" * 60)

        print(raw_text)

        # ====================================================
        # PARSE JSON
        # ====================================================

        print("\n" + "=" * 60)
        print("PARSING JSON")
        print("=" * 60)

        design_plan = extract_json(raw_text)

        print("JSON parsed successfully.")

        # ====================================================
        # VALIDATE JSON
        # ====================================================

        validate_design_plan(design_plan)

        print("JSON validation successful.")
        
        
        
            
        # ====================================================
        # YOLO FURNITURE DETECTION
        # ====================================================

        print("\n" + "=" * 60)
        print("DETECTING EXISTING FURNITURE")
        print("=" * 60)

        detections = detect_furniture(
            "room12.jpg",
            confidence=0.5
        )

        print(
            f"Detected furniture: {len(detections)}"
        )

        for detection in detections:

            print(
                f"- {detection['name']} "
                f"→ {detection['catalog_name']} "
                f"(confidence: "
                f"{detection['confidence']:.2f})"
            )


        # ====================================================
        # MATCH GEMINI ACTIONS WITH YOLO DETECTIONS
        # ====================================================

        print("\n" + "=" * 60)
        print("MATCHING GEMINI ACTIONS WITH DETECTIONS")
        print("=" * 60)

        matched_actions = match_all_actions(
            design_plan["actions"],
            detections
        )

        for item in matched_actions:

            action = item["action"]
            detection = item["detection"]

            print(
                f"\nAction: {action.get('action')}"
            )

            print(
                f"Object: {action.get('object')}"
            )

            print(
                f"Catalog: {action.get('catalog_name')}"
            )

            if detection:

                print("MATCH: YES")

                print(
                    f"Detected as: "
                    f"{detection['name']}"
                )

                print(
                    f"BBox: "
                    f"{detection['bbox']}"
                )

            else:

                print("MATCH: NO")
                print(
                    "WARNING: Furniture was not "
                    "detected by YOLO."
                )


                # ====================================================
                # CHECK CATALOG
                # ====================================================

                catalog_warnings = check_catalog_usage(
                    design_plan
                )

                if catalog_warnings:

                    print("\n" + "=" * 60)
                    print("CATALOG WARNINGS")
                    print("=" * 60)

                    for warning in catalog_warnings:
                        print(f"- {warning}")

                else:

                    print(
                        "All furniture catalog references are valid."
                    )

                # ====================================================
                # FINAL JSON
                # ====================================================

                print("\n" + "=" * 60)
                print("FINAL GEMINI ROOM DESIGN PLAN")
                print("=" * 60)

                print(
                    json.dumps(
                        design_plan,
                        indent=4,
                        ensure_ascii=False
                    )
                )

                # ====================================================
                # SUMMARY
                # ====================================================

                print("\n" + "=" * 60)
                print("DESIGN SUMMARY")
                print("=" * 60)

                print(
                    f"Room type: "
                    f"{design_plan['room_type']}"
                )

                print(
                    f"Style: "
                    f"{design_plan['style']}"
                )

                print(
                    f"Existing furniture: "
                    f"{len(design_plan['existing_furniture'])}"
                )

                print(
                    f"Recommended actions: "
                    f"{len(design_plan['actions'])}"
                )

                print(
                    f"Overall recommendation:\n"
                    f"{design_plan['overall_recommendation']}"
                )

    except Exception as e:

        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)

        print(str(e))

        raise