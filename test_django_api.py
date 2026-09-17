import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from rest_framework.test import APIClient
from django.core.files.uploadedfile import SimpleUploadedFile
from rooms.models import Room

def test_auto_decorate_api():
    client = APIClient()

    # Test 1: Empty Living Room with custom description & style
    empty_room_path = os.path.join(os.path.dirname(__file__), "ai-service", "sample_empty_room.jpg")
    if os.path.exists(empty_room_path):
        with open(empty_room_path, "rb") as f:
            img_bytes = f.read()
        uploaded = SimpleUploadedFile("sample_empty.jpg", img_bytes, content_type="image/jpeg")

        print("\n=======================================================")
        print("Test 1: Auto Decorate Empty Room with User Description")
        print("=======================================================")
        response = client.post(
            "/api/rooms/auto-decorate/",
            {
                "name": "Luxury Staged Living Room",
                "image": uploaded,
                "room_type": "living_room",
                "description": "minimalist scandinavian living room with modern sofa, coffee table and greenery",
                "style": "scandinavian",
                "room_width_m": 4.5
            },
            format="multipart"
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
        data = response.json()
        assert data["success"] is True
        assert len(data["placed_items"]) >= 3
        print(f"Room ID: {data['room_id']}")
        print(f"Room Type: {data['room_type']}")
        print(f"Applied Style: {data.get('applied_style')}")
        print(f"User Description: {data.get('user_description')}")
        print(f"Decorated Image URL: {data['decorated_image_url']}")
        print(f"Annotated Image URL: {data['annotated_image_url']}")
        print(f"Depth Map URL: {data['depth_map_url']}")
        print(f"Point Cloud (3D) URL: {data['point_cloud_url']}")
        print(f"Placed Items ({len(data['placed_items'])}):")
        for item in data['placed_items']:
            print(f"  - {item['title']} ({item['category']}) at {item['position']}")

    # Test 2: Existing Furnished Room (Bedroom)
    image_path = os.path.join(os.path.dirname(__file__), "ai-service", "room12.jpg")
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    uploaded2 = SimpleUploadedFile("test_room.jpg", img_bytes, content_type="image/jpeg")

    print("\n=======================================================")
    print("Test 2: Auto Decorate Existing Furnished Bedroom")
    print("=======================================================")
    response2 = client.post(
        "/api/rooms/auto-decorate/",
        {
            "name": "Master Bedroom Test",
            "image": uploaded2,
            "room_type": "auto",
            "room_width_m": 4.5
        },
        format="multipart"
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["success"] is True
    print(f"Room ID: {data2['room_id']}")
    print(f"Room Type: {data2['room_type']}")
    print(f"Placed Items ({len(data2['placed_items'])}):")
    for item in data2['placed_items']:
        print(f"  - {item['title']} ({item['category']}) at {item['position']}")

if __name__ == "__main__":
    test_auto_decorate_api()
