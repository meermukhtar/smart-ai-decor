import os
from room_analysis.pipeline import RoomDecorationPipeline

def main():
    pipeline = RoomDecorationPipeline()
    image_path = os.path.join(os.path.dirname(__file__), "room12.jpg")
    output_dir = os.path.join(os.path.dirname(__file__), "output")

    print(f"Running pipeline on {image_path} with gym theme...")
    result = pipeline.process(
        image_path=image_path,
        room_type="gym",
        room_width_m=4.5,
        output_dir=output_dir,
        file_prefix="test_gym",
        generate_3d=True
    )

    print("\n--- Pipeline Result for Gym ---")
    print(f"Room Type: {result['room_type']}")
    print(f"\nPlaced Items ({len(result['placed_items'])}):")
    for item in result['placed_items']:
        print(f"  - {item['title']} at {item['position']}")

    print(f"\nSuggestions ({len(result['suggestions'])}):")
    for s in result['suggestions']:
        placed_mark = "✓ Placed" if s.get('placed') else "○ Tip"
        print(f"  [{placed_mark}] {s['title']} ({s['category']}): {s['reason']}")

    print("\nGenerated Files:")
    for k, v in result['files'].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
