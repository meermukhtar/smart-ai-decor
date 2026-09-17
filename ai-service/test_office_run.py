import os
from room_analysis.pipeline import RoomDecorationPipeline

def main():
    pipeline = RoomDecorationPipeline()
    image_path = os.path.join(os.path.dirname(__file__), "rooms.jpeg")
    output_dir = os.path.join(os.path.dirname(__file__), "output")

    print(f"Running pipeline on {image_path}...")
    result = pipeline.process(
        image_path=image_path,
        room_type="office",
        room_width_m=6.0,
        output_dir=output_dir,
        file_prefix="test_rooms_office",
        generate_3d=True
    )

    print("\n--- Pipeline Result for Empty Office ---")
    print(f"Room Type: {result['room_type']}")
    print(f"Detected Existing Objects: {[o['name'] for o in result['detected_existing_objects']]}")
    print(f"Free Space Percentage: {result['free_space_summary']['free_space_percentage']}%")
    print(f"Candidate Regions: {result['free_space_summary']['candidate_regions_count']}")
    print(f"\nPlaced Items ({len(result['placed_items'])}):")
    for item in result['placed_items']:
        print(f"  - {item['title']} at {item['position']}")

    print("\nGenerated Files:")
    for k, v in result['files'].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
