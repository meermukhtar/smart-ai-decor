from rembg import remove
from PIL import Image
from pathlib import Path


INPUT_DIR = Path("models/furniture")
OUTPUT_DIR = Path("models/furniture_transparent")


def process_image(input_path, output_path):
    print(f"Processing: {input_path}")

    image = Image.open(input_path).convert("RGBA")

    result = remove(image)

    result.save(output_path)

    print(f"Saved: {output_path}")


def process_all_furniture():

    for image_path in INPUT_DIR.rglob("*.png"):

        relative_path = image_path.relative_to(INPUT_DIR)

        output_path = (
            OUTPUT_DIR / relative_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        process_image(
            image_path,
            output_path
        )


if __name__ == "__main__":
    process_all_furniture()