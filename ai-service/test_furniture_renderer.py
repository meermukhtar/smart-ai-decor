import cv2

from room_analysis.furniture_renderer import (
    render_furniture
)


ROOM = room12.jpg

FURNITURE = (
    "models/furniture/"
    "living_room/double_sofa.png"
)


image = cv2.imread(
    ROOM
)

if image is None:
    raise FileNotFoundError(ROOM)


# Temporary position
x = 200
y = 650

width = 549
height = 309


render_furniture(
    image,
    FURNITURE,
    x,
    y,
    width,
    height,
    rotation=0,
    add_shadow=True
)


cv2.imwrite(
    "furniture_render_test.jpg",
    image
)


print(
    "Saved: furniture_render_test.jpg"
)