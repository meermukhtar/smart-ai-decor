import os
from PIL import Image, ImageDraw
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models", "furniture")

def ensure_dirs():
    for sub in ["gym", "entertainment", "decor"]:
        os.makedirs(os.path.join(MODELS_DIR, sub), exist_ok=True)

def create_dumbbells_rack():
    """Create a high quality dumbbell rack / gym weights station."""
    w, h = 800, 600
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)

    steel = (35, 39, 42, 255)
    steel_light = (70, 75, 80, 255)
    chrome = (200, 205, 215, 255)
    rubber_black = (25, 25, 28, 255)
    rubber_accent = (220, 50, 40, 255)

    # A-frame legs
    draw.polygon([(150, 550), (180, 550), (250, 150), (220, 150)], fill=steel)
    draw.polygon([(650, 550), (620, 550), (550, 150), (580, 150)], fill=steel)
    # Base feet
    draw.rounded_rectangle([100, 530, 250, 560], radius=8, fill=steel_light)
    draw.rounded_rectangle([550, 530, 700, 560], radius=8, fill=steel_light)

    # Crossbars / shelves (3 tiers)
    for y, thickness in [(200, 24), (330, 28), (460, 32)]:
        draw.rounded_rectangle([160, y, 640, y + thickness], radius=6, fill=steel_light)
        draw.rounded_rectangle([170, y + 4, 630, y + thickness - 4], fill=steel)

        # Dumbbells on this shelf
        num_db = 4
        spacing = (610 - 190) // (num_db - 1)
        for i in range(num_db):
            cx = 200 + i * spacing
            cy = y - 10
            scale = 1.0 + (y - 200) * 0.0015
            rw = int(35 * scale)
            rh = int(60 * scale)
            hw = int(30 * scale)

            # Left weight
            draw.rounded_rectangle([cx - rw - hw, cy - rh//2, cx - hw, cy + rh//2], radius=6, fill=rubber_black)
            draw.rounded_rectangle([cx - rw - hw + 4, cy - rh//2 + 4, cx - hw - 4, cy + rh//2 - 4], radius=4, fill=(45, 45, 50, 255))
            # Handle (chrome)
            draw.rectangle([cx - hw, cy - 6, cx + hw, cy + 6], fill=chrome)
            # Right weight
            draw.rounded_rectangle([cx + hw, cy - rh//2, cx + rw + hw, cy + rh//2], radius=6, fill=rubber_black)
            draw.rounded_rectangle([cx + hw + 4, cy - rh//2 + 4, cx + rw + hw - 4, cy + rh//2 - 4], radius=4, fill=(45, 45, 50, 255))
            # Colored accent ring
            draw.line([(cx - hw - 6, cy - rh//2 + 5), (cx - hw - 6, cy + rh//2 - 5)], fill=rubber_accent, width=3)
            draw.line([(cx + hw + 6, cy - rh//2 + 5), (cx + hw + 6, cy + rh//2 - 5)], fill=rubber_accent, width=3)

    path = os.path.join(MODELS_DIR, "gym", "dumbbells_rack.png")
    im.save(path)
    print("Saved:", path)

def create_pull_up_bar():
    """Create a wall-mounted pull-up bar / workout station."""
    w, h = 800, 500
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)

    steel = (30, 32, 35, 255)
    steel_hi = (75, 80, 85, 255)
    grip = (210, 45, 40, 255)
    metal_screw = (180, 185, 195, 255)

    # Wall mounting brackets
    draw.rounded_rectangle([120, 120, 180, 360], radius=8, fill=steel)
    draw.rounded_rectangle([620, 120, 680, 360], radius=8, fill=steel)
    for bx in [150, 650]:
        for by in [150, 240, 330]:
            draw.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=metal_screw)

    # Cantilever arms extending out
    draw.polygon([(150, 150), (250, 270), (250, 300), (150, 190)], fill=steel_hi)
    draw.polygon([(650, 150), (550, 270), (550, 300), (650, 190)], fill=steel_hi)
    draw.rectangle([150, 260, 270, 290], fill=steel)
    draw.rectangle([530, 260, 650, 290], fill=steel)

    # Main pull-up bar (horizontal)
    draw.rounded_rectangle([70, 265, 730, 295], radius=14, fill=steel)
    draw.line([(80, 275), (720, 275)], fill=steel_hi, width=4)

    # Ergonomic curved wide grip ends
    draw.line([(70, 280), (30, 330)], fill=steel, width=24)
    draw.line([(730, 280), (770, 330)], fill=steel, width=24)

    # Textured foam grips
    for gx1, gx2 in [(90, 210), (590, 710), (340, 460)]:
        draw.rounded_rectangle([gx1, 262, gx2, 298], radius=8, fill=grip)
        for rx in range(gx1 + 10, gx2, 12):
            draw.line([(rx, 264), (rx, 296)], fill=(160, 30, 25, 255), width=2)

    path = os.path.join(MODELS_DIR, "gym", "pull_up_bar.png")
    im.save(path)
    print("Saved:", path)

def create_fitness_mirror():
    """Create a modern framed gym / floor mirror."""
    w, h = 500, 900
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)

    frame_color = (30, 30, 32, 255)
    glass_top = (215, 230, 245, 235)
    glass_bot = (185, 205, 225, 235)

    draw.rounded_rectangle([40, 40, 460, 860], radius=16, fill=frame_color)
    draw.rounded_rectangle([52, 52, 448, 848], radius=10, fill=(60, 65, 70, 255))

    # Glass reflection gradient
    for y in range(60, 840):
        t = (y - 60) / (840 - 60)
        r = int(glass_top[0] * (1 - t) + glass_bot[0] * t)
        g = int(glass_top[1] * (1 - t) + glass_bot[1] * t)
        b = int(glass_top[2] * (1 - t) + glass_bot[2] * t)
        draw.line([(60, y), (440, y)], fill=(r, g, b, 230))

    # Diagonal glossy light streaks (reflection)
    draw.polygon([(140, 60), (220, 60), (90, 840), (60, 840)], fill=(255, 255, 255, 70))
    draw.polygon([(260, 60), (300, 60), (140, 840), (110, 840)], fill=(255, 255, 255, 45))

    path = os.path.join(MODELS_DIR, "gym", "fitness_mirror.png")
    im.save(path)
    print("Saved:", path)

def create_gaming_tv_setup():
    """Create a sleek TV console with 4K screen, soundbar, and gaming console (PlayStation)."""
    w, h = 900, 750
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)

    wood_dark = (45, 35, 30, 255)
    wood_light = (75, 58, 48, 255)
    tv_frame = (20, 20, 22, 255)
    screen = (15, 22, 35, 255)
    ps5_white = (245, 245, 250, 255)
    ps5_blue = (0, 140, 255, 255)

    # Media Console Table (lower half)
    draw.rounded_rectangle([100, 480, 800, 680], radius=10, fill=wood_dark)
    draw.rounded_rectangle([105, 485, 795, 520], radius=6, fill=wood_light)
    # Console Legs
    draw.polygon([(130, 680), (150, 680), (140, 730), (120, 730)], fill=(25, 25, 25, 255))
    draw.polygon([(770, 680), (750, 680), (760, 730), (780, 730)], fill=(25, 25, 25, 255))
    draw.polygon([(440, 680), (460, 680), (460, 730), (440, 730)], fill=(25, 25, 25, 255))

    # Console shelves
    draw.rectangle([130, 540, 330, 650], fill=(28, 22, 18, 255))
    draw.rectangle([350, 540, 550, 650], fill=(28, 22, 18, 255))
    draw.rectangle([570, 540, 770, 650], fill=(28, 22, 18, 255))

    # TV Stand / Mount
    draw.rectangle([430, 420, 470, 485], fill=(30, 30, 35, 255))
    draw.rounded_rectangle([390, 478, 510, 488], radius=4, fill=(40, 40, 45, 255))

    # TV Frame & Screen
    draw.rounded_rectangle([150, 90, 750, 430], radius=8, fill=tv_frame)
    draw.rounded_rectangle([158, 98, 742, 422], radius=4, fill=screen)

    # Ambient screen graphic
    for sy in range(260, 420):
        t = (sy - 260) / 160.0
        sc = (int(10 + 40*t), int(30 + 70*t), int(80 + 120*t), 255)
        draw.line([(160, sy), (740, sy)], fill=sc)
    draw.ellipse([400, 220, 500, 320], fill=(255, 180, 80, 200))

    # Soundbar
    draw.rounded_rectangle([280, 465, 620, 485], radius=6, fill=(25, 25, 28, 255))
    draw.rounded_rectangle([285, 468, 615, 482], radius=4, fill=(45, 45, 50, 255))

    # PS5 console on table
    draw.polygon([(660, 390), (690, 395), (685, 482), (655, 482)], fill=ps5_white)
    draw.polygon([(690, 395), (710, 400), (705, 482), (685, 482)], fill=(20, 20, 25, 255))
    draw.polygon([(710, 400), (730, 405), (725, 482), (705, 482)], fill=ps5_white)
    draw.line([(690, 400), (685, 475)], fill=ps5_blue, width=3)

    # Controller on shelf
    draw.rounded_rectangle([200, 590, 260, 625], radius=8, fill=ps5_white)
    draw.ellipse([215, 600, 225, 610], fill=(40, 40, 40, 255))
    draw.ellipse([235, 600, 245, 610], fill=(40, 40, 40, 255))

    path = os.path.join(MODELS_DIR, "entertainment", "gaming_tv_unit.png")
    im.save(path)
    print("Saved:", path)

def create_wall_art():
    """Create an aesthetic framed wall painting / canvas."""
    w, h = 500, 400
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)

    frame = (28, 28, 30, 255)
    passepartout = (250, 248, 245, 255)

    draw.rounded_rectangle([30, 30, 470, 370], radius=8, fill=frame)
    draw.rectangle([42, 42, 458, 358], fill=passepartout)

    # Artwork inside
    draw.rectangle([70, 70, 430, 330], fill=(235, 225, 215, 255))
    draw.ellipse([220, 95, 290, 165], fill=(220, 130, 80, 255))
    draw.polygon([(70, 330), (170, 180), (280, 330)], fill=(75, 95, 90, 255))
    draw.polygon([(210, 330), (330, 150), (430, 330)], fill=(185, 140, 105, 255))
    draw.polygon([(140, 330), (250, 240), (360, 330)], fill=(120, 145, 140, 255))

    path = os.path.join(MODELS_DIR, "decor", "wall_art.png")
    im.save(path)
    print("Saved:", path)

def create_wall_shelf():
    """Create modern floating wall shelf with books and planter."""
    w, h = 600, 280
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)

    wood = (65, 50, 40, 255)
    wood_top = (95, 75, 60, 255)

    # Shelf board
    draw.polygon([(80, 200), (520, 200), (500, 230), (100, 230)], fill=wood_top)
    draw.rectangle([100, 230, 500, 248], fill=wood)

    # Brackets
    draw.polygon([(150, 248), (170, 248), (150, 275)], fill=(30, 30, 30, 255))
    draw.polygon([(450, 248), (430, 248), (450, 275)], fill=(30, 30, 30, 255))

    # Books
    draw.rectangle([140, 110, 160, 200], fill=(180, 60, 50, 255))
    draw.rectangle([162, 120, 182, 200], fill=(45, 105, 150, 255))
    draw.rectangle([184, 100, 208, 200], fill=(210, 160, 50, 255))
    draw.polygon([(210, 200), (230, 200), (260, 125), (240, 120)], fill=(70, 130, 80, 255))

    # Potted plant
    pot = (210, 190, 175, 255)
    draw.polygon([(400, 160), (440, 160), (432, 200), (408, 200)], fill=pot)
    leaf = (55, 135, 70, 255)
    leaf_hi = (85, 175, 95, 255)
    for ox, oy in [(-15, -15), (0, -28), (15, -15), (-8, -8), (10, -5)]:
        draw.ellipse([410 + ox, 155 + oy, 430 + ox, 175 + oy], fill=leaf)
        draw.ellipse([414 + ox, 158 + oy, 426 + ox, 171 + oy], fill=leaf_hi)

    path = os.path.join(MODELS_DIR, "decor", "wall_shelf.png")
    im.save(path)
    print("Saved:", path)

def create_indoor_plant():
    """Create a luxury potted indoor fiddle leaf fig plant."""
    w, h = 450, 800
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)

    pot_color = (235, 235, 235, 255)
    stand_color = (130, 90, 60, 255)

    draw.line([(150, 550), (130, 770)], fill=stand_color, width=12)
    draw.line([(300, 550), (320, 770)], fill=stand_color, width=12)
    draw.line([(225, 550), (225, 760)], fill=stand_color, width=10)
    draw.line([(140, 620), (310, 620)], fill=stand_color, width=10)

    # Pot
    draw.polygon([(160, 480), (290, 480), (275, 640), (175, 640)], fill=pot_color)
    draw.ellipse([160, 465, 290, 495], fill=(220, 220, 220, 255))
    draw.ellipse([165, 470, 285, 490], fill=(60, 45, 35, 255))

    # Trunk & stems
    stem_color = (75, 60, 45, 255)
    draw.line([(225, 475), (225, 200)], fill=stem_color, width=12)
    draw.line([(225, 380), (140, 260)], fill=stem_color, width=8)
    draw.line([(225, 330), (310, 220)], fill=stem_color, width=8)
    draw.line([(225, 240), (160, 140)], fill=stem_color, width=8)

    # Leaves
    leaves = [
        (130, 230, 70, 110, -25),
        (320, 200, 75, 115, 30),
        (150, 120, 65, 100, -15),
        (290, 130, 65, 95, 20),
        (225, 70, 80, 120, 0),
        (225, 160, 75, 105, 5),
        (180, 310, 60, 90, -40),
        (270, 300, 65, 95, 35),
    ]

    for cx, cy, rx, ry, ang in leaves:
        leaf_im = Image.new("RGBA", (rx * 2, ry * 2), (0, 0, 0, 0))
        ldraw = ImageDraw.Draw(leaf_im)
        ldraw.ellipse([0, 0, rx * 2, ry * 2], fill=(40, 125, 55, 255))
        ldraw.ellipse([6, 6, rx * 2 - 6, ry * 2 - 6], fill=(60, 155, 75, 255))
        ldraw.line([(rx, 4), (rx, ry * 2 - 4)], fill=(85, 195, 100, 255), width=3)
        rotated = leaf_im.rotate(ang, resample=Image.BICUBIC, expand=True)
        rw, rh = rotated.size
        im.paste(rotated, (cx - rw // 2, cy - rh // 2), rotated)

    path = os.path.join(MODELS_DIR, "decor", "indoor_plant.png")
    im.save(path)
    print("Saved:", path)

if __name__ == "__main__":
    ensure_dirs()
    create_dumbbells_rack()
    create_pull_up_bar()
    create_fitness_mirror()
    create_gaming_tv_setup()
    create_wall_art()
    create_wall_shelf()
    create_indoor_plant()
    print("All additional furniture & decor assets created successfully!")
