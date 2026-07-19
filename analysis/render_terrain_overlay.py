#!/usr/bin/env python3
"""Render the conservative RMUC 2026 terrain locator without AI-redrawing the map."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MAP = ROOT / "web" / "assets" / "map.webp"
OUTPUT = ROOT / "analysis" / "outputs" / "terrain_semantic_v2.png"
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

MAP_WIDTH = 2337
MAP_HEIGHT = 1283
PANEL_HEIGHT = 320

# These are point anchors, not guessed semantic masks. Coordinates are pixels in map.webp.
MARKERS = (
    ("1", (1170, 490), "highland"),  # central highland
    ("2", (520, 185), "highland"),  # red trapezoid highland
    ("3", (1817, 1068), "highland"),  # blue trapezoid highland
    ("4", (748, 300), "road"),  # red road
    ("5", (1589, 983), "road"),  # blue road
    ("6", (948, 157), "ramp"),  # red fly ramp
    ("7", (1389, 1126), "ramp"),  # blue fly ramp
    ("8", (1092, 151), "tunnel"),  # upper road tunnel
    ("9", (1245, 1132), "tunnel"),  # lower road tunnel
    ("10", (520, 1060), "rough"),  # red uneven road
    ("11", (1817, 223), "rough"),  # blue uneven road
    ("×", (555, 646), "not_terrain"),  # red fortress
    ("×", (1782, 637), "not_terrain"),  # blue fortress
)

COLORS = {
    "highland": "#c477ff",
    "road": "#3fe5ff",
    "ramp": "#ff4fb8",
    "tunnel": "#ffffff",
    "rough": "#ffd238",
    "not_terrain": "#ff6b6b",
}


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size, index=0)


def centered_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text(
        (center[0] - width / 2, center[1] - height / 2 - box[1]),
        text,
        font=text_font,
        fill=fill,
    )


def marker(
    draw: ImageDraw.ImageDraw,
    label: str,
    center: tuple[int, int],
    category: str,
    label_font: ImageFont.FreeTypeFont,
) -> None:
    radius = 29
    x, y = center
    draw.ellipse(
        (x - radius + 3, y - radius + 5, x + radius + 3, y + radius + 5),
        fill="#05080c",
    )
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill="#121820",
        outline=COLORS[category],
        width=6,
    )
    if category == "tunnel":
        # Four gaps make the tunnel marker visually distinct without changing the map.
        for angle_box in (
            (x - 5, y - radius - 2, x + 5, y - radius + 8),
            (x - 5, y + radius - 8, x + 5, y + radius + 2),
            (x - radius - 2, y - 5, x - radius + 8, y + 5),
            (x + radius - 8, y - 5, x + radius + 2, y + 5),
        ):
            draw.rectangle(angle_box, fill="#121820")
    centered_text(draw, center, label, label_font, "#ffffff")


def chip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    category: str,
    chip_font: ImageFont.FreeTypeFont,
) -> int:
    x, y = xy
    width = 50 if len(label) == 1 else 84 if len(label) <= 3 else 104
    draw.rounded_rectangle(
        (x, y, x + width, y + 38),
        radius=19,
        fill="#17202a",
        outline=COLORS[category],
        width=4,
    )
    centered_text(draw, (x + width // 2, y + 19), label, chip_font, "#ffffff")
    return width


def main() -> None:
    source = Image.open(SOURCE_MAP).convert("RGB")
    if source.size != (MAP_WIDTH, MAP_HEIGHT):
        raise ValueError(f"unexpected map size: {source.size}")

    canvas = Image.new("RGB", (MAP_WIDTH, MAP_HEIGHT + PANEL_HEIGHT), "#10161d")
    canvas.paste(source, (0, 0))
    draw = ImageDraw.Draw(canvas)

    marker_font = font(FONT_BOLD, 25)
    title_font = font(FONT_BOLD, 31)
    body_font = font(FONT_REGULAR, 24)
    note_font = font(FONT_REGULAR, 21)
    chip_font = font(FONT_BOLD, 21)

    for label, center, category in MARKERS:
        marker(draw, label, center, category, marker_font)

    panel_y = MAP_HEIGHT
    draw.rectangle((0, panel_y, MAP_WIDTH, panel_y + PANEL_HEIGHT), fill="#10161d")
    draw.rectangle((0, panel_y, MAP_WIDTH, panel_y + 4), fill="#303b48")
    draw.text(
        (42, panel_y + 23),
        "RMUC 2026 地形定位 V0.2（编号只落点，不猜边界）",
        font=title_font,
        fill="#ffffff",
    )

    left_rows = (
        ("1", "highland", "中央高地：中部大平台；两端以 10.5° 坡连接公路区。"),
        ("2/3", "highland", "梯形高地：红方左上、蓝方右下；带 R 资源站的整块异形平台。"),
        ("4/5", "road", "公路区：两条长斜向通道，连接梯形高地与中央高地。"),
    )
    right_rows = (
        ("6/7", "ramp", "飞坡：三箭头标记处，坡度 17°。"),
        ("8/9", "tunnel", "公路隧道：中央高地上、下两处连接口。"),
        ("10/11", "rough", "起伏路段：红方左下、蓝方右上的密集凸起区。"),
    )

    for column_x, rows in ((43, left_rows), (1220, right_rows)):
        for index, (label, category, description) in enumerate(rows):
            y = panel_y + 84 + index * 60
            width = chip(draw, (column_x, y), label, category, chip_font)
            draw.text(
                (column_x + width + 18, y + 3),
                description,
                font=body_font,
                fill="#e9eef5",
            )

    draw.text(
        (1220, panel_y + 273),
        "红色 ×：六边形结构是堡垒，不是梯形高地。R 是资源站标识。",
        font=note_font,
        fill="#ffb8b8",
    )
    draw.text(
        (43, panel_y + 273),
        "依据：RMUC 2026 规则手册 V1.4 图 4-26、4-27、4-34、4-35、4-37。",
        font=note_font,
        fill="#b8c0cc",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, format="PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
