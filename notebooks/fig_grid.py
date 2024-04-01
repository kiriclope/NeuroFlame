import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from svgutils.compose import SVG, Figure, Text


def get_svg_size(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()
    width = root.attrib.get("width")
    height = root.attrib.get("height")

    if "mm" in width:
        width = 2.8346456693 * float(width.replace("mm", ""))
        height = 2.8346456693 * float(height.replace("mm", ""))
    else:
        width = float(
            width.replace("pt", "")
        )  # This removes "pt" and converts the number to int
        height = float(
            height.replace("pt", "")
        )  # This removes "pt" and converts the number to int
        
    return width, height


def create_test_svg(title):
    plt.figure()
    plt.title(title)
    plt.plot([0, 1], [0, 1], "-o")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(title + ".svg", format="svg")
    plt.close("all")

def create_grid(figlist, figname, dim):
    cols, rows = dim
    max_widths = [0] * cols
    max_heights = [0] * rows

    for i, fig in enumerate(figlist):
        col = i % cols
        row = i // cols
        width, height = get_svg_size(fig)
        max_widths[col] = max(max_widths[col], width)
        max_heights[row] = max(max_heights[row], height)

    total_width = sum(max_widths)
    total_height = sum(max_heights)
    size = ["%dpt" % total_width, "%dpt" % total_height]
    print(size)

    # create new SVG figure
    panels = []
    current_x = 0
    current_y = 0

    for i, fig in enumerate(figlist):
        width, height = get_svg_size(fig)
        col = i % cols
        row = i // cols
        
        if col == 0:
            current_x = 0
        if i > 0 and col == 0:
            current_y += max_heights[row - 1]

        svg = SVG(fig).move(current_x, current_y)
        label = Text(
            chr(65 + i),
            current_x + 5,
            current_y + 15,
            size=20,
            weight="bold",
        )
        panels.append(svg)
        panels.append(label)

        current_x += max_widths[col]

    grid = Figure(size[0], size[1], *panels)
    grid.save(figname)

if __name__ == "__main__":
    width = 7
    golden_ratio = (5**0.5 - 1) / 2
    height = width * golden_ratio
    matplotlib.rcParams["figure.figsize"] = [width, height]

    create_test_svg("A")
    create_test_svg("B")
    create_test_svg("C")
    create_test_svg("D")

    figlist = ["A.svg", "B.svg", "C.svg", "D.svg", "A.svg"]
    dim = [2, 3]

    # width, height = get_svg_size("A.svg")

    # print(width, height)

    # size = ["%dpt" % (dim[0] * width), "%dpt" % (dim[1] * height)]

    # print(size)

    create_grid(figlist, "fig_final.svg", dim)
