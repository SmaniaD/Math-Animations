 
"""
Tube Lemma Proof  Visualization with Moving Dot on Vertical Line
This code generates an animation illustrating the Tube Lemma
by constructing a cover of circles and showing how squares fit inside them.
The animation shows a red dot moving vertically along a line,
with squares being added as the dot moves, demonstrating the proof of the tube lemma visually.

This module creates an animated visualization of the Tube Lemma proof in topology,
The animation shows a red dot moving vertically while squares are constructed around it,
illustrating the dependency of neighborhoods U_y and V_y on the y-coordinate.

The output format (GIF, MP4, or interactive display) depends on command line parameters:
- python tube.py gif    # Saves as GIF only
- python tube.py mp4    # Saves as MP4 only  
- python tube.py        # Shows interactive display

Key Components:
- Generates a random open cover of circles over the unit square [0,1]²
- Animates a red dot moving along a vertical line
- Constructs maximal inscribed squares at each position
- Shows how squares accumulate to form a finite subcover
- Exports the animation as both GIF and MP4 formats

Global Parameters:
    SQ_MIN, SQ_MAX (float): Bounds of the unit square (0.0, 1.0)
    X_LINE (float): x-coordinate of the vertical line (0.5)
    N_FRAMES (int): Total number of animation frames (160)
    FPS (int): Frames per second for animation playback (25)
    DOT_SIZE (int): Size of the moving dot in points (6)
    MARGIN (float): Safety margin for geometric calculations (0.015)

Functions:
    generate_cover(): Creates optimized open cover using greedy algorithm
    inside_circle(pt, center, radius): Computes signed distance to circle boundary
    rect_side_length(pt): Finds largest inscribed square size at given point
    point_in_rect(pt, rect): Tests point containment in rectangle
    init(): Animation initialization function
    update(frame): Animation frame update function
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from itertools import cycle
from matplotlib.cm import get_cmap
import sys

# Check command line arguments for output format
SAVE_FORMAT = None
if len(sys.argv) > 1:
    format_arg = sys.argv[1].lower()
    if format_arg in ["gif", "mp4"]:
        SAVE_FORMAT = format_arg

# -------------------- global parameters --------------------
SQ_MIN, SQ_MAX = 0.0, 1.0    # bounds of the unit square [0,1] x [0,1]
X_LINE         = 0.5          # vertical line position where the red dot moves
N_FRAMES       = 160          # total number of animation frames
FPS            = 25           # frames per second for animation playback
DOT_SIZE       = 6            # size of the red moving dot marker
MARGIN         = 0.015        # safety margin for rectangle fitting inside circles
SEED           = 42           # random seed for reproducibility

def generate_cover():
    """
    (1) Constructs a greedy cover of circles with radius ∈ [0.20, 0.40]
        until ALL points of a 300×300 grid are covered.
    (2) Removes redundant circles while ensuring the cover remains total.
    Returns the final list.
    """
    # create a random number generator
    rng = np.random.default_rng(SEED)  
    grid  = np.linspace(0.0, 1.0, 300)
    # creates complete 300x300 grid (all points of the square [0,1]x[0,1])
    xx, yy = np.meshgrid(grid, grid)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    uncovered = np.ones(len(pts), dtype=bool)
    circles   = []

    R_MIN, R_MAX = 0.20, 0.40
    TOL = 1e-12

    # ---------- greedy phase ----------
    while uncovered.any():
        # point still uncovered
        p = pts[np.argmax(uncovered)]

        # random radius in [R_MIN, R_MAX]
        r = rng.uniform(R_MIN, R_MAX)

        circles.append({"center": tuple(p), "radius": r})

        # points now covered
        dist = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
        uncovered &= dist > r + TOL

    # ---------- safe minimization phase ----------
    i = 0
    while i < len(circles):
        # try to remove circle i
        test_circs = circles[:i] + circles[i+1:]

        # check entire grid
        still_covered = np.all([
            any(np.hypot(x - c["center"][0], y - c["center"][1]) <= c["radius"] + TOL
                for c in test_circs)
            for x, y in pts
        ])

        if still_covered:
            circles.pop(i)          # redundant: remove and DON'T advance index
        else:
            i += 1                  # necessary: move to next

    print(f"Final circles: {len(circles)}")
    return circles

OPEN_SETS = generate_cover()

# palette
cmap   = get_cmap("tab10")
COLORS = cycle([cmap(i) for i in range(cmap.N)])

# -------------------- utilities -----------------------------
def inside_circle(pt, center, radius):
    """d > 0 ⇒ inside the circle."""
    return radius - np.hypot(pt[0]-center[0], pt[1]-center[1])

def rect_side_length(pt):
    """largest side s such that square centered at pt fits in a circle."""
    candidates = []
    for c in OPEN_SETS:
        d = inside_circle(pt, c["center"], c["radius"])
        if d > 0:
            s = (d - MARGIN) * np.sqrt(2)
            if s > 0:
                candidates.append(s)
    return max(candidates) if candidates else 0.0

def point_in_rect(pt, rect):
    """checks if point is inside the stored rectangle."""
    x, y, s = rect  # center x, center y, side
    return (abs(pt[0] - x) <= s/2 + 1e-12) and (abs(pt[1] - y) <= s/2 + 1e-12)

# -------------------- base figure ----------------------------
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect("equal")
ax.set_xlim(SQ_MIN, SQ_MAX)
ax.set_ylim(SQ_MIN, SQ_MAX)
ax.axis("off")

# square
ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, lw=2))

# X and Y axes
ax.text(0.5, -0.06, "X", ha="center", va="center",
        fontsize=16, color="blue", transform=ax.transAxes)
ax.text(-0.06, 0.5, "Y", ha="center", va="center",
        fontsize=16, color="blue", rotation=90, transform=ax.transAxes)

# black vertical line
ax.plot([X_LINE, X_LINE], [0, 1], lw=2, color="black")

# open circles
for circ, col in zip(OPEN_SETS, COLORS):
    ax.add_patch(patches.Circle(
        circ["center"], circ["radius"],
        facecolor=col, alpha=0.35, edgecolor=col, lw=1.5))

# red dot (mobile)
dot, = ax.plot([], [], "o", color="red", markersize=DOT_SIZE)

# current point rectangle (red)
red_rect = patches.Rectangle((0, 0), 0, 0, fill=False,
                             lw=2, edgecolor="red")
ax.add_patch(red_rect)
Uy_text = ax.text(0, 0, "", color="red", fontsize=11,
                  ha="center", va="top")
Vy_text = ax.text(0, 0, "", color="red", fontsize=11,
                  ha="right",  va="center")

# list of fixed rectangles (gray)
stored_rects = []          # elements = (xc, yc, side, patch)

# -------------------- animation --------------------------------
def init():
    red_rect.set_visible(False)
    Uy_text.set_visible(False)
    Vy_text.set_visible(False)
    return (dot, red_rect, Uy_text, Vy_text)

def update(frame):
    # point at (X_LINE, y)
    y = 1 - frame / (N_FRAMES - 1)
    pt = (X_LINE, y)
    dot.set_data(*pt)

    # calculate maximum side for the point
    side = rect_side_length(pt)
    if side <= 0:
        red_rect.set_visible(False)
        Uy_text.set_visible(False)
        Vy_text.set_visible(False)
        return (dot, red_rect, Uy_text, Vy_text, *[r[3] for r in stored_rects])

    # check if already covered by stored rectangle
    covered = any(point_in_rect(pt, r[:3]) for r in stored_rects)

    # current square (red)
    red_rect.set_visible(True)
    red_rect.set_width(side)
    red_rect.set_height(side)
    red_rect.set_xy((pt[0] - side/2, pt[1] - side/2))

    Uy_text.set_text(r"$U_y$")
    Uy_text.set_position((pt[0] + 0.04, pt[1] - side/2 - 0.014)) 
    Uy_text.set_visible(True)

    Vy_text.set_text(r"$V_y$")
    Vy_text.set_position((pt[0] - side/2 - 0.015, pt[1]))
    Vy_text.set_visible(True)

    # if not covered, add fixed gray rectangle
    if not covered:
        patch = patches.Rectangle(
            (pt[0] - side/2, pt[1] - side/2),
            side, side,
            facecolor="white", alpha=0.4,
            edgecolor="black", lw=1.2)
        ax.add_patch(patch)
        stored_rects.append((pt[0], pt[1], side, patch))

    # return all artists to be redrawn
    return (dot, red_rect, Uy_text, Vy_text,
            *[r[3] for r in stored_rects])

ani = animation.FuncAnimation(
    fig, update,
    frames=N_FRAMES,
    init_func=init,
    interval=1000/FPS,
    blit=False   # ← new patches are added over time
)



# Save based on SAVE_FORMAT or save both if not specified
if SAVE_FORMAT == "gif":
    ani.save("tube_lemma.gif", writer="pillow", fps=FPS)
    print("GIF animation saved")
elif SAVE_FORMAT == "mp4":
    ani.save("tube_lemma.mp4", writer="ffmpeg", fps=FPS)
    print("MP4 animation saved")
else:
    plt.show()  # Show the animation in an interactive window
