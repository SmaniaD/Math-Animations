 
"""
Tube Lemma Proof Visualization with Two Vertical Lines

This code generates an animation illustrating the Tube Lemma proof by showing
two colored dots moving vertically along separate vertical lines, with squares
being added at each position to demonstrate the finite subcover construction.

The output format (GIF, MP4, or interactive display) depends on command line parameters:
- python tube_two_verticals.py gif    # Saves as GIF only
- python tube_two_verticals.py mp4    # Saves as MP4 only  
- python tube_two_verticals.py        # Shows interactive display

Key Components:
- Generates a random open cover of circles over the unit square [0,1]²
- Animates two colored dots (red and blue) moving along vertical lines
- Constructs maximal inscribed squares at each position (U_y and V_y)
- Shows how squares accumulate to form a finite subcover
- Demonstrates the dependency of U_y and V_y on the x-coordinate

Global Parameters:
    SQ_MIN, SQ_MAX (float): Bounds of the unit square (0.0, 1.0)
    X_LINES (list): x-coordinates of the two vertical lines [0.25, 0.75]
    N_FRAMES (int): Total number of animation frames (160)
    FPS (int): Frames per second for animation playback (25)
    DOT_SIZE (int): Size of the moving dots in points (6)
    MARGIN (float): Safety margin for geometric calculations (0.015)
    SEED (int): Random seed for reproducibility (42)

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
import sys


# Check command line arguments for output format
SAVE_FORMAT = None
if len(sys.argv) > 1:
    format_arg = sys.argv[1].lower()
    if format_arg in ["gif", "mp4"]:
        SAVE_FORMAT = format_arg

# -------------------- global parameters --------------------
SQ_MIN, SQ_MAX = 0.0, 1.0
X_LINES        = [0.25, 0.75]   # x positions of the two vertical lines
N_FRAMES       = 160
FPS            = 25
DOT_SIZE       = 6
MARGIN         = 0.015
SEED           = 42  # random seed for reproducibility




# ---------- generate open cover (same algorithm) ----------
def generate_cover():

    rng   = np.random.default_rng(SEED)
    grid  = np.linspace(0.0, 1.0, 300)
    xx, yy = np.meshgrid(grid, grid)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    uncovered = np.ones(len(pts), dtype=bool)
    circles   = []

    R_MIN, R_MAX, TOL = 0.20, 0.40, 1e-12

    # greedy phase
    while uncovered.any():
        p = pts[np.argmax(uncovered)]
        r = rng.uniform(R_MIN, R_MAX)
        circles.append({"center": tuple(p), "radius": r})
        dist = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
        uncovered &= dist > r + TOL

    # safe minimization (remove redundant disks)
    i = 0
    while i < len(circles):
        test_circs = circles[:i] + circles[i+1:]
        still_covered = np.all([
            any(np.hypot(x - c["center"][0], y - c["center"][1]) <= c["radius"] + TOL
                for c in test_circs)
            for x, y in pts
        ])
        if still_covered:
            circles.pop(i)
        else:
            i += 1

    print(f"Final circles: {len(circles)}")
    return circles

OPEN_SETS = generate_cover()

# color palette (recommended API from Matplotlib ≥ 3.7)
cmap   = plt.colormaps.get_cmap("tab10")
COLORS = cycle([cmap(i) for i in range(cmap.N)])

# --------------- geometric utilities ----------------------
def inside_circle(pt, center, radius):
    """Signed distance: positive inside the disk."""
    return radius - np.hypot(pt[0] - center[0], pt[1] - center[1])

def rect_side_length(pt):
    """Largest square inscribed in some disk that contains pt."""
    candidates = []
    for c in OPEN_SETS:
        d = inside_circle(pt, c["center"], c["radius"])
        if d > 0:
            s = (d - MARGIN) * np.sqrt(2)
            if s > 0:
                candidates.append(s)
    return max(candidates) if candidates else 0.0

def point_in_rect(pt, rect):
    """Check if pt is inside a fixed rectangle."""
    x, y, s = rect
    return (abs(pt[0] - x) <= s/2 + 1e-12) and (abs(pt[1] - y) <= s/2 + 1e-12)

# ---------------- base figure -------------------------------
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect("equal")
ax.set_xlim(SQ_MIN, SQ_MAX)
ax.set_ylim(SQ_MIN, SQ_MAX)
ax.axis("off")

# contour of the unit square [0,1]²
ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, lw=2))

# styled axes
ax.text(0.5, -0.06, "X", ha="center", va="center",
        fontsize=16, color="blue", transform=ax.transAxes)
ax.text(-0.06, 0.5, "Y", ha="center", va="center",
        fontsize=16, color="blue", rotation=90, transform=ax.transAxes)

# vertical lines
for xl in X_LINES:
    ax.plot([xl, xl], [0, 1], lw=2, color="black")

# cover disks
for circ, col in zip(OPEN_SETS, COLORS):
    ax.add_patch(patches.Circle(
        circ["center"], circ["radius"],
        facecolor=col, alpha=0.35, edgecolor=col, lw=1.5))

# -------- dynamic artists and state structures ----------
dots         = []                 # moving markers
red_rects    = []                 # instantaneous squares
Uy_texts     = []                 # U_y labels
Vy_texts     = []                 # V_y labels
stored_rects = [[], []]           # rectangles fixed by line
colors_pts   = ["red", "blue"]    # colors of the two points

for col in colors_pts:
    # point
    dot, = ax.plot([], [], "o", color=col, markersize=DOT_SIZE)
    dots.append(dot)

    # instantaneous square
    rect = patches.Rectangle((0, 0), 0, 0, fill=False, lw=2, edgecolor=col)
    ax.add_patch(rect)
    red_rects.append(rect)

    # texts
    Uy_texts.append(ax.text(0, 0, "", color=col, fontsize=11,
                            ha="center", va="top"))
    Vy_texts.append(ax.text(0, 0, "", color=col, fontsize=11,
                            ha="right",  va="center"))

# --------------- animation ------------------------------------
def init():
    artists = []
    for rect, Uy, Vy in zip(red_rects, Uy_texts, Vy_texts):
        rect.set_visible(False)
        Uy.set_visible(False)
        Vy.set_visible(False)
        artists.extend([rect, Uy, Vy])
    return (*dots, *artists)

def update(frame):
    """Move the points and draw/update the elements."""
    y = 1 - frame / (N_FRAMES - 1)
    artists = list(dots)

    for idx, xl in enumerate(X_LINES):
        pt = (xl, y)
        # set_data needs sequences → pass lists!
        dots[idx].set_data([pt[0]], [pt[1]])

        side = rect_side_length(pt)
        if side <= 0:
            red_rects[idx].set_visible(False)
            Uy_texts[idx].set_visible(False)
            Vy_texts[idx].set_visible(False)
            continue

        # check if there's already a fixed square that contains pt
        covered = any(point_in_rect(pt, r[:3]) for r in stored_rects[idx])

        # update instantaneous square
        rr = red_rects[idx]
        rr.set_visible(True)
        rr.set_width(side)
        rr.set_height(side)
        rr.set_xy((pt[0] - side/2, pt[1] - side/2))

        # labels
        Uy_texts[idx].set_text(r"$U_y$")
        Uy_texts[idx].set_position((pt[0]+0.04, pt[1] - side/2 - 0.015))
        Uy_texts[idx].set_visible(True)

        Vy_texts[idx].set_text(r"$V_y$")
        Vy_texts[idx].set_position((pt[0] - side/2 - 0.015, pt[1]))
        Vy_texts[idx].set_visible(True)

        # if pt wasn't covered by a fixed square, fix it now
        if not covered:
            patch = patches.Rectangle(
                (pt[0] - side/2, pt[1] - side/2),
                side, side,
                facecolor="white", alpha=0.4,
                edgecolor="black", lw=1.2)
            ax.add_patch(patch)
            stored_rects[idx].append((pt[0], pt[1], side, patch))

        artists.extend([rr, Uy_texts[idx], Vy_texts[idx]])
        artists.extend([r[3] for r in stored_rects[idx]])

    return tuple(artists)

ani = animation.FuncAnimation(
    fig, update,
    frames=N_FRAMES,
    init_func=init,
    interval=1000/FPS,
    blit=False
)


# Save based on SAVE_FORMAT or save both if not specified
if SAVE_FORMAT == "gif":
    ani.save("tube_lemma_two_lines.gif", writer="pillow", fps=FPS)
    print("GIF animation saved")
elif SAVE_FORMAT == "mp4":
    ani.save("tube_lemma_two_lines.mp4", writer="ffmpeg", fps=FPS)
    print("MP4 animation saved")
else:
    plt.show()  # Show the animation in an interactive window

