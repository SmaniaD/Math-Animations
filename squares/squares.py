import argparse
import random
import math
import numpy as np
import networkx as nx
from PIL import Image
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation

# ==========================
# Utilities
# ==========================

def get_border_edges(grid, size):
    """
    Returns border edge segments of active cells (1) in binary grid.
    Each edge is a pair of points ((x1,y1),(x2,y2)) in plane coordinates,
    where each cell has side = size.
    """
    h, w = len(grid), len(grid[0])
    edges = []
    for y in range(h):
        for x in range(w):
            if grid[y][x]:
                sx, sy = x * size, y * size
                # Left
                if x == 0 or not grid[y][x-1]:
                    edges.append(((sx, sy), (sx, sy+size)))
                # Right
                if x == w-1 or not grid[y][x+1]:
                    edges.append(((sx+size, sy), (sx+size, sy+size)))
                # Top
                if y == 0 or not grid[y-1][x]:
                    edges.append(((sx, sy), (sx+size, sy)))
                # Bottom
                if y == h-1 or not grid[y+1][x]:
                    edges.append(((sx, sy+size), (sx+size, sy+size)))
    return edges


def make_regions(grid_size, square_size, seed=42, color=[0, 0, 0]):
    """
    Generates RGB image with black background and light color scheme.
    """
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    # Light gray instead of dark gray
    light_gray = np.array([200, 200, 200], dtype=np.uint8)

    random.seed(seed)
    y, x = np.ogrid[:grid_size, :grid_size]

    # Gray region
    gray_mask = np.zeros((grid_size, grid_size), dtype=bool)
    border_margin = grid_size // 12
    num_gray_balls = random.randint(8, 16)

    for _ in range(num_gray_balls):
        cx = random.randint(border_margin, grid_size - border_margin - 1)
        cy = random.randint(border_margin, grid_size - border_margin - 1)
        max_r = min(
            cx - border_margin,
            cy - border_margin,
            grid_size - border_margin - cx - 1,
            grid_size - border_margin - cy - 1,
            grid_size // 4
        )
        min_r = grid_size // 8
        r = min_r if max_r < min_r else random.randint(min_r, max_r)
        gray_mask |= (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

    # Remove gray pixels touching border
    gray_mask[0, :] = False
    gray_mask[-1, :] = False
    gray_mask[:, 0] = False
    gray_mask[:, -1] = False

    # Black background + light gray
    grid[:, :] = [0, 0, 0]  # Black background
    grid[gray_mask] = light_gray

    # Colored region inside gray
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    num_balls = random.randint(8, 16)
    for _ in range(num_balls):
        cx = random.randint(grid_size // 6, grid_size * 5 // 6)
        cy = random.randint(grid_size // 6, grid_size * 5 // 6)
        r = random.randint(grid_size // 8, grid_size // 4)
        eroded_gray = binary_erosion(gray_mask, structure=np.ones((r, r), dtype=bool))
        new_mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        mask |= new_mask & eroded_gray

    # Holes in colored region
    num_holes = random.randint(10, 20)
    for _ in range(num_holes):
        cx = random.randint(grid_size // 6, grid_size * 5 // 6)
        cy = random.randint(grid_size // 6, grid_size * 5 // 6)
        r = random.randint(grid_size // 20, grid_size // 10)
        hole_mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        mask[hole_mask] = False

    # Ensure blocks touched by mask are inside gray
    n = grid_size // square_size
    for by in range(n):
        for bx in range(n):
            sy, sx = by * square_size, bx * square_size
            block = mask[sy:sy+square_size, sx:sx+square_size]
            if np.any(block):
                gray_block = gray_mask[sy:sy+square_size, sx:sx+square_size]
                mask[sy:sy+square_size, sx:sx+square_size] &= gray_block

    grid[mask] = np.array(color, dtype=np.uint8)
    return np.array(Image.fromarray(grid))


def eulerian_or_dfs_path(undirected_graph):
    """
    Returns a list of edges (u,v) forming a path to draw the border.
    Tries Eulerian circuit; if not possible, uses DFS.
    """
    if nx.is_eulerian(undirected_graph):
        return list(nx.eulerian_circuit(undirected_graph))
    else:
        return list(nx.dfs_edges(undirected_graph))


# ==========================
# Main program
# ==========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--grid_size', type=int, default=512)
    parser.add_argument('--square_size', type=int, default=16)
    parser.add_argument('--small_square', type=int, default=4)
    parser.add_argument('--color', type=str, default='0,150,255')  # Modern blue
    parser.add_argument('--save', action='store_true', help='Save GIF instead of showing')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--outfile', type=str, default='pipeline_animation.gif')
    parser.add_argument('--edges_per_frame', type=int, default=6,
                        help='How many new edges appear per frame in phases 5 and 7')
    args = parser.parse_args()

    grid_size = args.grid_size
    square_size = args.square_size
    small_square = args.small_square
    n = grid_size // square_size
    n_small = grid_size // small_square
    ink_color = tuple(int(c) for c in args.color.split(','))

    # 1) Generate base image
    image = make_regions(grid_size, square_size, seed=args.seed, color=ink_color)

    # 2) Build coarse grid (0/1) from image
    grid = np.zeros((n, n), dtype=int)
    for by in range(n):
        for bx in range(n):
            sy, sx = by * square_size, bx * square_size
            block = image[sy:sy+square_size, sx:sx+square_size]
            color_mask = np.all(block == np.array(ink_color, dtype=np.uint8), axis=-1)
            gray_mask = np.all(block == np.array([200, 200, 200], dtype=np.uint8), axis=-1)
            if np.sum(color_mask) > 0 and np.all(gray_mask | color_mask):
                grid[by, bx] = 1

    # Zero borders of coarse grid
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0

    # 3) Refine to small grid
    small_grid = np.zeros((n_small, n_small), dtype=int)
    factor = square_size // small_square
    for by in range(n):
        for bx in range(n):
            if grid[by, bx]:
                sy = by * factor
                sx = bx * factor
                small_grid[sy:sy+factor, sx:sx+factor] = 1

    # Fix corner ambiguity (diagonal fillings)
    for y in range(1, n-1):
        for x in range(1, n-1):
            if grid[y-1][x-1] and grid[y][x] and not grid[y][x-1] and not grid[y-1][x]:
                for dy in [0, 1]:
                    for dx in [0, 1]:
                        sx = x*square_size + (dx-1) * small_square
                        sy = y*square_size + (dy-1) * small_square
                        small_x = sx // small_square
                        small_y = sy // small_square
                        if 0 <= small_x < n_small and 0 <= small_y < n_small:
                            small_grid[small_y][small_x] = 1
            if grid[y][x-1] and grid[y-1][x] and not grid[y][x] and not grid[y-1][x-1]:
                for dy in [0, 1]:
                    for dx in [0, 1]:
                        sx = x*square_size + (dx-1) * small_square
                        sy = y*square_size + (dy-1) * small_square
                        small_x = sx // small_square
                        small_y = sy // small_square
                        if 0 <= small_x < n_small and 0 <= small_y < n_small:
                            small_grid[small_y][small_x] = 1

    # 4) Inflation
    inflated_grid = np.array(small_grid)
    inflated_grid = np.pad(inflated_grid, 1, mode='constant')
    inflated_grid_new = np.zeros_like(inflated_grid)
    added_squares = np.zeros_like(inflated_grid)
    
    for y in range(1, inflated_grid.shape[0]-1):
        for x in range(1, inflated_grid.shape[1]-1):
            if np.any(inflated_grid[y-1:y+2, x-1:x+2]):
                inflated_grid_new[y, x] = 1
                if inflated_grid[y, x] == 0:
                    added_squares[y, x] = 1
    
    inflated_grid = inflated_grid_new[1:-1, 1:-1]
    added_squares = added_squares[1:-1, 1:-1]

    # 5) Border edges and Jordan curves
    small_edges = get_border_edges(small_grid.tolist(), small_square)
    G_small = nx.Graph()
    for (a, b) in small_edges:
        G_small.add_edge(a, b)
    comps_small = list(nx.connected_components(G_small))
    draw_paths_small = []
    for comp in comps_small:
        sub = G_small.subgraph(comp)
        path = eulerian_or_dfs_path(sub)
        draw_paths_small.append(path)

    infl_edges = get_border_edges(inflated_grid.tolist(), small_square)
    G_infl = nx.Graph()
    for (a, b) in infl_edges:
        G_infl.add_edge(a, b)
    comps_infl = list(nx.connected_components(G_infl))
    draw_paths_infl = []
    for comp in comps_infl:
        sub = G_infl.subgraph(comp)
        path = eulerian_or_dfs_path(sub)
        draw_paths_infl.append(path)

    # ==========================
    # ANIMATION - Portrait format
    # ==========================

    plt.style.use('dark_background')  # Dark theme
    fig, ax = plt.subplots(figsize=(6, 10))  # Portrait aspect ratio
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.axis('off')
     

    # Background image with higher alpha for visibility on black
    bg = ax.imshow(image, extent=[0, grid_size, 0, grid_size],
                   origin='lower', alpha=0.5, zorder=10)

    # Modern color schemes
    cmap = mcolors.ListedColormap(['black', '#00D4FF'])  # Cyan
    cmap2 = mcolors.ListedColormap(['black', '#FF3D71'])  # Modern pink/red

    # Animation phases
    phases = [
        "Given an open set V ⊂ ℝ² \n and a compact set K\ninside it, there is an open set U \n that contains K,its closure is inside V,\n  and its boundary is a\ndisjoint union of Jordan curves.",
        "Compact set inside open set",
        "Cover of compact set by squares", 
        "Refined cover to avoid\nproblematic vertices",
        "Finding edges in the border",
        "Finding Jordan curves\nin the border",
        "Inflated cover to avoid intersection\nof borders with compact set",
        "Finding Jordan curves in the\nborder (inflated)",
        "We are done!",
    ]
    
    default_frames_per_phase = [80, 40, 40, 40, 40, 40, 40, 40, 80]  # Added 40 frames for theorem statement

    # Pre-create all plot elements
    coarse_layer = ax.imshow(np.zeros_like(grid), extent=[0, grid_size, 0, grid_size],
                             origin='lower', cmap=cmap, vmin=0, vmax=1, alpha=0, zorder=1)
    
    small_layer = ax.imshow(np.zeros_like(small_grid), extent=[0, grid_size, 0, grid_size],
                            origin='lower', cmap=cmap, vmin=0, vmax=1, alpha=0, zorder=2)
    
    infl_layer = ax.imshow(np.zeros_like(inflated_grid), extent=[0, grid_size, 0, grid_size],
                           origin='lower', cmap=cmap2, vmin=0, vmax=1, alpha=0, zorder=3)

    # Modern color palette for edges
    modern_colors = ['#00D4FF', '#FF3D71', '#00E676', '#FFD740', '#E040FB', '#FF5722', '#03DAC6', '#FFAB00']

    # Edge lines
    edge_lines = []
    for edge in small_edges:
        line, = ax.plot([], [], color='white', linewidth=1.5, alpha=0, zorder=4)
        edge_lines.append((line, edge))

    # Jordan curve lines
    jordan_lines_small = []
    for i, path in enumerate(draw_paths_small):
        line, = ax.plot([], [], color=modern_colors[i % len(modern_colors)], 
                       linewidth=3, alpha=0, zorder=5)
        jordan_lines_small.append((line, path))

    jordan_lines_infl = []
    for i, path in enumerate(draw_paths_infl):
        line, = ax.plot([], [], color=modern_colors[(i+3) % len(modern_colors)], 
                       linewidth=3, alpha=0, zorder=6)
        jordan_lines_infl.append((line, path))

    # Edge-by-edge lines
    edge_by_edge_lines_small = []
    for i, path in enumerate(draw_paths_small):
        color = modern_colors[i % len(modern_colors)]
        for j, edge in enumerate(path):
            line, = ax.plot([], [], color=color, linewidth=2, alpha=0, zorder=7)
            edge_by_edge_lines_small.append((line, edge, i, j))
    
    edge_by_edge_lines = []
    for i, path in enumerate(draw_paths_infl):
        color = modern_colors[(i+3) % len(modern_colors)]
        for j, edge in enumerate(path):
            line, = ax.plot([], [], color=color, linewidth=2, alpha=0, zorder=7)
            edge_by_edge_lines.append((line, edge, i, j))

    # Calculate frame counts
    total_edges_small = len(edge_by_edge_lines_small)
    total_edges_infl = len(edge_by_edge_lines)
    edges_per_frame = max(1, int(args.edges_per_frame))

    frames_phase5 = max(30, math.ceil(total_edges_small / edges_per_frame))  # Updated variable name
    frames_phase7 = max(30, math.ceil(total_edges_infl / edges_per_frame))   # Updated variable name

    phase_frames = default_frames_per_phase[:]
    phase_frames[5] = frames_phase5  # Updated index
    phase_frames[7] = frames_phase7  # Updated index
    phase_frames[8]= 80  # Added frames for theorem statement

    cum_frames = [0]
    for f in phase_frames:
        cum_frames.append(cum_frames[-1] + f)
    total_frames = cum_frames[-1]

    # Title with modern styling
    title_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, 
                         fontsize=16, ha='center', va='top', zorder=10,
                         color='white', weight='bold')

    def frame_to_phase(frame_idx):
        for p in range(len(phase_frames)):
            if cum_frames[p] <= frame_idx < cum_frames[p+1]:
                return p, frame_idx - cum_frames[p]
        return len(phase_frames)-1, phase_frames[-1]-1

    def update(frame):
        phase, phase_frame = frame_to_phase(frame)
        alpha = min(1.0, phase_frame / 10.0) if phase not in (5, 7) else 1.0  # Updated phase numbers

        # Reset alphas
        coarse_layer.set_alpha(0)
        small_layer.set_alpha(0) 
        infl_layer.set_alpha(0)
        for line, _ in edge_lines:
            line.set_alpha(0)
        for line, _ in jordan_lines_small:
            line.set_alpha(0)
        for line, _ in jordan_lines_infl:
            line.set_alpha(0)
        for line, _, _, _ in edge_by_edge_lines_small:
            line.set_alpha(0)
        for line, _, _, _ in edge_by_edge_lines:
            line.set_alpha(0)

        title_text.set_text(phases[phase])
        title_text.set_position((0.5, 1.4))  # Move text to top of image

        if phase == 0:  # New theorem statement phase
            pass  # Just show the theorem text and background image
            
        elif phase == 1:  # Previously phase 0
            pass
            
        elif phase == 2:  # Previously phase 1
            coarse_layer.set_data(grid)
            coarse_layer.set_alpha(alpha * 0.8)
            
        elif phase == 3:  # Previously phase 2
            coarse_layer.set_data(grid)
            coarse_layer.set_alpha(0.3)
            small_layer.set_data(small_grid)
            small_layer.set_alpha(alpha * 0.8)
            
        elif phase == 4:  # Previously phase 3
            small_layer.set_data(small_grid)
            small_layer.set_alpha(0.6)
            for line, edge in edge_lines:
                (x1, y1), (x2, y2) = edge
                line.set_data([x1, x2], [y1, y2])
                line.set_alpha(alpha * 0.8)
            
        elif phase == 5:  # Previously phase 4
            small_layer.set_data(small_grid)
            small_layer.set_alpha(0.4)
            show_k = min(total_edges_small, (phase_frame + 1) * edges_per_frame)
            for idx in range(show_k):
                line, edge, comp_i, j = edge_by_edge_lines_small[idx]
                (u, v) = edge
                (x1, y1), (x2, y2) = u, v
                line.set_data([x1, x2], [y1, y2])
                line.set_alpha(1.0)
            
        elif phase == 6:  # Previously phase 5
            small_layer.set_data(small_grid)
            small_layer.set_alpha(0.3)
            infl_layer.set_data(inflated_grid)
            infl_layer.set_alpha(alpha * 0.8)
            
        elif phase == 7:  # Previously phase 6
            infl_layer.set_data(inflated_grid)
            infl_layer.set_alpha(0.4)
            show_k = min(total_edges_infl, (phase_frame + 1) * edges_per_frame)
            for idx in range(show_k):
                line, edge, comp_i, j = edge_by_edge_lines[idx]
                (u, v) = edge
                (x1, y1), (x2, y2) = u, v
                line.set_data([x1, x2], [y1, y2])
                line.set_alpha(1.0)

        elif phase == 8:
            infl_layer.set_data(inflated_grid)
            infl_layer.set_alpha(0.5)
            for line, path in jordan_lines_infl:
                if path:  # Check if path is not empty
                    # Extract coordinates from the path of edges
                    coords = []
                    for edge in path:
                        (u, v) = edge
                        if not coords:  # First edge
                            coords.extend([u, v])
                        else:
                            coords.append(v)
                    if coords:
                        x, y = zip(*coords)
                        line.set_data(x, y)
                        line.set_alpha(alpha * 0.8)


        return [title_text, coarse_layer, small_layer, infl_layer] + \
               [line for line, _ in edge_lines] + \
               [line for line, _, _, _ in edge_by_edge_lines_small] + \
               [line for line, _, _, _ in edge_by_edge_lines] + \
               [line for line, _ in jordan_lines_small] + \
               [line for line, _ in jordan_lines_infl]

    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=int(1000/args.fps), blit=False
    )

    if args.save:
        if args.outfile.lower().endswith('.mp4'):
            ani.save(args.outfile, writer='ffmpeg', fps=args.fps)
            print(f"MP4 saved as {args.outfile}")
        else:
            if not args.outfile.lower().endswith('.gif'):
                args.outfile = args.outfile.rsplit('.', 1)[0] + '.gif'
            ani.save(args.outfile, writer='pillow', fps=args.fps)
            print(f"GIF saved as {args.outfile}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
