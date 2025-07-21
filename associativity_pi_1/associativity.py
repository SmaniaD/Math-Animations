#!/usr/bin/env python3
"""
Creates 'theta_s_isotopy_grid.gif' with:
  • curve t ↦ θ_s(t) for 0 ≤ s ≤ 1
  • break-point dots
  • horizontal dashed lines  y = 1/2, 3/4
  • vertical  dashed lines  t = 1/2, 3/4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------------------------------------
# θ_s(t) --- piecewise definition
def theta_s(t, s):
    t  = np.asarray(t)
    out = np.zeros_like(t)

    t1 = (2 - s) / 4        # first cut
    t2 = (3 - s) / 4        # second cut

    m1 = t <= t1
    m2 = (t > t1) & (t <= t2)
    m3 = t > t2

    out[m1] = (2 * t[m1]) / (2 - s)
    out[m2] = t[m2] + s / 4
    out[m3] = 0.75 + (t[m3] + s / 4 - 0.75) / (s + 1)
    return out

# -----------------------------------------------------------
# animation set-up
t_vals = np.linspace(0, 1, 400)

fig, ax = plt.subplots(figsize=(4, 4))
graph, = ax.plot([], [], lw=2, color="tab:orange")
dot1,  = ax.plot([], [], 'o', color="crimson")
dot2,  = ax.plot([], [], 'o', color="crimson")

# guidelines
ax.axhline(0.5,  lw=1, ls="--", color="orange")
ax.axhline(0.75, lw=1, ls="--", color="orange")
ax.axvline(0.25,  lw=1, ls="--", color="orange")
ax.axvline(0.50, lw=1, ls="--", color="orange")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("$t$")
#ax.set_ylabel("$\\theta_s(t)$")
title = ax.set_title("")

def init():
    graph.set_data([], [])
    dot1.set_data([], [])
    dot2.set_data([], [])
    title.set_text("")
    return graph, dot1, dot2, title

def animate(frame):
    s  = frame / 100          # 0, 0.01, …, 1.00
    y  = theta_s(t_vals, s)
    graph.set_data(t_vals, y)

    t1 = (2 - s) / 4
    t2 = (3 - s) / 4
    dot1.set_data([t1], [theta_s(t1, s)])
    dot2.set_data([t2], [theta_s(t2, s)])

    title.set_text(f"$\\theta_s(t)$,  $s={s:.2f}$")
    return graph, dot1, dot2, title

ani = animation.FuncAnimation(
    fig, animate, init_func=init,
    frames=101, interval=100, blit=True
)

ani.save("homotopy.gif", writer="pillow", fps=10)
print("GIF saved as 'homotopy.gif'")
