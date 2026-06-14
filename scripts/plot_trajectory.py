#!/usr/bin/env python3
"""
Live trajectory plot for TUM-format output from SlidingWindowGraph::writeTUM().

Usage:
    python3 scripts/plot_trajectory.py [trajectory.txt]

The plot refreshes every 200 ms as the C++ system appends new poses.
TUM format: timestamp tx ty tz qx qy qz qw

Left panel  — top-down  (X vs Z)
Right panel — height + speed over time
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("trajectory.txt")

fig, (ax_top, ax_time) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Live trajectory — {PATH}", fontsize=10)

line_top,  = ax_top.plot([], [], "o-", ms=3, lw=1, color="#00c8a0")
point_cur, = ax_top.plot([], [], "o",  ms=8, color="#ff6040", zorder=5)

line_y,    = ax_time.plot([], [], lw=1.2, label="Y (height)")
line_spd,  = ax_time.plot([], [], lw=1.2, label="speed (Δpos/Δt)", linestyle="--")

ax_top.set_xlabel("X  [m]");  ax_top.set_ylabel("Z  [m]")
ax_top.set_title("Top-down (X–Z plane)");  ax_top.set_aspect("equal")
ax_top.grid(True, alpha=0.3)

ax_time.set_xlabel("time  [s]");  ax_time.set_ylabel("[m] / [m/s]")
ax_time.set_title("Height & speed over time")
ax_time.legend(fontsize=8);  ax_time.grid(True, alpha=0.3)


def load(path: Path):
    """Return (N,8) array or empty array if file missing / unreadable."""
    try:
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data[np.newaxis, :]   # single row
        return data if data.shape[1] == 8 else np.empty((0, 8))
    except Exception:
        return np.empty((0, 8))


def update(_frame):
    data = load(PATH)
    if data.shape[0] < 2:
        return line_top, point_cur, line_y, line_spd

    ts  = data[:, 0]
    x, y, z = data[:, 1], data[:, 2], data[:, 3]

    # Top-down
    line_top.set_data(x, z)
    point_cur.set_data([x[-1]], [z[-1]])
    pad = max((x.max()-x.min()), (z.max()-z.min()), 0.5) * 0.1
    ax_top.set_xlim(x.min()-pad, x.max()+pad)
    ax_top.set_ylim(z.min()-pad, z.max()+pad)

    # Height
    line_y.set_data(ts, y)

    # Speed estimate from consecutive positions
    dpos = np.linalg.norm(np.diff(data[:, 1:4], axis=0), axis=1)
    dt   = np.diff(ts)
    dt[dt < 1e-9] = 1e-9
    spd  = dpos / dt
    line_spd.set_data(ts[1:], spd)

    ax_time.set_xlim(ts[0], ts[-1] + 0.1)
    ax_time.set_ylim(
        min(y.min(), spd.min()) - 0.1,
        max(y.max(), spd.max()) + 0.1
    )

    return line_top, point_cur, line_y, line_spd


ani = animation.FuncAnimation(fig, update, interval=200, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()
