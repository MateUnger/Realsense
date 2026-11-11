"""Interactive 3D keypoint viewer, taken from the "Draw keypoints" cell in test.ipynb.

Navigation:
    mouse wheel   next / previous frame
    right / left  next / previous frame
    up / down     30 frames forward / back

Usage (from the repo root):
    realsense_env\\Scripts\\python.exe draw_keypoints.py tug2
    realsense_env\\Scripts\\python.exe draw_keypoints.py tug2 --source qualisys
"""

import os
import argparse
import matplotlib.pyplot as plt
from Util.util import load_keypoints_and_labels

parser = argparse.ArgumentParser(description="Interactive 3D keypoint viewer")
parser.add_argument("recording_name", help="e.g. tug2")
parser.add_argument("--source", choices=["stereo", "qualisys"], default="stereo")
args = parser.parse_args()

recording_name = args.recording_name
keypoints, labels = load_keypoints_and_labels(
    filename=os.path.join("data", recording_name, f"{recording_name}_{args.source}.npz")
)

frame = 0

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")


def update():
    ax.clear()

    pts = keypoints[frame]

    # Plot all keypoints
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=50, c="red")

    ax.set_title(f"{recording_name} ({args.source})  Frame {frame + 1} / {len(keypoints)}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    x_min, x_max = (-1, 1)
    y_min, y_max = (-4, 8)
    z_min, z_max = (-0.1, 1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect([x_max - x_min, y_max - y_min, z_max - z_min])

    fig.canvas.draw_idle()


def on_key(event):
    global frame

    if event.key == "right":
        frame = min(frame + 1, len(keypoints) - 1)
        update()
    elif event.key == "left":
        frame = max(frame - 1, 0)
        update()
    elif event.key == "up":
        frame = min(frame + 30, len(keypoints) - 1)
        update()
    elif event.key == "down":
        frame = max(frame - 30, 0)
        update()


def on_scroll(event):
    global frame

    if event.button == "up":
        frame = min(frame + 1, len(keypoints) - 1)
        update()
    elif event.button == "down":
        frame = max(frame - 1, 0)
        update()


fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("scroll_event", on_scroll)

update()
plt.show()
