"""Show the depth patch sampled for the two keypoints of a bone, in the frame where the bone length is
furthest from its median, among the frames where the bone is far from the camera.

Works on the raw, unfiltered _keypoints_3d.npy (camera frame, original frame numbers), so spikes are
seen before the Hampel / Butterworth filtering in the resample stage.

Usage (from the repo root):
    realsense_env\\Scripts\\python.exe check_depth_patch.py walk1
    realsense_env\\Scripts\\python.exe check_depth_patch.py walk1 --bone right_forearm --frame 1799
"""

import json
import argparse
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from Util.util import sample_depth_patch, get_segment_vectors
from preprocess_stereo import LABELS, get_path

BONES = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "left_shank": ("left_knee", "left_ankle"),
    "right_thigh": ("right_hip", "right_knee"),
    "right_shank": ("right_knee", "right_ankle"),
    "shoulder_width": ("left_shoulder", "right_shoulder"),
    "hip_width": ("left_hip", "right_hip"),
}

parser = argparse.ArgumentParser(description="Show the depth patches of a bone's keypoints")
parser.add_argument("recording_name", help="e.g. walk1")
parser.add_argument("--bone", default="left_shank", choices=BONES.keys())
parser.add_argument("--frame", type=int, default=None, help="frame to show (default: worst far frame)")
parser.add_argument("--far-percentile", type=float, default=75, help="only frames farther than this percentile of the bone's distance are searched")
parser.add_argument("--patch-radius", type=int, default=3, help="same as in keypoints_3d (preprocess_stereo.py)")
parser.add_argument("--context-radius", type=int, default=30, help="pixels shown around the keypoint")
args = parser.parse_args()

recording_name = args.recording_name
patch_radius = args.patch_radius
context_radius = args.context_radius

keypoints_3d_camera = np.load(get_path(recording_name, "_keypoints_3d.npy"))
keypoints_2d_portrait = np.load(get_path(recording_name, "_keypoints_2d.npy"))
with open(get_path(recording_name, "_metadata.json")) as f:
    metadata = json.load(f)
width, height = metadata["width"], metadata["height"]

# raw bone length, relative to its median over the whole recording; the calibration is a rigid
# transform, so camera-frame lengths are the same as world-frame lengths
start_name, end_name = BONES[args.bone]
bone_lengths_raw = np.linalg.norm(get_segment_vectors(keypoints_3d_camera, LABELS, start_name, end_name), axis=1)
relative_change_raw = (bone_lengths_raw / np.nanmedian(bone_lengths_raw) - 1) * 100

# distance from the camera = camera-frame z (index 2)
bone_camera_distance = np.mean(
    [keypoints_3d_camera[:, LABELS.index(keypoint_name), 2] for keypoint_name in (start_name, end_name)], axis=0
)
if args.frame is None:
    far_frames = bone_camera_distance > np.nanpercentile(bone_camera_distance, args.far_percentile)
    frame_number = int(np.nanargmax(np.where(far_frames, np.abs(relative_change_raw), np.nan)))
else:
    frame_number = args.frame
print(
    f"frame {frame_number}: {args.bone} {relative_change_raw[frame_number]:+.1f}%, "
    f"distance {bone_camera_distance[frame_number]:.2f} m"
)

# read that one depth frame
ffmpeg_command = [
    "ffmpeg", "-loglevel", "error", "-i", get_path(recording_name, "_depth.mkv"),
    "-vf", f"select=eq(n\\,{frame_number})", "-vframes", "1",
    "-f", "rawvideo", "-pix_fmt", "gray16le", "-",
]
raw_depth = subprocess.run(ffmpeg_command, capture_output=True, check=True).stdout
depth_map = np.frombuffer(raw_depth, dtype=np.uint16).reshape(height, width) * metadata["depth_scale_meters"]
depth_map_display = np.where(depth_map > 0, depth_map, np.nan)  # 0 = no depth reading, drawn white

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle(
    f"{recording_name} frame {frame_number}: {args.bone} {relative_change_raw[frame_number]:+.1f}%, "
    f"distance {bone_camera_distance[frame_number]:.2f} m"
)
for row_axes, keypoint_name in zip(axes, (start_name, end_name)):
    x_portrait, y_portrait = keypoints_2d_portrait[frame_number, LABELS.index(keypoint_name)]
    x_landscape = int(round(width - 1 - y_portrait))  # same mapping as keypoints_3d
    y_landscape = int(round(x_portrait))
    keypoint_depth = keypoints_3d_camera[frame_number, LABELS.index(keypoint_name), 2]

    # context crop, drawn in full-image pixel coordinates
    x_min, x_max = max(0, x_landscape - context_radius), min(width, x_landscape + context_radius + 1)
    y_min, y_max = max(0, y_landscape - context_radius), min(height, y_landscape + context_radius + 1)
    context = depth_map_display[y_min:y_max, x_min:x_max]
    color_limits = np.nanpercentile(context, [2, 98])
    image = row_axes[0].imshow(
        context,
        extent=(x_min - 0.5, x_max - 0.5, y_max - 0.5, y_min - 0.5),
        vmin=color_limits[0],
        vmax=color_limits[1],
        cmap="viridis",
    )
    row_axes[0].add_patch(
        plt.Rectangle(
            (x_landscape - patch_radius - 0.5, y_landscape - patch_radius - 0.5),
            2 * patch_radius + 1,
            2 * patch_radius + 1,
            fill=False,
            edgecolor="red",
        )
    )
    row_axes[0].plot(x_landscape, y_landscape, "r+")
    row_axes[0].set_title(f"{keypoint_name}: depth {keypoint_depth:.2f} m\n(landscape, camera rotated 90°)")
    fig.colorbar(image, ax=row_axes[0], label="depth [m]")

    # the patch itself, with every value written in; pixels without depth are marked with "-"
    patch = depth_map_display[
        max(0, y_landscape - patch_radius):y_landscape + patch_radius + 1,
        max(0, x_landscape - patch_radius):x_landscape + patch_radius + 1,
    ]
    row_axes[1].imshow(patch, vmin=color_limits[0], vmax=color_limits[1], cmap="viridis")
    for (patch_row, patch_column), depth_value in np.ndenumerate(patch):
        if np.isnan(depth_value):
            row_axes[1].text(patch_column, patch_row, "-", ha="center", va="center", fontsize=8, color="gray")
        else:
            row_axes[1].text(patch_column, patch_row, f"{depth_value:.2f}", ha="center", va="center", fontsize=8, color="white")
    patch_sampled = sample_depth_patch(depth_map, x_landscape, y_landscape, radius=patch_radius)
    patch_median = np.nanmedian(patch) if np.any(~np.isnan(patch)) else np.nan
    row_axes[1].set_title(f"patch: sampled {patch_sampled:.2f} m, median {patch_median:.2f} m")

fig.tight_layout()
plt.show()
