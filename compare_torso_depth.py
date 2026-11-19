"""Compare two ways of getting the depth of mid_hip / mid_shoulder against Qualisys:
    joint patch: the current _keypoints_3d.npy (small patch at each keypoint, sample_depth_patch)
    torso:       median depth over the torso (shoulder-hip quadrilateral shrunk toward its center)
Both go through the same calibration, filtering, resampling and axis mapping as preprocess_stereo.py.
The stereo and Qualisys recordings are aligned in time by cross-correlating the depth coordinate of
mid_hip, and in space by removing the median offset per axis.

Usage (from the repo root):
    realsense_env\\Scripts\\python.exe compare_torso_depth.py walk1
    realsense_env\\Scripts\\python.exe compare_torso_depth.py walk1 --torso-shrink 0.5 --save torso.png
"""

import os
import json
import argparse
import subprocess

import cv2
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from Util.util import (
    create_intrinsics_object,
    get_image_points,
    filter_keypoints_stereo,
    get_frame_times,
    resample_keypoints_to_rate,
    stereo_to_qualisys_axes,
    load_keypoints_and_labels,
)
from preprocess_stereo import LABELS, CALIBRATION_PATTERN, DATA_FOLDER, get_path

COMPARED_KEYPOINTS = ["mid_hip", "mid_shoulder"]
TORSO_CORNERS = ["left_shoulder", "right_shoulder", "right_hip", "left_hip"]  # polygon order
DEPTH_AXIS = 1  # (horizontal, depth, vertical) in the shared frame

parser = argparse.ArgumentParser(description="Torso depth sampling vs joint patch, against Qualisys")
parser.add_argument("recording_name", help="e.g. walk1")
parser.add_argument("--torso-shrink", type=float, default=0.6, help="torso polygon scaled toward its center by this factor")
parser.add_argument("--stereo-cut", type=int, default=650, help="stereo samples (100 Hz) dropped at the start, before the subject is tracked")
parser.add_argument("--save", default=None, help="save the figure to this path instead of showing it")
args = parser.parse_args()
recording_name = args.recording_name

keypoints_2d_portrait = np.load(get_path(recording_name, "_keypoints_2d.npy"))
keypoints_3d_joint_patch = np.load(get_path(recording_name, "_keypoints_3d.npy"))
with open(get_path(recording_name, "_metadata.json")) as f:
    metadata = json.load(f)
width, height = metadata["width"], metadata["height"]
n_frames = metadata["frames"]
depth_scale = metadata["depth_scale_meters"]
intrinsics = create_intrinsics_object(intrinsics_dict=metadata["color_intrinsics"])

# ---- torso depth for every frame ----
ffmpeg_command = ["ffmpeg", "-i", get_path(recording_name, "_depth.mkv"), "-f", "rawvideo", "-pix_fmt", "gray16le", "-"]
video_depth = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
bytes_per_frame = width * height * 2

torso_corner_indices = [LABELS.index(keypoint_name) for keypoint_name in TORSO_CORNERS]
compared_indices = [LABELS.index(keypoint_name) for keypoint_name in COMPARED_KEYPOINTS]
keypoints_3d_torso = np.full((n_frames, len(COMPARED_KEYPOINTS), 3), np.nan)
torso_pixel_count = np.zeros(n_frames, dtype=int)

for frame_number in tqdm(range(n_frames), desc="torso depth", unit="frame"):
    raw_frame_data = video_depth.stdout.read(bytes_per_frame)
    if len(raw_frame_data) < bytes_per_frame:
        raise RuntimeError(f"depth video ended at frame {frame_number}, expected {n_frames} frames")
    depth_map = np.frombuffer(raw_frame_data, dtype=np.uint16).reshape(height, width)

    # torso corners in landscape pixels, same mapping as keypoints_3d in preprocess_stereo.py
    corners_portrait = keypoints_2d_portrait[frame_number, torso_corner_indices]
    if np.isnan(corners_portrait).any():
        continue
    corners_landscape = np.column_stack([width - 1 - corners_portrait[:, 1], corners_portrait[:, 0]])
    torso_center = corners_landscape.mean(axis=0)
    corners_shrunk = torso_center + args.torso_shrink * (corners_landscape - torso_center)

    torso_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(torso_mask, [np.round(corners_shrunk).astype(np.int32)], 1)
    torso_depth_values = depth_map[(torso_mask == 1) & (depth_map > 0)]
    torso_pixel_count[frame_number] = torso_depth_values.size
    if torso_depth_values.size == 0:
        continue
    torso_depth = float(np.median(torso_depth_values)) * depth_scale

    # each compared keypoint keeps its own pixel, only its depth comes from the torso
    for compared_number, keypoint_index in enumerate(compared_indices):
        x_portrait, y_portrait = keypoints_2d_portrait[frame_number, keypoint_index]
        pixel_landscape = (width - 1 - y_portrait, x_portrait)
        keypoints_3d_torso[frame_number, compared_number] = rs.rs2_deproject_pixel_to_point(
            intrin=intrinsics, pixel=pixel_landscape, depth=torso_depth
        )

video_depth.stdout.close()
video_depth.wait()
print(f"torso pixels per frame: median {np.median(torso_pixel_count):.0f}, min {torso_pixel_count.min()}")

# ---- calibration, same as calibration() in preprocess_stereo.py ----
color_intrinsics = metadata["color_intrinsics"]
calibration_image = cv2.imread(os.path.join(DATA_FOLDER, recording_name, "calibration_image.png"), cv2.IMREAD_GRAYSCALE)
image_points, _ = get_image_points(calibration_image=calibration_image, calibration_pattern=CALIBRATION_PATTERN)
camera_matrix = np.array(
    [[color_intrinsics["fx"], 0, color_intrinsics["ppx"]], [0, color_intrinsics["fy"], color_intrinsics["ppy"]], [0, 0, 1]],
    dtype=np.float64,
)
_, rvec, t_vec = cv2.solvePnP(
    CALIBRATION_PATTERN.pattern_points, image_points, camera_matrix, np.array(color_intrinsics["coeffs"]), flags=cv2.SOLVEPNP_ITERATIVE
)
rotation_matrix, _ = cv2.Rodrigues(rvec)
camera_position_global_frame = (-rotation_matrix.T @ t_vec).ravel()
camera_position = stereo_to_qualisys_axes(camera_position_global_frame[None, None, :])[0, 0]

# ---- filter, resample and map axes, same as resample() in preprocess_stereo.py ----
frames, times, fps = get_frame_times(get_path(recording_name, "_timestamps_color.bin"), n_frames=n_frames)


def to_shared_frame(keypoints_3d_camera: np.ndarray) -> np.ndarray:
    keypoints_3d_global = keypoints_3d_camera @ rotation_matrix + camera_position_global_frame
    keypoints_filtered = filter_keypoints_stereo(keypoints_3d_global[frames], sampling_rate=fps, cutoffs=(3, 3, 5))
    keypoints_resampled, _ = resample_keypoints_to_rate(keypoints_filtered, times, 100)
    return stereo_to_qualisys_axes(keypoints_resampled)[args.stereo_cut:]


stereo_versions = {
    "joint patch": to_shared_frame(keypoints_3d_joint_patch[:, compared_indices]),
    "torso": to_shared_frame(keypoints_3d_torso),
}

qualisys_keypoints, qualisys_labels = load_keypoints_and_labels(filename=get_path(recording_name, "_qualisys.npz"))
qualisys_keypoints = qualisys_keypoints[:, [qualisys_labels.index(keypoint_name) for keypoint_name in COMPARED_KEYPOINTS]]

# ---- time alignment: cross-correlate the mid_hip depth coordinate (torso version) ----
def fill_nan(signal: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(signal)
    return np.interp(np.arange(len(signal)), np.flatnonzero(valid), signal[valid])


stereo_signal = fill_nan(stereo_versions["torso"][:, 0, DEPTH_AXIS])
qualisys_signal = fill_nan(qualisys_keypoints[:, 0, DEPTH_AXIS])
correlation = np.correlate(qualisys_signal - qualisys_signal.mean(), stereo_signal - stereo_signal.mean(), mode="full")
lag = int(np.argmax(correlation)) - (len(stereo_signal) - 1)  # qualisys sample = stereo sample + lag
stereo_start = max(0, -lag)
overlap_length = min(len(stereo_signal) - stereo_start, len(qualisys_signal) - (stereo_start + lag))
stereo_slice = slice(stereo_start, stereo_start + overlap_length)
qualisys_slice = slice(stereo_start + lag, stereo_start + lag + overlap_length)
print(f"time offset: qualisys = stereo + {lag} samples ({lag / 100:.2f} s), overlap {overlap_length / 100:.1f} s")

# drift check: best lag in each quarter of the overlap, around the global lag
quarter_length = overlap_length // 4
for quarter_number in range(4):
    quarter_start = stereo_start + quarter_number * quarter_length
    stereo_quarter = stereo_signal[quarter_start:quarter_start + quarter_length]
    lag_errors = []
    for lag_change in range(-100, 101):
        qualisys_start = quarter_start + lag + lag_change
        qualisys_quarter = qualisys_signal[qualisys_start:qualisys_start + quarter_length]
        if qualisys_start < 0 or len(qualisys_quarter) < quarter_length:
            lag_errors.append(np.inf)
            continue
        difference = stereo_quarter - qualisys_quarter
        lag_errors.append(np.std(difference))
    print(f"    quarter {quarter_number + 1}: lag change {np.argmin(lag_errors) - 100:+d} samples")

qualisys_aligned = qualisys_keypoints[qualisys_slice]
distance_from_camera = np.linalg.norm(stereo_versions["torso"][stereo_slice, 0] - camera_position, axis=1)
walking_away = np.gradient(fill_nan(distance_from_camera)) > 0
distance_bins = np.arange(np.floor(np.nanmin(distance_from_camera)), np.ceil(np.nanmax(distance_from_camera)) + 1, 1.0)

# ---- errors after removing the median offset per axis ----
errors = {}
for version_name, stereo_keypoints in stereo_versions.items():
    difference = stereo_keypoints[stereo_slice] - qualisys_aligned
    median_offset = np.nanmedian(difference, axis=0)
    errors[version_name] = difference - median_offset
    print(f"\n=== {version_name} ===")
    for compared_number, keypoint_name in enumerate(COMPARED_KEYPOINTS):
        keypoint_errors = errors[version_name][:, compared_number]
        rms_per_axis = np.sqrt(np.nanmean(keypoint_errors**2, axis=0)) * 100
        offset_text = ", ".join(f"{value * 100:.1f}" for value in median_offset[compared_number])
        print(
            f"{keypoint_name}: RMS error [cm] horizontal {rms_per_axis[0]:.1f}, depth {rms_per_axis[1]:.1f}, "
            f"vertical {rms_per_axis[2]:.1f} (removed offset [cm]: {offset_text})"
        )
        depth_errors = keypoint_errors[:, DEPTH_AXIS]
        print(
            f"    depth error mean [cm]: walking away {np.nanmean(depth_errors[walking_away]) * 100:+.1f}, "
            f"walking toward {np.nanmean(depth_errors[~walking_away]) * 100:+.1f}"
        )
        bin_texts = []
        for bin_start, bin_end in zip(distance_bins[:-1], distance_bins[1:]):
            in_bin = (distance_from_camera >= bin_start) & (distance_from_camera < bin_end)
            if np.sum(in_bin & ~np.isnan(depth_errors)) > 50:
                bin_texts.append(f"{bin_start:.0f}-{bin_end:.0f} m: {np.sqrt(np.nanmean(depth_errors[in_bin] ** 2)) * 100:.1f}")
        print("    depth RMS by distance [cm]: " + ", ".join(bin_texts))

# ---- plots: depth coordinate over time, depth error vs distance ----
time_axis = np.arange(overlap_length) / 100
fig, axes = plt.subplots(len(COMPARED_KEYPOINTS), 2, figsize=(14, 4 * len(COMPARED_KEYPOINTS)), squeeze=False)
for compared_number, keypoint_name in enumerate(COMPARED_KEYPOINTS):
    axis_time, axis_error = axes[compared_number]
    axis_time.plot(time_axis, qualisys_aligned[:, compared_number, DEPTH_AXIS], color="black", label="qualisys")
    for version_name, stereo_keypoints in stereo_versions.items():
        # shown with its median offset removed, so the curves overlap
        shifted = qualisys_aligned[:, compared_number, DEPTH_AXIS] + errors[version_name][:, compared_number, DEPTH_AXIS]
        axis_time.plot(time_axis, shifted, label=version_name, alpha=0.8)
        axis_error.scatter(distance_from_camera, errors[version_name][:, compared_number, DEPTH_AXIS] * 100, s=2, label=version_name, alpha=0.5)
    axis_time.set_title(f"{keypoint_name}: depth coordinate")
    axis_time.set_xlabel("time [s]")
    axis_time.set_ylabel("depth [m]")
    axis_time.legend(loc="upper right")
    axis_error.axhline(0, color="black", linewidth=0.8)
    axis_error.set_title(f"{keypoint_name}: depth error vs Qualisys")
    axis_error.set_xlabel("distance from camera [m]")
    axis_error.set_ylabel("error [cm]")
    axis_error.legend(loc="upper left", markerscale=4)
fig.tight_layout()
if args.save:
    fig.savefig(args.save, dpi=80)
else:
    plt.show()
