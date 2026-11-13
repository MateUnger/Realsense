import os
import time
import json
import cv2
import numpy as np
import importlib
from typing import Literal
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import pandas as pd
import csv
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt, resample_poly, find_peaks
from scipy.ndimage import median_filter
from fractions import Fraction

rtmlib_module = importlib.import_module("rtmlib")


# =========================================== RECORD, ALIGN ===========================================

# resolution-fps pairs that work with both color and depth
resolution_presets = {
    "848x480@60": {"width": 848, "height": 480, "fps": 60},
    "640x360@90": {"width": 640, "height": 360, "fps": 90},
    "480x270@90": {"width": 480, "height": 270, "fps": 90},
}


def create_ffmpeg_command(
    width: int,
    height: int,
    fps: float,
    encoder: Literal["h264_qsv", "h264_nvenc", "h264_amf", "libx264"],
    out_filename: str,
):
    """
    Create ffmpeg cmd command for video encoding

    Args:
        width: width of frame
        height: height of frame
        fps: sampling frequency of video
        encoder: encoder to use ("h264_qsv"-Intel GPU, "h264_nvenc"-NVIDIA GPU, "h264_amf"-AMD GPU, "libx264"-CPU)

    Returns:
        ffmpeg: cmd command for ffmpeg
    """
    encoders = {
        "h264_qsv": [
            "-preset",
            "veryfast",
            "-global_quality",
            "23",
        ],
        "h264_nvenc": [
            "-preset",
            "p1",
            "-cq",
            "23",
        ],
        "h264_amf": [
            "-quality",
            "speed",
        ],
        "libx264": [
            "-preset",
            "ultrafast",
            "-crf",
            "23",
        ],
    }
    # ffmpeg cmd command
    ffmpeg = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        encoder,
    ]
    ffmpeg += encoders[encoder]
    ffmpeg += ["-pix_fmt", "yuv420p", out_filename]

    return ffmpeg


def create_intrinsics_object(intrinsics_dict: dict) -> rs.intrinsics:
    intrinsics_object = rs.intrinsics()

    intrinsics_object.width = intrinsics_dict["width"]
    intrinsics_object.height = intrinsics_dict["height"]
    intrinsics_object.fx = intrinsics_dict["fx"]
    intrinsics_object.fy = intrinsics_dict["fy"]
    intrinsics_object.ppx = intrinsics_dict["ppx"]
    intrinsics_object.ppy = intrinsics_dict["ppy"]
    intrinsics_object.model = rs.distortion.inverse_brown_conrady
    intrinsics_object.coeffs = intrinsics_dict["coeffs"]

    return intrinsics_object


# =========================================== POSE ESTIMATION ===========================================
labels_26 = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "top_head",
    "neck",
    "center_hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
    "mid_shoulder",
    "mid_hip",
    "center_of_mass",
]


labels_17 = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "mid_shoulder",
    "mid_hip",
    "center_of_mass",
]


def hex_to_bgr(hex_color: str) -> tuple:
    """
    Convert hex color to BGR format (for cv2)

    Args:
        hex_color: color in hex format

    Returns:
        bgr_color: same color in bgr format
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    bgr_color = (b, g, r)
    return bgr_color


class Custom:

    def __init__(
        self,
        det_class: str = None,
        det: str = None,
        det_input_size: tuple = (640, 640),
        pose_class: str = None,
        pose: str = None,
        pose_input_size: tuple = (192, 256),
        mode: str = None,
        to_openpose: bool = False,
        backend: str = "onnxruntime",
        device: str = "cuda",
    ):

        if det_class is not None:
            try:
                det_class = getattr(rtmlib_module, det_class)
                self.det_model = det_class(
                    det, model_input_size=det_input_size, backend=backend, device=device
                )
                self.one_stage = False

            except ImportError:
                raise ImportError(f"{det_class} is not supported by rtmlib.")
        else:
            self.one_stage = True

        if pose_class is not None:
            try:
                pose_class = getattr(rtmlib_module, pose_class)
                self.pose_model = pose_class(
                    pose,
                    model_input_size=pose_input_size,
                    to_openpose=to_openpose,
                    backend=backend,
                    device=device,
                )
            except ImportError:
                raise ImportError(f"{pose_class} is not supported by rtmlib.")

    MODE = {
        "performance": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",  # noqa
            "det_input_size": (640, 640),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",  # noqa
            "pose_input_size": (288, 384),
        },
        "lightweight": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",  # noqa
            "det_input_size": (416, 416),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",  # noqa
            "pose_input_size": (192, 256),
        },
        "balanced": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",  # noqa
            "det_input_size": (640, 640),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",  # noqa
            "pose_input_size": (192, 256),
        },
    }

    RTMO_MODE = {
        "performance": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip",  # noqa
            "pose_input_size": (640, 640),
        },
        "lightweight": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip",  # noqa
            "pose_input_size": (640, 640),
        },
        "balanced": {
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",  # noqa
            "pose_input_size": (640, 640),
        },
    }

    def __call__(self, image: np.ndarray):

        if self.one_stage:
            t0 = time.perf_counter()

            keypoints, scores = self.pose_model(image)

            t1 = time.perf_counter()

            # print(f"RTMO: {(t1 - t0) * 1000:.2f} ms")

        else:
            t0 = time.perf_counter()

            bboxes = self.det_model(image)

            t1 = time.perf_counter()
            # print("bboxes:", len(bboxes))
            keypoints, scores = self.pose_model(image, bboxes=bboxes)

            t2 = time.perf_counter()

            # print(
            #     f"DET: {(t1 - t0) * 1000:.2f} ms | "
            #     f"POSE: {(t2 - t1) * 1000:.2f} ms | "
            #     f"TOTAL: {(t2 - t0) * 1000:.2f} ms"
            # )

        return keypoints, scores


models = {
    "detectors": {
        "yolo_nano": {
            "input_size": [416, 416],
            "path": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip",
        },
        "yolo_m": {
            "input_size": [640, 640],
            "path": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        },
        "yolo_l": {
            "input_size": [640, 640],
            "path": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_l_8xb8-300e_humanart-ce1d7a62.zip",
        },
        "yolo_x": {
            "input_size": [640, 640],
            "path": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        },
    },
    "pose": {
        "rtm_pose_t_17": {
            "input_size": [192, 256],
            "path": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip",
        },
        "rtm_pose_m_17": {
            "input_size": [192, 256],
            "path": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        },
        "rtm_pose_l_17": {
            "input_size": [288, 384],
            "path": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip",
        },
        "rtm_pose_x_17": {
            "input_size": [288, 384],
            "path": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
        },
        "vit_pose_l_17": {
            "input_size": [192, 256],
            "path": "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-l-coco.onnx",
        },
        "rtm_pose_t_26": {
            "input_size": [192, 256],
            "path": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.zip",
        },
        "rtm_pose_m_26": {
            "input_size": [288, 384],
            "path": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
        },
        "rtm_pose_l_26": {
            "input_size": [288, 384],
            "path": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip",
        },
        "rtm_pose_x_26": {
            "input_size": [288, 384],
            "path": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
        },
    },
}

# =========================================== 3D COORDINATES ===========================================


def sample_depth_patch(depth_map: np.ndarray, x: int, y: int, radius: int = 1) -> float:
    """
    Samples the depth map using the median value of a patch (size radius+2).

    Args:
        depth_map: array of depth values in landscape orientation
        x,y: center coords of sampling patch
        radius: size of patch
    Returns:
        depth: median depth value of sampled patch (np.nan if no valid values)
    """
    h, w = depth_map.shape
    x_min, x_max = max(0, x - radius), min(w, x + radius + 1)
    y_min, y_max = max(0, y - radius), min(h, y + radius + 1)

    ROI = depth_map[y_min:y_max, x_min:x_max]
    valid_ROI = ROI[ROI > 0]

    depth = float(np.nanmedian(valid_ROI)) if valid_ROI.size > 0 else np.nan

    return depth


def interpolate_gaps(keypoints: np.ndarray, fps: float, max_gap: float) -> np.ndarray:
    """
    Fill interior NaN gaps of at most `max_gap` seconds, independently per
    keypoint and dimension. Leading/trailing gaps are left untouched.

    Args:
        keypoints: shape (n_frames, n_keypoints, n_dims).
        fps: sampling rate in Hz
        max_gap: maximum gap length in seconds
    """
    if keypoints.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {keypoints.shape}")

    n_frames, n_keypoints, n_dims = keypoints.shape
    max_gap_frames = int(round(max_gap * fps))
    keypoints_filled = keypoints.astype(float, copy=True)

    for kpt_idx in range(n_keypoints):
        for dim_idx in range(n_dims):
            coordinate = keypoints[:, kpt_idx, dim_idx]
            is_nan = np.isnan(coordinate)
            valid = np.flatnonzero(~is_nan)
            if valid.size < 2 or valid.size == n_frames:
                continue

            # Run-length encode NaN runs: starts inclusive, ends exclusive
            padded = np.concatenate(([False], is_nan, [False])).astype(np.int8)
            edges = np.flatnonzero(np.diff(padded))
            starts, ends = edges[::2], edges[1::2]

            spline = None
            for start, end in zip(starts, ends):
                if start == 0 or end == n_frames:  # edge gap: no anchor on one side
                    continue
                if end - start > max_gap_frames:  # too long
                    continue
                if spline is None:  # fit once per signal
                    spline = CubicSpline(valid, coordinate[valid], extrapolate=False)
                keypoints_filled[start:end, kpt_idx, dim_idx] = spline(np.arange(start, end))

    return keypoints_filled


def get_segment_borders(valid: np.ndarray):
    """
    Return starts, ends of contiguous valid (True) segments (end exclusive).
    """
    diff = np.diff(np.concatenate(([0], valid.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)))


def fill_short_runs(binary_signal: np.ndarray, value: int, min_length: int) -> np.ndarray:
    """
    Flip runs of `value` that are shorter than `min_length` samples to the other value.
    Runs touching the start or end of the signal are kept, since their real length is unknown.

    Args:
        binary_signal: array of 0s and 1s
        value: which runs to check (0 or 1)
        min_length: minimum run length in samples
    Returns:
        copy of the signal with the short runs flipped
    """
    filled_signal = binary_signal.copy()
    change_indices = np.flatnonzero(np.diff(binary_signal)) + 1
    run_starts = np.concatenate([[0], change_indices])
    run_ends = np.concatenate([change_indices, [len(binary_signal)]])
    for run_start, run_end in zip(run_starts, run_ends):
        is_interior_run = run_start > 0 and run_end < len(binary_signal)
        if (
            binary_signal[run_start] == value
            and is_interior_run
            and run_end - run_start < min_length
        ):
            filled_signal[run_start:run_end] = 1 - value
    return filled_signal


def filter_keypoints_median_butter(
    keypoints: np.ndarray,
    sampling_rate: float,
    cutoff: float,
    order: int,
    gap_size: float,
    median_window: float | None = None,
) -> np.ndarray:
    """
    Fills short gaps, optionally applies a median filter, then applies a
    zero-phase Butterworth filter to each contiguous segment of valid samples.
    Remaining NaNs (long gaps) are kept.

    Args:
        keypoints: shape (n_frames, n_kpt, n_dims)
        sampling_rate: sampling rate in Hz
        cutoff: cutoff frequency in Hz
        order: Butterworth order (effective order is doubled by filtfilt)
        gap_size: maximum gap (seconds) to interpolate
        median_window: median filter window in seconds, applied before the
            Butterworth filter to remove spikes. None disables it.
    """

    if cutoff >= sampling_rate / 2:
        raise ValueError("cutoff must be below the Nyquist frequency")

    sos = butter(order, cutoff, btype="lowpass", fs=sampling_rate, output="sos")
    min_length = 3 * (2 * sos.shape[0] + 1) + 1  # safe lower bound for sosfiltfilt padding

    kernel_size = None
    if median_window is not None:
        if median_window <= 0:
            raise ValueError("median_window must be positive")
        kernel_size = max(3, int(round(median_window * sampling_rate)))
        if kernel_size % 2 == 0:
            kernel_size += 1  # median filter needs an odd kernel
        min_length = max(min_length, kernel_size)

    keypoints = interpolate_gaps(keypoints, sampling_rate, gap_size)
    keypoints_filtered = np.array(keypoints, dtype=float, copy=True)  # never touch the input

    for kpt, dim in np.ndindex(keypoints.shape[1], keypoints.shape[2]):
        coordinate = keypoints[:, kpt, dim]
        for start, end in get_segment_borders(~np.isnan(coordinate)):
            if end - start >= min_length:
                segment = coordinate[start:end]
                if kernel_size is not None:
                    # mode="reflect" avoids the zero-padding bias of scipy.signal.medfilt
                    segment = median_filter(segment, size=kernel_size, mode="reflect")
                keypoints_filtered[start:end, kpt, dim] = sosfiltfilt(sos, segment)
            else:
                keypoints_filtered[start:end, kpt, dim] = np.nan  # too short for filtering
    return keypoints_filtered


def reject_outliers_hampel(
    keypoints: np.ndarray,
    sampling_rate: float,
    window: float = 0.2,
    n_sigma: float = 3,
    min_deviation: float = 0.02,
) -> np.ndarray:
    """
    Hampel filter: replace samples that deviate too far from the rolling median with NaN,
    independently per keypoint and dimension. Removes single-frame depth spikes that a
    lowpass filter would only smear out.

    Args:
        keypoints: shape (n_frames, n_kpt, n_dims)
        sampling_rate: sampling rate in Hz
        window: rolling window in seconds
        n_sigma: threshold in robust standard deviations (1.4826 * MAD)
        min_deviation: threshold never drops below this (same units as keypoints, m)
    Returns:
        keypoints with outliers set to NaN
    """
    n_frames = keypoints.shape[0]
    window_frames = int(round(window * sampling_rate)) | 1  # odd, centered
    df = pd.DataFrame(keypoints.reshape(n_frames, -1))
    median = df.rolling(window_frames, center=True, min_periods=3).median()
    mad = (df - median).abs().rolling(window_frames, center=True, min_periods=3).median() * 1.4826
    threshold = np.maximum(n_sigma * mad.to_numpy(), min_deviation)

    cleaned = keypoints.reshape(n_frames, -1).astype(float, copy=True)
    cleaned[np.abs(cleaned - median.to_numpy()) > threshold] = np.nan
    return cleaned.reshape(keypoints.shape)


def filter_keypoints_stereo(
    keypoints: np.ndarray,
    sampling_rate: float,
    cutoffs: tuple[float, float, float] = (3, 3, 5),
    order: int = 4,
    gap_size: float = 0.25,
    hampel_window: float = 0.2,
) -> np.ndarray:
    """
    Filter stereo keypoints for velocity estimates: Hampel outlier rejection, then a zero-phase
    Butterworth with a separate cutoff per axis.

    Args:
        keypoints: shape (n_frames, n_kpt, 3), in the stereo global frame (horizontal, depth, vertical)
        sampling_rate: actual sampling rate in Hz (from the timestamps, not the nominal fps)
        cutoffs: cutoff frequency in Hz per axis (horizontal, depth, vertical)
        order: Butterworth order (effective order is doubled by filtfilt)
        gap_size: maximum gap (seconds) to interpolate, including removed outliers
        hampel_window: Hampel window in seconds
    """
    keypoints = reject_outliers_hampel(keypoints, sampling_rate, window=hampel_window)

    keypoints_filtered = np.empty(keypoints.shape, dtype=float)
    for dim, cutoff in enumerate(cutoffs):
        keypoints_filtered[:, :, [dim]] = filter_keypoints_median_butter(
            keypoints[:, :, [dim]], sampling_rate, cutoff, order, gap_size
        )
    return keypoints_filtered


def get_frame_times(
    filename: str, n_frames: int | None = None, startup: float = 2.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Clean the frame timestamps written during conversion (_timestamps_color.bin, float64 ms) and
    estimate the actual frame rate (the camera runs at 59.755 fps in the 60 fps mode and
    89.868 fps in the 90 fps mode; using the nominal rate drifts ~0.25 s/min against qualisys).

    Drops the frames up to a startup dropout (gap > 0.1 s in the first `startup` seconds),
    duplicated frames (same frame delivered twice) and frames whose timestamp goes backwards.
    Each kept frame is placed on a uniform grid at the actual frame rate, which removes the
    timestamp jitter and keeps dropped frames as gaps in the times.

    Args:
        filename: path to the _timestamps_color.bin file (_timestamps.bin for older conversions)
        n_frames: number of keypoint frames, timestamps beyond it are ignored
        startup: time window (s) at the start in which a dropout is searched for
    Returns:
        frames: indices of the frames to keep, use as keypoints[frames]
        times: time of each kept frame in seconds, starting at 0
        fps: actual frame rate in Hz
    """
    timestamps = np.fromfile(filename, dtype=np.float64) / 1000
    if n_frames is not None:
        timestamps = timestamps[:n_frames]
    intervals = np.diff(timestamps)
    median_interval = np.median(intervals)
    normal = (intervals > 0.8 * median_interval) & (intervals < 1.2 * median_interval)
    fps = 1 / intervals[normal].mean()

    # drop everything up to (and including) the last dropout in the first seconds
    dropouts = np.flatnonzero((intervals > 0.1) & (np.arange(len(intervals)) < startup * fps))
    first = dropouts[-1] + 1 if dropouts.size else 0

    # slot of each frame on the frame grid; keep a frame only if it is later than all before it
    slots = np.round((timestamps[first:] - timestamps[first]) * fps).astype(int)
    keep = np.concatenate(([True], slots[1:] > np.maximum.accumulate(slots)[:-1]))

    frames = first + np.flatnonzero(keep)
    times = slots[keep] / fps
    return frames, times, fps


def resample_keypoints_to_rate(
    keypoints: np.ndarray, times: np.ndarray, target_fps: float, max_gap: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample keypoints with known frame times onto a uniform grid by cubic interpolation, per
    keypoint and dimension. Unlike resample_keypoints this works for any rate ratio
    (e.g. 59.755 -> 100 Hz) and accounts for dropped frames. Grid points inside NaN gaps
    longer than max_gap stay NaN.

    Args:
        keypoints: shape (n_frames, n_kpt, n_dims)
        times: time of each frame in seconds (e.g. from get_frame_times)
        target_fps: sampling rate of the output in Hz
        max_gap: longest gap (s) between valid samples that is interpolated over
    Returns:
        resampled: shape (n_samples, n_kpt, n_dims)
        grid: time of each output sample in seconds
    """
    grid = np.arange(times[0], times[-1], 1 / target_fps)
    resampled = np.full((len(grid),) + keypoints.shape[1:], np.nan)

    for kpt, dim in np.ndindex(keypoints.shape[1], keypoints.shape[2]):
        coordinate = keypoints[:, kpt, dim]
        valid = ~np.isnan(coordinate)
        if valid.sum() < 4:
            continue
        valid_times = times[valid]
        values = CubicSpline(valid_times, coordinate[valid], extrapolate=False)(grid)

        # blank grid points whose neighbouring valid samples are too far apart
        pos = np.clip(np.searchsorted(valid_times, grid), 1, len(valid_times) - 1)
        values[valid_times[pos] - valid_times[pos - 1] > max_gap] = np.nan
        resampled[:, kpt, dim] = values

    return resampled, grid


def stereo_to_qualisys_axes(
    keypoints: np.ndarray,
    axis_order: tuple[int, int, int] = (0, 1, 2),
    axis_signs: tuple[int, int, int] = (-1, 1, -1),
) -> np.ndarray:
    """
    Rearrange stereo keypoints from the stereo global frame (chessboard frame) to the qualisys
    axis order and flip axes so that they point the same way. The chessboard frame always points
    the vertical axis down, but its horizontal axes depend on where the camera and chessboard are
    placed, so the mapping is set per recording setup:
        pq1 (default): order (1, 0, 2), signs (1, 1, -1), swap the two horizontal axes, flip vertical
        tug1: order (0, 1, 2), signs (-1, 1, -1), no swap, flip the first horizontal axis and vertical
    The mapping has to be a proper rotation (no mirroring): a swap of two axes needs an odd number
    of flips, no swap needs an even number.

    Args:
        keypoints: shape (n_frames, n_kpt, 3), in the stereo global frame
        axis_order: stereo axis used for each qualisys axis
        axis_signs: sign applied to each qualisys axis after reordering
    Returns:
        keypoints in the qualisys axis order, shape (n_frames, n_kpt, 3)
    """
    keypoints_qualisys = keypoints[..., list(axis_order)].astype(float, copy=True)
    keypoints_qualisys *= np.array(axis_signs, dtype=float)
    return keypoints_qualisys


def get_velocity_and_acceleration(keypoint: np.ndarray, fps: float):
    delta_t = 1 / fps  # frames/sec
    position = keypoint  # point, meters
    velocity = np.gradient(position, delta_t, axis=0)  # vector, m/s
    velocity_magnitude = np.linalg.norm(velocity, axis=1)  # scalar
    acceleration_magnitude = np.gradient(velocity_magnitude, delta_t, axis=0)  # scalar, m/s^2

    return velocity_magnitude, acceleration_magnitude


def get_velocity_and_acceleration_1D(keypoint: np.ndarray, fps: float):
    delta_t = 1 / fps
    position = keypoint
    velocity = np.gradient(position, delta_t, axis=0)  # scalar
    acceleration = np.gradient(velocity, delta_t, axis=0)  # scalar

    return velocity, acceleration


def get_segment_vectors(
    keypoints: np.ndarray, labels: list[str], start_kpt_name: str, end_kpt_name: str
) -> np.ndarray:
    """
    Vector from one keypoint to another for every frame, e.g. right_shoulder -> left_shoulder.

    Args:
        keypoints: shape (n_frames, n_kpt, n_dims)
        labels: keypoint labels
        start_kpt_name: label of the keypoint the vector starts at
        end_kpt_name: label of the keypoint the vector points to
    Returns:
        segment vectors, shape (n_frames, n_dims)
    """
    return (
        keypoints[:, labels.index(end_kpt_name), :] - keypoints[:, labels.index(start_kpt_name), :]
    )


def normalize_vector(vector: np.ndarray):
    """
    Returns the unit vector of the input vector.

    Args:
        vector: input vector

    Returns:
        normalized_vector: unit vector of magnitude 1 with the same direction as the input
    """
    return vector / np.linalg.norm(vector)


def angle_between_vectors(vector_1: np.ndarray, vector_2: np.ndarray):
    """
    Returns the angle in degrees between vectors 'vector_1' and 'vector_2'

    Args:
        vector_1: input vector
        vector_2: other input vector
    Returns:
        angle: angle between the two input vectors in degrees.

    """
    # calculate unit vectors
    vector_1_unit = normalize_vector(vector_1)
    vector_2_unit = normalize_vector(vector_2)

    # get scalar product
    scalar_product = np.clip(np.dot(vector_1_unit, vector_2_unit), -1, 1)

    # clip scalar product to [-1,1] range so trig. function works normal,
    # use arccos to get the angle from scalar product
    angle = np.degrees(np.arccos(scalar_product))

    return angle


def project_vector_on_plane(plane_normal_vector: np.ndarray, vector: np.ndarray):
    """
    Project an n-dimensional vector onto an n-dimensional plane defined by its normal (orthogonal) vector.

    Args:
        plane_normal_vector: vector orthogonal to the reference plane
        vector: vector to be projected
    Returns:
        vector_projection: projection of the input vector onto reference plane
    """
    # normalize surface normal
    normal_vector = plane_normal_vector / np.linalg.norm(plane_normal_vector)
    # multily vector with its scalar product (dot product w surface normal), shift it
    vector_projection = vector - np.dot(vector, normal_vector) * normal_vector
    return vector_projection


def get_projection(plane_normal_vectors: np.ndarray, vector_array: np.ndarray):
    """
    Project each member of an array of vectors onto plane.

    Args:
        plane_normal_vectors: vectors orthogonal to the reference plane, one per frame, shape (n_frames, n_dims)
        vector_array: array of vectors to be projected, shape (n_frames, n_dims). Usually a limb for 1 gait-cycle

    Returns:
        array of projected vectors, shape (n_frames, n_dims)
    """

    return np.array(
        [
            project_vector_on_plane(plane_normal, vector)
            for plane_normal, vector in zip(plane_normal_vectors, vector_array)
        ]
    )


# =========================================== CALIBRATION ===========================================


class Pattern:
    """
    Chessboard pattern parameters
    """

    def __init__(self, height, width, square_size):
        self.pattern_height = height  # number of INNER corners height wise
        self.pattern_width = width  # number of INNER corners width wise
        self.square_size = float(square_size)  # in world coordinate system (m,mm,...)
        self.pattern_type = "chessboard"
        self.pattern_size = (
            self.pattern_width,
            self.pattern_height,
        )  # number of (inner) corners

        self.pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        self.pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        self.pattern_points *= square_size


def get_image_points(
    calibration_image: np.ndarray, calibration_pattern: Pattern
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Find (inner) corners of chessboard calibration pattern, approximate them (subpixel precision)

    Args:
        calibration_image: input image contaninig calibration pattern
        calibration_pattern: parameters (height, width, ...) of the real world calibration pattern used

    Returns:
        image_points: array of coordinates of found inner corners of pattern
        debug_image: image with found corners and origin visualized
    """
    found = False
    corners = 0
    found, corners = cv2.findChessboardCorners(
        image=calibration_image, patternSize=calibration_pattern.pattern_size
    )
    if found:
        if corners.shape[0] != calibration_pattern.pattern_points.shape[0]:
            print(
                f"Not all corners found. Expected: {calibration_pattern.pattern_points.shape[0]}, found: {corners.shape[0]}"
            )
        termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.1)
        # refine corners
        cv2.cornerSubPix(
            image=calibration_image,
            corners=corners,
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=termination_criteria,
        )

        # points in camera/image space
        imgage_points = corners.reshape(-1, 2)

    else:
        print("corners not found")
        return None

    debug_image = cv2.cvtColor(calibration_image, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(debug_image, calibration_pattern.pattern_size, corners, found)

    # draw real world origin on image
    debug_image = cv2.circle(
        debug_image,
        (int(imgage_points[0][0]), int(imgage_points[0][1])),
        10,
        (255, 0, 0),
        2,
    )

    return imgage_points, debug_image


def get_extrinsics(calibration_img_path: str, debug_fig_path: str = None):
    """
    Get the extrinsic camera parameters using a calibration image and the intrinsic camera parameters.

    Args:
        calibration_img_path: path of the calibration image
        camera_serial_number: serial number of the camera, e.g.: SN123456. Used to get the intrinsic parameters for the exact model.
        degub_fig_path: path to save a degub img to

    Returns:
        R: rotation matrix
        t_vec: translation vector
    """

    with open("./calibration_metadata.json", "r") as f:
        metadata = json.load(f)

    distortion_coeffs = metadata["color_intrinsics"]["coeffs"]
    distortion_coeffs = np.array(distortion_coeffs)

    fx = metadata["color_intrinsics"]["fx"]
    fy = metadata["color_intrinsics"]["fy"]
    cx = metadata["color_intrinsics"]["ppx"]
    cy = metadata["color_intrinsics"]["ppy"]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    calibration_image = cv2.imread(calibration_img_path, cv2.IMREAD_GRAYSCALE)
    calibration_image = cv2.rotate(calibration_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # describe real world calibration pattern parameteres
    board_pattern = Pattern(height=7, width=4, square_size=0.1)

    camera_coords = []

    # get specific points of the image used for calibration, associate them with real-life points of calib pattern
    img_points, obj_points, debug_img = get_image_points(calibration_image, board_pattern)

    # get the transformations describing image-object relationship
    success, rvec, t_vec = cv2.solvePnP(
        obj_points,
        img_points,
        camera_matrix,
        distortion_coeffs,
        useExtrinsicGuess=False,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if debug_fig_path:
        cv2.imwrite(filename=debug_fig_path + "detected_points.png", img=debug_img)

    if success:
        R, jac = cv2.Rodrigues(rvec)

        Xw = -np.matrix(R).T * np.matrix(t_vec)
        camera_coords.append(Xw)

        print(f"distance: {round(np.linalg.norm(Xw),2)} m")

    else:
        print(f"{calibration_img_path}... FAILED")

    return R, t_vec, rvec


# =========================================== QUALISYS ===========================================


def get_qualisys_metadata(filename: str) -> dict:
    """
    Read qualisys export file in .tsv format, parse metadata.
    Exported file has to include tsv-header (qualisys export setting).

    Args:
        filename: path to the qualisys export file in .tsv format

    Returns:
        qualisys_metadata: metadata in dict format
    """

    qualisys_metadata = {}
    with open(filename) as f:
        # read file with csv reader. no need for pandas for just a few lines
        reader = csv.reader(f, delimiter="\t", quotechar='"')

        for ind, row in enumerate(reader):
            # only the first 9 rows of the whole file contain the tsv header metadata
            if ind <= 8:
                if ind < 6:
                    qualisys_metadata[row[0]] = float(row[1])
                # this row has 2 pieces of info; timestamp of the recording from qualisys (this cant be used for sync), timestamp from the start of host system
                elif ind == 7:
                    qualisys_metadata[row[0] + "_QUALISYS"] = row[1]
                    qualisys_metadata[row[0] + "_FROM_SYSTEM_START"] = row[2]
                else:
                    qualisys_metadata[row[0]] = row[1]
            if ind == 9:
                qualisys_metadata["marker_names"] = row[1:]

    return qualisys_metadata


def get_qualisys_keypoints(filename: str) -> tuple[np.ndarray, list]:
    """
    Read qualisys export file in .tsv format, format contents to fit multidimensional np.array,
    delete gap-filled entries and convert measurements from milimeter to meter.

    Args:
        input_filename: path to input file (.tsv)
        output_filename: optional, path to save the result as a .npz file. Default is the input_filename directory.

    Returns:
        keypoints: np.array of keypoint data (frames, markers, dims)
        labels: keypoint labels as list
    """
    # read first 10 rows to get number of markers, frames, marker names...
    metadata = get_qualisys_metadata(filename)
    labels = metadata["marker_names"]

    # Read the TSV file, skipping the first 10 rows (only metadata)
    df = pd.read_csv(filename, sep="\t", header=None, skiprows=11, dtype=None)

    # drop last column since it is empty
    df = df.drop(df.columns[[-1]], axis=1)

    # Convert the DataFrame to a numpy array, transpose it (so it is (markers x dims) x frames)
    keypoints = df.to_numpy()[1:, :92].T

    # reshape the data to    markers x dims x frames
    keypoints = keypoints.reshape(int(metadata["NO_OF_MARKERS"]), 4, int(metadata["NO_OF_FRAMES"]))

    # iterate over all keypoints
    for kpt_idx, keypoint in enumerate(keypoints):
        # get the type of measurement for all frames of a given keypoint
        measurement_types = keypoint[3, :]
        # iterate over all dims (x,y,z,type) of a keypoint
        for dim_idx, dim in enumerate(keypoint):

            # replace coordinates with nan when it is gap-filled by qualisys, convert coords to float
            if dim_idx <= 2:
                keypoints[kpt_idx, dim_idx, :] = np.where(
                    measurement_types == "Measured", dim, "Nan"
                ).astype(float)

            # leave type as str ('Measured' or 'Gap-filled')
            elif dim_idx == 3:
                pass
    # remove type of measurement
    keypoints = keypoints[:, :3, :].astype(float)

    # convert from milimeters to meters
    keypoints = keypoints / 1000

    # calculate mid_hip_front keypoint
    left_hip_front = keypoints[labels.index("left_hip"), :, :]
    right_hip_front = keypoints[labels.index("right_hip"), :, :]
    mid_hip_front = np.mean((left_hip_front, right_hip_front), axis=0)

    # calculate mid_shoulder front keypoint
    left_shoulder = keypoints[labels.index("left_shoulder"), :]
    right_shoulder = keypoints[labels.index("right_shoulder"), :]
    mid_shoulder = np.mean((left_shoulder, right_shoulder), axis=0)

    # calculate center_of_mass keypoint (com_fraction of the way from mid_hip to mid_shoulder)
    com_fraction = 0.15
    center_of_mass = mid_hip_front + com_fraction * (mid_shoulder - mid_hip_front)

    # add new keypoints and labels to existing keypoints and labels
    keypoints = np.concatenate((keypoints, mid_shoulder[np.newaxis, :, :]), axis=0)
    # keypoints = np.concatenate((keypoints, mid_hip_back[np.newaxis, :, :]), axis=0)
    keypoints = np.concatenate((keypoints, mid_hip_front[np.newaxis, :, :]), axis=0)
    keypoints = np.concatenate((keypoints, center_of_mass[np.newaxis, :, :]), axis=0)

    keypoints = keypoints.transpose(2, 0, 1)
    labels.append("mid_shoulder")
    labels.append("mid_hip")
    labels.append("center_of_mass")

    return keypoints, labels


def load_keypoints_and_labels(filename: str) -> tuple[np.ndarray, list]:
    """
    Load keypoint and label data from .npz file
    File has to have keys: "keypoints", "labels"

    Args:
        filename: path to the .npz file

    Returns:
        keypoints: loaded keypoint data as np.array
        labels: loaded labels as list
    """
    if os.path.exists(filename):
        loaded_data = np.load(filename)
        keypoints = loaded_data["keypoints"]
        labels = loaded_data["labels"].tolist()
        print(f"loaded data from: {filename} with keys: {[key for key in loaded_data.keys()]}")
    else:
        raise FileNotFoundError(f"File: {filename} does not exist!")
    return keypoints, labels


# =========================================== GAIT EVENTS ===========================================


@dataclass
class GaitEvent:
    """
    Dataclass to store data related to a single Gait Event.

    Attributes:
        type: Initial Contact (IC) or Final Contact (FC)
        frame: point of time (frame) at which the event occured
        side: foot (left/right) the event belongs to
        perspective: perspective (front/back) of segment to which the event belong to
        segment: indicates whether the segment contains straight walking or turning action
        segment_index: index of segment (starting from 0 at the beginning of recording). Incremented with perspective change
        relative_position: position of the event within the segment, relative to the segment's end (1=segment start in time, 0=segment end)
    """

    type: Literal["IC", "FC"]
    frame: int
    side: Literal["left", "right"]
    perspective: Literal["front", "back"] | None = None
    segment: Literal["straight", "turn"] | None = None
    segment_index: int | None = None
    relative_position: float | None = None

    def __str__(self):
        # one table row; str() so that None fields can be padded
        fields = [
            self.type,
            self.side,
            self.frame,
            self.perspective,
            self.segment,
            self.segment_index,
            self.relative_position,
        ]
        return "".join(f"{str(field):<10}" for field in fields)


def get_ankle_speed(keypoints: np.ndarray, labels: list[str], side: str, fps: float) -> np.ndarray:
    """3D speed (m/s) of the "<side>_ankle" keypoint, from frame-to-frame differences; (frames,)."""
    ankle = keypoints[:, labels.index(f"{side}_ankle"), :]
    return np.linalg.norm(np.diff(ankle, axis=0, prepend=ankle[[0]]), axis=1) * fps


def get_ground_contact_stages(
    ankle_speed: np.ndarray,
    fps: float,
    speed_threshold: float,
    min_swing_time: float,
    min_stance_time: float,
    event_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Ground contact signal (1 = on the ground) after each cleaning stage, in order:
    "raw" (speed below threshold), "masked" (on the ground outside event_mask),
    "no short swings", "no short contacts". The last one is the final result.
    """
    ground_contact_raw = (ankle_speed < speed_threshold).astype(int)
    # outside the mask the foot is treated as on the ground, so no events are created there
    if event_mask is None:
        ground_contact_masked = ground_contact_raw
    else:
        ground_contact_masked = np.where(event_mask, ground_contact_raw, 1)
    ground_contact_no_short_swing = fill_short_runs(
        ground_contact_masked, value=0, min_length=round(min_swing_time * fps)
    )
    ground_contact_no_short_contact = fill_short_runs(
        ground_contact_no_short_swing, value=1, min_length=round(min_stance_time * fps)
    )
    return {
        "raw": ground_contact_raw,
        "masked": ground_contact_masked,
        "no short swings": ground_contact_no_short_swing,
        "no short contacts": ground_contact_no_short_contact,
    }


def get_gait_events(
    keypoints: np.ndarray,
    labels: list[str],
    fps: float,
    speed_threshold: float,
    min_swing_time: float,
    min_stance_time: float,
    event_mask: np.ndarray | None = None,
    show_debug_figure: bool = False,
) -> list[GaitEvent]:
    """
    Initial and final contacts from ankle speed: the foot is on the ground while its 3D speed is
    below `speed_threshold`. Swing and stance periods that are too short are merged into their neighbours.
    The returned events only have type, frame and side; context (gait segment, pull, ...) is up to the caller.

    Args:
        keypoints: (frames, keypoints, 3) positions in meters
        labels: keypoint labels, must contain "left_ankle" and "right_ankle"
        fps: sampling rate of the positions
        speed_threshold: ankle speed (m/s) below which the foot counts as on the ground
        min_swing_time: swing periods shorter than this (s) are treated as ground contact
        min_stance_time: stance periods shorter than this (s) are treated as swing
        event_mask: (frames,) bool, True where events are allowed (e.g. straight walking); None = everywhere
        show_debug_figure: if True, a debug figure per side is shown
    Returns:
        events of all sides, sorted by frame
    """
    all_events = []
    for side in ["left", "right"]:
        ankle_speed = get_ankle_speed(keypoints, labels, side, fps)
        stages = get_ground_contact_stages(
            ankle_speed, fps, speed_threshold, min_swing_time, min_stance_time, event_mask
        )
        ground_contact = stages["no short contacts"]

        contact_changes = np.diff(ground_contact, prepend=ground_contact[0])
        initial_contact_frames = np.flatnonzero(contact_changes == 1)
        final_contact_frames = np.flatnonzero(contact_changes == -1)

        all_events += [
            GaitEvent(type="IC", frame=int(frame), side=side) for frame in initial_contact_frames
        ]
        all_events += [
            GaitEvent(type="FC", frame=int(frame), side=side) for frame in final_contact_frames
        ]

        if show_debug_figure:
            plot_gait_event_debug_figure(
                ankle_speed, speed_threshold, stages, title=f"{side} ankle"
            )

    return sorted(all_events, key=lambda event: event.frame)


def get_gait_events_two_stage(
    keypoints: np.ndarray,
    labels: list[str],
    fps: float,
    ankle_thr: float,
    initial_speed_threshold: float,
    max_stride_time: float,
    min_swing_time: float,
    min_stance_time: float,
    event_mask: np.ndarray | None = None,
    show_debug_figure: bool = False,
) -> tuple[list[GaitEvent], float]:
    """
    Two-stage gait event detection:
    1. gait events with ankle_thr * initial_speed_threshold as the ankle speed threshold
    2. initial gait cycles from these events: from one IC to the next IC of the same foot, if it
       comes within max_stride_time. Stride velocity of a cycle = 3D ankle displacement between
       the two ICs / time between them
    3. gait events again, with ankle_thr * mean stride velocity of the initial cycles as the
       threshold

    Args:
        keypoints: (frames, keypoints, 3) positions in meters
        labels: keypoint labels, must contain "left_ankle" and "right_ankle"
        fps: sampling rate of the positions
        ankle_thr: multiplier of the velocity, gives the ankle speed threshold in both stages
            (same as heel_thr in gait_util.GaitRecording)
        initial_speed_threshold: velocity (m/s) of the first stage
        max_stride_time: longer IC -> IC intervals (s) are not used as initial gait cycles
            (e.g. sitting or standing still between two steps)
        min_swing_time, min_stance_time, event_mask: passed to get_gait_events
        show_debug_figure: if True, the debug figures of the second stage are shown
    Returns:
        gait events of the second stage, ankle speed threshold (m/s) used in the second stage
    """
    initial_gait_events = get_gait_events(
        keypoints,
        labels,
        fps=fps,
        speed_threshold=ankle_thr * initial_speed_threshold,
        min_swing_time=min_swing_time,
        min_stance_time=min_stance_time,
        event_mask=event_mask,
    )

    initial_stride_velocities = []
    for side in ["left", "right"]:
        ankle = keypoints[:, labels.index(f"{side}_ankle"), :]
        initial_contact_frames = [
            event.frame
            for event in initial_gait_events
            if event.type == "IC" and event.side == side
        ]
        for stride_start, stride_end in zip(initial_contact_frames, initial_contact_frames[1:]):
            stride_time = (stride_end - stride_start) / fps
            if stride_time > max_stride_time:
                continue
            stride_length = np.linalg.norm(ankle[stride_end] - ankle[stride_start])
            initial_stride_velocities.append(stride_length / stride_time)

    if not initial_stride_velocities:
        raise ValueError(
            f"no initial gait cycles found with initial_speed_threshold={initial_speed_threshold} m/s "
            f"and max_stride_time={max_stride_time} s"
        )
    mean_stride_velocity = np.nanmean(initial_stride_velocities)
    speed_threshold = ankle_thr * mean_stride_velocity

    gait_events = get_gait_events(
        keypoints,
        labels,
        fps=fps,
        speed_threshold=speed_threshold,
        min_swing_time=min_swing_time,
        min_stance_time=min_stance_time,
        event_mask=event_mask,
        show_debug_figure=show_debug_figure,
    )
    return gait_events, speed_threshold


@dataclass
class Step:
    """
    A single step of one foot: the foot leaves the ground (FC) and lands again (IC).

    Attributes:
        FC: FC event, the foot leaves the ground
        IC: IC event of the same foot, the foot lands again
        step_length: straight-line distance (m) of the ankle between FC and IC, set by set_step_length
        step_velocity: average ankle velocity (m/s) between FC and IC, set by set_step_velocity
        type: corrective (the ankle moves away from the baseline position) or return (it moves back
            toward it), set by set_type; None if a position needed for the decision is NaN
    """

    FC: GaitEvent
    IC: GaitEvent
    step_length: float | None = field(default=None, init=False)
    step_velocity: float | None = field(default=None, init=False)
    type: Literal["corrective", "return"] | None = field(default=None, init=False)

    def __post_init__(self):
        if self.FC.type != "FC" or self.IC.type != "IC":
            raise ValueError("a step needs an FC event and an IC event")
        if self.FC.side != self.IC.side:
            raise ValueError("the FC and IC events of a step must belong to the same foot")
        if self.FC.frame >= self.IC.frame:
            raise ValueError("the FC event of a step must come before its IC event")

    @property
    def side(self) -> Literal["left", "right"]:
        return self.FC.side

    def set_step_length(self, keypoints: np.ndarray, labels: list[str]):
        """
        Euclidean distance between the ankle position at FC and at IC.

        Args:
            keypoints: (frames, keypoints, 3) of the whole recording, indexed with recording frames
                like the events
            labels: keypoint labels, must contain "<side>_ankle"
        """
        ankle = keypoints[:, labels.index(f"{self.side}_ankle"), :]
        self.step_length = float(np.linalg.norm(ankle[self.IC.frame] - ankle[self.FC.frame]))

    def set_step_velocity(self, fps: float):
        """
        Step length divided by the step duration (FC to IC). Needs set_step_length first.

        Args:
            fps: sampling rate of the keypoints the events were detected on
        """
        if self.step_length is None:
            raise ValueError("call set_step_length before set_step_velocity")
        step_duration = (self.IC.frame - self.FC.frame) / fps
        self.step_velocity = self.step_length / step_duration

    def set_type(self, keypoints: np.ndarray, labels: list[str], baseline_position: np.ndarray):
        """
        Corrective if the ankle lands farther from the baseline position than where it lifted off,
        return if it lands closer. None if the baseline or the ankle at FC or IC is NaN.

        Args:
            keypoints: (frames, keypoints, 3) of the whole recording, indexed with recording frames
                like the events
            labels: keypoint labels, must contain "<side>_ankle"
            baseline_position: (3,) reference position, e.g. Pull.baseline_ankle_position
        """
        ankle = keypoints[:, labels.index(f"{self.side}_ankle"), :]
        distance_at_lift_off = np.linalg.norm(ankle[self.FC.frame] - baseline_position)
        distance_at_landing = np.linalg.norm(ankle[self.IC.frame] - baseline_position)
        # a NaN distance would fail the comparison and silently give "return"
        if np.isnan(distance_at_lift_off) or np.isnan(distance_at_landing):
            self.type = None
        else:
            self.type = "corrective" if distance_at_landing > distance_at_lift_off else "return"


def get_steps(
    gait_events: list[GaitEvent],
    keypoints: np.ndarray,
    labels: list[str],
    min_step_length: float = 0.0,
) -> list[Step]:
    """
    Pair each FC with the next event of the same foot if that event is an IC.
    An FC followed by another FC (missed IC) or by nothing gives no step.
    Every step gets its step length; steps shorter than min_step_length (shuffles) are dropped.

    Args:
        gait_events: events of the whole recording, e.g. from get_gait_events
        keypoints: (frames, keypoints, 3) of the whole recording, indexed with recording frames
        labels: keypoint labels, must contain "left_ankle" and "right_ankle"
        min_step_length: minimum step length (m); 0 keeps all steps
    Returns:
        steps of both feet, sorted by FC frame
    """
    steps = []
    for side in ["left", "right"]:
        side_events = sorted(
            (event for event in gait_events if event.side == side), key=lambda event: event.frame
        )
        for event, next_event in zip(side_events, side_events[1:]):
            if event.type == "FC" and next_event.type == "IC":
                step = Step(FC=event, IC=next_event)
                step.set_step_length(keypoints, labels)
                # a NaN length (missing ankle) fails the comparison, so the step is kept
                if not step.step_length < min_step_length:
                    steps.append(step)
    return sorted(steps, key=lambda step: step.FC.frame)


def plot_gait_event_debug_figure(
    ankle_speed, speed_threshold, ground_contact_stages: dict, title: str = ""
):
    """Ankle speed with the threshold, and the ground contact signal after each cleaning stage."""
    fig, axs = plt.subplots(2, 1, figsize=(14, 3), sharex=True)
    fig.suptitle(title)

    axs[0].set_title("ankle speed")
    axs[0].plot(ankle_speed, label="speed")
    axs[0].axhline(speed_threshold, color="r", linestyle="--", label="threshold")

    axs[1].set_title("ground contact (1 = on the ground), offset per stage")
    stage_count = len(ground_contact_stages)
    for stage_number, (stage_name, ground_contact) in enumerate(ground_contact_stages.items()):
        axs[1].plot(ground_contact + 2 * (stage_count - 1 - stage_number), label=stage_name)

    for ax in axs:
        # outside the axes, to the right, so it doesn't cover the data
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.grid()
    plt.tight_layout()
    return fig
