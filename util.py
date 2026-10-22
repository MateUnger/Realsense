import os
import time
import json
import cv2
import numpy as np
import importlib
from typing import Literal
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math
from scipy.interpolate import CubicSpline
import copy
from OneEuroFilter import OneEuroFilter
from scipy.signal import find_peaks, peak_widths, butter, sosfiltfilt, filtfilt, resample_poly
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
    return zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1))


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


def subtract_mean(keypoints: np.ndarray) -> tuple:
    """Subtract the mean (over time) from every keypoint for each dimention.
    Args:
        keypoints: array shaped (frames, keypoints, dims)
    Returns:
        centered: keypoints centered around their mean over time
        mean: (1, keypoints, dims) used for centering"""
    mean = np.nanmean(keypoints, axis=0, keepdims=True)
    centered_keypoints = keypoints - mean
    return centered_keypoints, mean


def resample_keypoints(keypoints: np.ndarray, original_fps: float, target_fps: float):
    """Resample keypoint data to desired sampling frequency.
    Note, the DC offset removed first/centered (to avoid filter leaks), then added back.
    Args:
        keypoints: array (frames, keypoints, dims)
        original_fps: sampling frequency of the input
        target_fps: desired sampling frequency
    Returns:
        resampled: keypoints resampled to target_fps (with mean added back)."""
    ratio = Fraction(str(target_fps)) / Fraction(str(original_fps))
    centered, mean = subtract_mean(keypoints)
    resampled = resample_poly(centered, ratio.numerator, ratio.denominator, axis=0, padtype="edge")
    return resampled + mean


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

    # add new keypoints and labels to existing keypoints and labels
    keypoints = np.concatenate((keypoints, mid_shoulder[np.newaxis, :, :]), axis=0)
    # keypoints = np.concatenate((keypoints, mid_hip_back[np.newaxis, :, :]), axis=0)
    keypoints = np.concatenate((keypoints, mid_hip_front[np.newaxis, :, :]), axis=0)

    keypoints = keypoints.transpose(2, 0, 1)
    labels.append("mid_shoulder")
    labels.append("mid_hip")

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
