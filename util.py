import cv2
import json
import numpy as np
import time
import importlib
from typing import Literal
import pyrealsense2 as rs
import os
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math
from scipy.interpolate import CubicSpline
import copy
from OneEuroFilter import OneEuroFilter
from scipy.signal import find_peaks, peak_widths, butter, sosfiltfilt
from scipy.ndimage import maximum_filter1d
from scipy.interpolate import CubicSpline, PchipInterpolator

rtmlib_module = importlib.import_module("rtmlib")

# resolution-fps pairs that work with both color and depth
resolution_presets = {
    "848x480@60": {"width": 848, "height": 480, "fps": 60},
    "640x360@90": {"width": 640, "height": 360, "fps": 90},
    "480x270@90": {"width": 480, "height": 270, "fps": 90},
}

stereo_labels = [
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
    "mid_hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
    "mid_shoulder",
]


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


# =========================================== POSE ESTIMATION ===========================================
labels = [
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
    "mid_hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
    "mid_shoulder",
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


def filter_keypoints_euro(keypoints: np.ndarray, fps: float):
    n_frames, n_keypoints, n_dims = keypoints.shape

    # One filter for each keypoint and each XYZ coordinate
    filters = [
        [OneEuroFilter(freq=fps, mincutoff=1, beta=0.1, dcutoff=1.0) for _ in range(n_dims)]
        for _ in range(n_keypoints)
    ]

    filtered = np.empty_like(keypoints, dtype=float)

    for frame_idx in range(n_frames):
        timestamp = frame_idx / fps

        for kpt_idx in range(n_keypoints):
            for xyz in range(n_dims):
                filtered[frame_idx, kpt_idx, xyz] = filters[kpt_idx][xyz].filter(
                    keypoints[frame_idx, kpt_idx, xyz], timestamp
                )

    return filtered


from scipy.signal import butter, filtfilt


def lowpass_keypoints(
    keypoints: np.ndarray, fps: float, cutoff: float, order: int = 4
) -> np.ndarray:
    """
    Butterworth low-pass along time axis with zero-phase shift (forward and backward pass).
    Only meant to be used with 2D keypoints as they contain no nans or gaps.

    Args:
        keypoints: (n_frames, n_keypoints, n_dims)
        fps: sampling rate in Hz
        cutoff: cutoff in Hz (-6 dB point after forward-backward filtering)
        order: per-pass filter order (effective order is doubled)
    """
    if not 0 < cutoff < fps / 2:
        raise ValueError(f"cutoff must be in (0, {fps / 2}) Hz, got {cutoff}")

    sos = butter(order, cutoff, btype="low", fs=fps, output="sos")
    return sosfiltfilt(sos, keypoints, axis=0)


def filter_2d_keypoints(
    keypoints_2d: np.ndarray,
    fps: float,
    plot_kpt_idx: int | None = None,
):
    keypoints_2d_filtered = filter_keypoints_euro(
        keypoints=keypoints_2d,
        fps=fps,
    ).copy()

    n_keypoints = keypoints_2d_filtered.shape[1]

    # Store plotting data for the requested keypoint
    plot_data = None

    for kpt_idx in range(n_keypoints):
        og = keypoints_2d_filtered[:, kpt_idx, :].copy()
        kpt_coords = og.copy()

        movement = np.linalg.norm(
            np.diff(
                kpt_coords,
                axis=0,
                prepend=np.expand_dims(kpt_coords[0, :], axis=0),
            ),
            axis=1,
        )

        std = np.std(movement)

        # Find peaks to remove jerky outliers
        min_peak_height = std * 2
        min_peak_distance = int(fps * 0.05)
        prominance = 2 * std

        peaks, _ = find_peaks(
            movement,
            height=min_peak_height,
            distance=min_peak_distance,
            prominence=prominance,
        )

        _, width_heights, left_ips, right_ips = peak_widths(
            x=movement,
            peaks=peaks,
            rel_height=0.95,
        )

        frames = np.arange(len(kpt_coords))

        # Remove all detected spike regions
        for left, right in zip(left_ips, right_ips):
            start = math.floor(left)
            end = math.ceil(right)

            kpt_coords[start : end + 1, :] = np.nan

        # Find short valid segments between NaN regions
        not_nan = ~np.isnan(kpt_coords[:, 0])

        diff = np.diff(not_nan.astype(int))

        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if not_nan[0]:
            starts = np.r_[0, starts]

        if not_nan[-1]:
            ends = np.r_[ends, len(not_nan)]

        # Remove short valid segments
        for start, end in zip(starts, ends):
            surrounded_by_nan = (
                start > 0 and end < len(not_nan) and not not_nan[start - 1] and not not_nan[end]
            )

            if surrounded_by_nan and end - start < fps * 0.25:
                kpt_coords[start:end, :] = np.nan

        # Cubic spline interpolation
        for dim in range(2):
            coordinate = kpt_coords[:, dim]
            not_nan = ~np.isnan(coordinate)

            if np.sum(not_nan) < 4:
                continue

            spline = CubicSpline(
                frames[not_nan],
                coordinate[not_nan],
            )

            coordinate[~not_nan] = spline(frames[~not_nan])

            keypoints_2d_filtered[:, kpt_idx, dim] = coordinate

        # Save plotting data for selected keypoint
        if kpt_idx == plot_kpt_idx:
            plot_data = {
                "og": og,
                "movement": movement,
                "peaks": peaks,
                "min_peak_height": min_peak_height,
                "left_ips": left_ips,
                "right_ips": right_ips,
                "width_heights": width_heights,
            }

    # Low-pass filter all keypoints
    keypoints_2d_filtered = lowpass_keypoints(
        keypoints_2d_filtered,
        fps,
        2,
        4,
    )

    # Plot selected keypoint
    if plot_kpt_idx is not None and plot_data is not None:
        fig, axs = plt.subplots(3, 1, figsize=(15, 7))

        axs[0].set_title(f"Keypoint {plot_kpt_idx}")
        axs[0].plot(plot_data["og"])

        axs[1].set_title("movement")
        axs[1].plot(plot_data["movement"])
        axs[1].plot(
            plot_data["peaks"],
            plot_data["movement"][plot_data["peaks"]],
            "r*",
            markersize=4,
        )
        axs[1].hlines(
            plot_data["min_peak_height"],
            0,
            plot_data["movement"].shape[0],
        )
        axs[1].plot(
            plot_data["left_ips"],
            plot_data["width_heights"],
            "o",
            label="turn start",
            markersize=4,
        )
        axs[1].plot(
            plot_data["right_ips"],
            plot_data["width_heights"],
            "o",
            label="turn end",
            markersize=4,
        )
        axs[1].set_ylim([-5, 50])

        axs[2].plot(keypoints_2d_filtered[:, plot_kpt_idx, :])
        axs[2].set_title(f"Keypoint {plot_kpt_idx} filtered")

        for ax in axs[:-2]:
            ax.set_xticks([])
            ax.grid()

        plt.tight_layout()
        plt.show()

    return keypoints_2d_filtered


# def interpolate_gaps(keypoints: np.ndarray, fps: int, max_gap: float) -> np.ndarray:
#     """
#     Fills gaps below max_gap size along each dimension (1D) of the input array.
#     Args:
#         keypoints (np.ndarray):  Input data with shape (n_keypoints, n_dims, n_frames).
#         fps (int):          Frames per second of the data.
#         max_gap (float):    Maximum gap size in seconds


#     """
#     # n_keypoints, n_dims, n_frames = keypoints.shape
#     n_frames, n_keypoints, n_dims = keypoints.shape
#     max_gap_frames = int(max_gap * fps)

#     filled_data = np.copy(keypoints)  # Preserve original data

#     for kpt in range(n_keypoints):
#         for dim in range(n_dims):
#             signal = keypoints[:, kpt, dim]
#             nan_indices = np.where(np.isnan(signal))[0]

#             if len(nan_indices) == 0:
#                 continue

#             # Identify NaN segments
#             diff = np.diff(nan_indices)
#             segment_starts = np.insert(nan_indices[np.where(diff > 1)[0] + 1], 0, nan_indices[0])
#             segment_ends = np.append(nan_indices[np.where(diff > 1)[0]], nan_indices[-1])

#             # Interpolate gaps within the allowed size
#             for start, end in zip(segment_starts, segment_ends):
#                 gap_size = end - start + 1
#                 if gap_size <= max_gap_frames:

#                     valid_indices = np.where(~np.isnan(signal))[0]
#                     # fit cubic spline if there are enough points
#                     if len(valid_indices) >= 2:
#                         cs = CubicSpline(valid_indices, signal[valid_indices])
#                         filled_data[start : end + 1, kpt, dim] = cs(np.arange(start, end + 1))

#                     # use linear interpolation if not enough valid points
#                     else:
#                         non_nan_idx = np.where(~np.isnan(signal))[0]
#                         if non_nan_idx.size == 0:
#                             continue

#                         filled_data[:, kpt, dim] = np.interp(
#                             np.arange(n_frames), non_nan_idx, signal[non_nan_idx]
#                         )

#     return filled_data


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


def get_basename(full_file_path: str) -> str:
    """
    Return the basename of input file path e.g.: ...folder1/folder2/image_name.png -> image_name

    Args:
        full_file_path: input file path string

    Returns:
        basename: basename of the input (without folders or extension)
    """
    basename_without_ext = os.path.splitext(full_file_path)[0]
    basename = os.path.basename(basename_without_ext)
    return basename


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


def format_qualisys_export(
    input_filename: str, output_filename: str = None
) -> tuple[np.ndarray, list]:
    # TODO: add logging, maybe return value
    """
    Read qualisys export file in .tsv format, format contents to fit multidimensional np.array,
    delete gap-filled entries and convert measurements from milimeter to meter.

    Args:
        input_filename: path to input file (.tsv)
        output_filename: optional, path to save the result as a .npz file. Default is the input_filename directory.

    Returns:
        keypoints: np.array of keypoint data
        labels: keypoint labels as list
    """
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"File {input_filename} does not exits")

    elif output_filename == None:
        basename = get_basename(input_filename)
        directory = os.path.dirname(input_filename)
        output_filename = os.path.join(directory, f"qualisys.npz")

    # read first 10 rows to get number of markers, frames, marker names...
    metadata = get_qualisys_metadata(input_filename)
    labels = metadata["marker_names"]

    # Read the TSV file, skipping the first 10 rows (only metadata)
    df = pd.read_csv(input_filename, sep="\t", header=None, skiprows=11, dtype=None)

    # TODO: find out why is is there at all
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
    # remove dim 3 (type of measurement)
    keypoints = keypoints[:, :3, :].astype(float)

    # convert from milimeters to meters
    keypoints = keypoints / 1000

    # calculate mid_hip_front keypoint
    left_hip_front = keypoints[labels.index("left_hip"), :, :]
    right_hip_front = keypoints[labels.index("right_hip"), :, :]
    mid_hip_front = np.mean((left_hip_front, right_hip_front), axis=0)

    # calculate mid_hip_back keypoint
    # left_hip_back = keypoints[labels.index("left_hip_back"), :, :]
    # right_hip_back = keypoints[labels.index("right_hip_back"), :, :]
    # mid_hip_back = np.mean((left_hip_back, right_hip_back), axis=0)

    # calculate mid_shoulder front keypoint
    left_shoulder = keypoints[labels.index("left_shoulder"), :]
    right_shoulder = keypoints[labels.index("right_shoulder"), :]
    mid_shoulder = np.mean((left_shoulder, right_shoulder), axis=0)

    # add new keypoints and labels to existing keypoints and labels
    keypoints = np.concatenate((keypoints, mid_shoulder[np.newaxis, :, :]), axis=0)
    # keypoints = np.concatenate((keypoints, mid_hip_back[np.newaxis, :, :]), axis=0)
    keypoints = np.concatenate((keypoints, mid_hip_front[np.newaxis, :, :]), axis=0)
    labels.append("mid_shoulder")
    # labels.append("mid_hip_back")
    labels.append("mid_hip")

    # np.savez(output_filename, keypoints=keypoints, labels=labels)
    return keypoints, labels


def filter_data(
    data: np.ndarray,
    sampling_rate: float,
    filter_type: Literal["lowpass", "bandpass"],
    cutoff: list,
    order: int,
    gap_size: int,
) -> np.ndarray:
    """
    Interpolates and applies filter to data (with NaNs).

    Args:
        data (np.ndarray):      Input data with shape (n_kpt, n_dims, n_frames).
        sampling_rate (float):  Sampling rate of the data.
        cutoff (float):         Cutoff frequency for the low-pass filter.
        order (int):            Order of the Butterworth filter.
        gap_size (int):         Maximum size (in seconds) of gaps to fill

    Returns:
        filtered_data (np.ndarray): Interpolated and filtered data with the origianl Nan values in place.

    """
    n_kpt, n_dims, n_frames = data.shape
    data = interpolate_gaps(data, sampling_rate, gap_size)  # Fill short gaps
    filtered_data = copy.deepcopy(data)

    match filter_type:
        case "lowpass":
            cutoff = cutoff[0]
        case "bandpass":
            cutoff = np.asarray(cutoff)
    # Design Butterworth low-pass filter
    b, a = butter(N=order, Wn=cutoff, btype=filter_type, analog=False, fs=sampling_rate)

    for kpt in range(n_kpt):
        for dim in range(n_dims):
            trajectory = data[kpt, dim, :]

            # search for NaNs
            nans = np.isnan(trajectory)

            # if there are nans, interpolate the missing values for subsequent filtering
            if np.any(nans):
                valid_indices = ~nans
                trajectory[nans] = np.interp(
                    np.flatnonzero(nans),
                    np.flatnonzero(valid_indices),
                    trajectory[valid_indices],
                )

            trajectory = filtfilt(b, a, trajectory)  # Apply filter
            trajectory[nans] = np.nan  # Restore NaNs

            filtered_data[kpt, dim, :] = trajectory

    return filtered_data


def get_segment_borders(valid: np.ndarray):
    """
    Return starts, ends of contiguous valid (True) segments (end exclusive).
    """
    diff = np.diff(np.concatenate(([0], valid.astype(np.int8), [0])))
    return zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1))


def filter_keypoints(
    keypoints: np.ndarray,
    sampling_rate: float,
    cutoff: float,
    order: int,
    gap_size: float,
) -> np.ndarray:
    """
    Fills short gaps, then applies a zero-phase Butterworth filter to each
    contiguous run of valid samples. Remaining NaNs (long gaps) are kept.

    Args:
        keypoints: shape (n_frames, n_kpt, n_dims)
        sampling_rate: sampling rate in Hz
        cutoff: cutoff frequency in Hz
        order: Butterworth order (effective order is doubled by filtfilt)
        gap_size: maximum gap (seconds) to interpolate
    """

    if cutoff >= sampling_rate / 2:
        raise ValueError("cutoff must be below the Nyquist frequency")

    sos = butter(order, cutoff, btype="lowpass", fs=sampling_rate, output="sos")
    min_length = 3 * (2 * sos.shape[0] + 1) + 1  # safe lower bound for sosfiltfilt padding

    keypoints = interpolate_gaps(keypoints, sampling_rate, gap_size)
    keypoints_filtered = np.array(keypoints, dtype=float, copy=True)  # never touch the input

    for kpt, dim in np.ndindex(keypoints.shape[1], keypoints.shape[2]):
        coordinate = keypoints[kpt, dim]
        for start, end in get_segment_borders(~np.isnan(coordinate)):
            if end - start >= min_length:
                keypoints_filtered[kpt, dim, start:end] = sosfiltfilt(sos, coordinate[start:end])
            else:
                keypoints_filtered[kpt, dim, start:end] = np.nan  # too short to filter reliably

    return keypoints_filtered


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
