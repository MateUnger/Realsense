"""Stereo preprocessing pipeline, taken from the cells in test.ipynb.

Stages (each reads the previous stage's file from data/<name>/):
    convert      .db3 -> filter -> align -> _color.mp4, _depth.mkv, _timestamps_*.bin, _metadata.json
    keypoints_2d _color.mp4 -> _keypoints_2d.npy
    keypoints_3d _keypoints_2d.npy + _depth.mkv -> _keypoints_3d.npy
    calibration  calibration_image.png + _keypoints_3d.npy -> _keypoints_3d_global.npy
    resample     _keypoints_3d_global.npy -> filter, resample to 100 Hz, axis swap -> _stereo.npz

Usage (from the repo root):
    realsense_env\\Scripts\\python.exe preprocess_stereo.py tug2
    realsense_env\\Scripts\\python.exe preprocess_stereo.py tug2 --stages keypoints_3d calibration resample
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from tqdm.auto import tqdm

from Util.util import (
    create_ffmpeg_command,
    create_intrinsics_object,
    sample_depth_patch,
    get_image_points,
    Pattern,
    filter_keypoints_stereo,
    get_frame_times,
    resample_keypoints_to_rate,
    stereo_to_qualisys_axes,
    models,
    Custom,
    labels_17,
    labels_26,
)

DATA_FOLDER = "data"

# pose estimation settings
DETECTOR = "yolo_nano"  # 'yolo_nano', 'yolo_m', 'yolo_l', 'yolo_x'
POSE_ESTIMATOR = "rtm_pose_x_17"  # 'rtm_pose_t_17', ..., 'rtm_pose_x_17', 'vit_pose_l_17', 'rtm_pose_t_26', ...
LABELS = labels_17  # labels_26, labels_17

# calibration chessboard
CALIBRATION_PATTERN = Pattern(height=6, width=4, square_size=0.15)

STAGES = ["convert", "keypoints_2d", "keypoints_3d", "calibration", "resample"]


def get_path(recording_name: str, suffix: str) -> str:
    return os.path.join(DATA_FOLDER, recording_name, recording_name + suffix)


def check_files_exist(filenames: list):
    for filename in filenames:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist")


def convert(recording_name: str):
    """.db3 -> filter -> align -> .mp4 (color), .mkv (depth), .bin (timestamps), .json (metadata)"""
    recording_file_name = get_path(recording_name, ".db3")
    check_files_exist([recording_file_name])
    out_filename_color = get_path(recording_name, "_color.mp4")
    out_filename_depth = get_path(recording_name, "_depth.mkv")
    out_filename_timestamps_color = get_path(recording_name, "_timestamps_color.bin")
    out_filename_timestamps_depth = get_path(recording_name, "_timestamps_depth.bin")
    out_filename_metadata = get_path(recording_name, "_metadata.json")

    # config
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(recording_file_name, repeat_playback=False)

    profile = pipeline.start(config)

    # Get depth intrinsics
    depth_profile = profile.get_stream(rs.stream.depth)
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

    # Get color intrinsics
    color_profile = profile.get_stream(rs.stream.color)
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # align_to=rs.stream.color/infra... -> aligns depth to other
    align = rs.align(align_to=rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # filters
    threshold = rs.threshold_filter(min_dist=0.05, max_dist=7.0)
    depth_to_disparity = rs.disparity_transform(True)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.7)
    temporal.set_option(rs.option.filter_smooth_delta, 15)
    temporal.set_option(rs.option.holes_fill, 0)
    disparity_to_depth = rs.disparity_transform(False)

    ffmpeg_color = create_ffmpeg_command(
        width=color_intrinsics.width,
        height=color_intrinsics.height,
        fps=color_profile.fps(),
        encoder="h264_qsv",
        out_filename=out_filename_color,
    )
    video_color = subprocess.Popen(ffmpeg_color, stdin=subprocess.PIPE)

    ffmpeg_depth = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray16le",
        "-s",
        f"{color_intrinsics.width}x{color_intrinsics.height}",
        "-r",
        str(color_profile.fps()),
        "-i",
        "-",
        "-c:v",
        "ffv1",
        out_filename_depth,
    ]
    video_depth = subprocess.Popen(ffmpeg_depth, stdin=subprocess.PIPE)

    # output files
    timestamps_color_out = open(out_filename_timestamps_color, "wb", buffering=1024 * 1024)
    timestamps_depth_out = open(out_filename_timestamps_depth, "wb", buffering=1024 * 1024)

    frame_count = 0

    print(f"Input:  {recording_file_name}")
    # the frame count is only known at the end of playback, so the bar shows count and rate only
    progress_bar = tqdm(desc="Converting", unit="frame")

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                # at the end of the file wait_for_frames times out and playback is stopped;
                # any other error must not end the conversion as if it had finished
                if playback.current_status() == rs.playback_status.stopped:
                    break
                raise

            # filter the raw depth first; the filters process the depth frame of the frameset
            # and pass the color frame through, so the result can still be aligned
            filtered = threshold.process(frames)
            filtered = depth_to_disparity.process(filtered)
            filtered = spatial.process(filtered)
            filtered = temporal.process(filtered)
            filtered = disparity_to_depth.process(filtered)

            # then align the filtered depth to the color camera
            aligned_frames = align.process(filtered.as_frameset())
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            video_color.stdin.write(color_image.tobytes())
            video_depth.stdin.write(depth_image.tobytes())

            # the frameset timestamp follows the depth frame, but the keypoints come from the color
            # frame, so get_frame_times has to remove duplicates and place frames by the color timestamp
            timestamps_color_out.write(np.float64(frames.get_color_frame().get_timestamp()).tobytes())
            timestamps_depth_out.write(np.float64(frames.get_depth_frame().get_timestamp()).tobytes())

            frame_count += 1
            progress_bar.update(1)

    finally:
        progress_bar.close()

        video_color.stdin.close()
        color_exit_code = video_color.wait()

        video_depth.stdin.close()
        depth_exit_code = video_depth.wait()

        timestamps_color_out.close()
        timestamps_depth_out.close()

        pipeline.stop()

    # checked after cleanup, so an error in the loop isn't hidden by this one
    if color_exit_code != 0:
        raise RuntimeError(f"ffmpeg color encoding failed with exit code {color_exit_code}")
    if depth_exit_code != 0:
        raise RuntimeError(f"ffmpeg depth encoding failed with exit code {depth_exit_code}")

    metadata = {
        "frames": frame_count,
        "width": color_intrinsics.width,
        "height": color_intrinsics.height,
        "fps": color_profile.fps(),
        "color": out_filename_color,
        "depth": out_filename_depth,
        "depth_format": "uint16",
        "depth_alignment": "depth_to_color",
        "depth_scale_meters": depth_scale,
        "timestamps_color": out_filename_timestamps_color,
        "timestamps_depth": out_filename_timestamps_depth,
        "timestamp_unit": "milliseconds",
        "depth_intrinsics": {
            "width": depth_intrinsics.width,
            "height": depth_intrinsics.height,
            "ppx": depth_intrinsics.ppx,
            "ppy": depth_intrinsics.ppy,
            "fx": depth_intrinsics.fx,
            "fy": depth_intrinsics.fy,
            "model": str(depth_intrinsics.model),
            "coeffs": depth_intrinsics.coeffs,
        },
        "color_intrinsics": {
            "width": color_intrinsics.width,
            "height": color_intrinsics.height,
            "ppx": color_intrinsics.ppx,
            "ppy": color_intrinsics.ppy,
            "fx": color_intrinsics.fx,
            "fy": color_intrinsics.fy,
            "model": str(color_intrinsics.model),
            "coeffs": color_intrinsics.coeffs,
        },
    }

    with open(out_filename_metadata, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Frames: {frame_count}")
    print(f"Metadata:  {out_filename_metadata}")


def setup_cuda():
    import onnxruntime as ort

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    cuda_bin = site_packages / "nvidia" / "cu13" / "bin" / "x86_64"
    cudnn_bin = site_packages / "nvidia" / "cudnn" / "bin"

    os.add_dll_directory(str(cuda_bin))
    os.add_dll_directory(str(cudnn_bin))
    ort.preload_dlls(cuda=True, cudnn=True, directory=str(cuda_bin))

    print("ORT providers:", ort.get_available_providers())


def keypoints_2d(recording_name: str):
    """mp4 -> 2D keypoints (camera frame)"""
    input_video_filename = get_path(recording_name, "_color.mp4")
    filename_metadata = get_path(recording_name, "_metadata.json")
    out_filename_keypoints_2d = get_path(recording_name, "_keypoints_2d.npy")
    check_files_exist([input_video_filename, filename_metadata])

    with open(filename_metadata) as f:
        metadata = json.load(f)

    setup_cuda()
    custom = Custom(
        det_class="YOLOX",
        det=models["detectors"][DETECTOR]["path"],
        det_input_size=models["detectors"][DETECTOR]["input_size"],
        pose_class="RTMPose",  # 'RTMPose', 'ViTPose'
        pose=models["pose"][POSE_ESTIMATOR]["path"],
        pose_input_size=models["pose"][POSE_ESTIMATOR]["input_size"],
        backend="onnxruntime",
        device="cuda",
    )

    print(f"processing: {input_video_filename}")

    keypoints_2d_per_frame = []

    cap = cv2.VideoCapture(input_video_filename)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error opening video file: {input_video_filename}")

    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_num_frames, desc="2D keypoints", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # rotate frame to portrait mode so pose estimation works
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # inference
        keypoints, scores = custom(frame)

        # TODO: add tracking or ROI
        # store results (frames * kpt * (x,y)), remove multiperson cases
        if keypoints.ndim > 2:
            keypoints = keypoints[0, :, :]

        keypoints = np.squeeze(keypoints)

        keypoints_2d_per_frame.append(keypoints)
        progress_bar.update(1)

    progress_bar.close()
    cap.release()

    keypoints_2d_array = np.asarray(keypoints_2d_per_frame)

    # every later stage pairs keypoint frame i with timestamp i, so a short run must not be saved
    if len(keypoints_2d_array) != metadata["frames"]:
        raise RuntimeError(
            f"got keypoints for {len(keypoints_2d_array)} frames, but the conversion wrote {metadata['frames']} frames"
        )

    # add mid_shoulder (between left and right)
    left_shoulder = keypoints_2d_array[:, LABELS.index("left_shoulder"), :]
    right_shoulder = keypoints_2d_array[:, LABELS.index("right_shoulder"), :]
    mid_shoulder = np.nanmean([left_shoulder, right_shoulder], axis=0)
    mid_shoulder = np.expand_dims(np.array(mid_shoulder), axis=1)
    keypoints_2d_array = np.hstack((keypoints_2d_array, mid_shoulder))

    # add mid_hip (between left and right)
    left_hip = keypoints_2d_array[:, LABELS.index("left_hip"), :]
    right_hip = keypoints_2d_array[:, LABELS.index("right_hip"), :]
    mid_hip = np.nanmean([left_hip, right_hip], axis=0)
    mid_hip = np.expand_dims(np.array(mid_hip), axis=1)
    keypoints_2d_array = np.hstack((keypoints_2d_array, mid_hip))

    # add center_of_mass (com_fraction of the way from mid_hip to mid_shoulder)
    com_fraction = 0.15
    center_of_mass = mid_hip + com_fraction * (mid_shoulder - mid_hip)
    keypoints_2d_array = np.hstack((keypoints_2d_array, center_of_mass))

    np.save(out_filename_keypoints_2d, keypoints_2d_array)
    print(f"shape of keypoints_2d:  {keypoints_2d_array.shape}")


def keypoints_3d(recording_name: str):
    """2D keypoints + aligned depth -> 3D keypoints (camera frame)"""
    filename_keypoints_2d = get_path(recording_name, "_keypoints_2d.npy")
    filename_metadata = get_path(recording_name, "_metadata.json")
    filename_depth = get_path(recording_name, "_depth.mkv")
    out_filename_keypoints_3d = get_path(recording_name, "_keypoints_3d.npy")
    check_files_exist([filename_keypoints_2d, filename_metadata, filename_depth])

    keypoints_2d_array = np.load(filename_keypoints_2d)
    with open(filename_metadata) as f:
        metadata = json.load(f)

    depth_scale = metadata["depth_scale_meters"]  # conversion to meters
    intrinsics = create_intrinsics_object(intrinsics_dict=metadata["color_intrinsics"])

    W = intrinsics.width
    H = intrinsics.height
    n_frames = metadata["frames"]

    bytes_per_frame = W * H * 2  # uint16 = 2 bytes/pixel

    ffmpeg_cmd = ["ffmpeg", "-i", filename_depth, "-f", "rawvideo", "-pix_fmt", "gray16le", "-"]
    video_depth = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
    )

    keypoints_3d_per_frame = []
    frame_idx = 0
    progress_bar = tqdm(total=n_frames, desc="3D keypoints", unit="frame")
    while frame_idx < n_frames:
        raw_frame_data = video_depth.stdout.read(bytes_per_frame)
        if len(raw_frame_data) < bytes_per_frame:
            break
        current_frame_depth_map = np.frombuffer(raw_frame_data, dtype=np.uint16).reshape(H, W)
        current_frame_depth_map = current_frame_depth_map.copy()

        current_keypoints_2d = keypoints_2d_array[frame_idx, :, :]

        current_keypoints_3d = []
        # NOTE: pose estimation runs in portrait, x,y must be landscape for depth
        for kpt_idx, (x_portrait, y_portrait) in enumerate(current_keypoints_2d):
            # 90 deg clockwise rotation of the physical camera
            x_landscape = int(round(W - 1 - y_portrait))
            y_landscape = int(round(x_portrait))

            if not (0 <= x_landscape < W and 0 <= y_landscape < H):
                current_keypoints_3d.append([np.nan, np.nan, np.nan])
                continue

            d = (
                sample_depth_patch(
                    depth_map=current_frame_depth_map, x=x_landscape, y=y_landscape, radius=3
                )
                * depth_scale
            )

            if d <= 0:  # invalid/missing depth reading
                current_keypoints_3d.append([np.nan, np.nan, np.nan])
                continue

            [x, y, d] = rs.rs2_deproject_pixel_to_point(
                intrin=intrinsics, pixel=(x_landscape, y_landscape), depth=d
            )

            # NOTE: dont touch these, that will mess up spatial calibration
            current_keypoints_3d.append([x, y, d])

        keypoints_3d_per_frame.append(current_keypoints_3d)
        frame_idx += 1
        progress_bar.update(1)

    progress_bar.close()
    video_depth.stdout.close()
    depth_exit_code = video_depth.wait()

    # every later stage pairs keypoint frame i with timestamp i, so a short read must not be saved
    if depth_exit_code != 0:
        raise RuntimeError(f"ffmpeg depth decoding failed with exit code {depth_exit_code}")
    if frame_idx != n_frames:
        raise RuntimeError(f"read {frame_idx} depth frames, but the conversion wrote {n_frames} frames")

    keypoints_3d_array = np.asanyarray(keypoints_3d_per_frame)
    np.save(out_filename_keypoints_3d, keypoints_3d_array)
    print(f"shape of keypoints_3d:  {keypoints_3d_array.shape}")


def calibration(recording_name: str):
    """chessboard on calibration_image.png -> 3D keypoints in the chessboard (global) frame"""
    filename_metadata = get_path(recording_name, "_metadata.json")
    filename_keypoints_3d = get_path(recording_name, "_keypoints_3d.npy")
    filename_calibration_img = os.path.join(DATA_FOLDER, recording_name, "calibration_image.png")
    out_filename_keypoints_3d_global = get_path(recording_name, "_keypoints_3d_global.npy")
    check_files_exist([filename_metadata, filename_keypoints_3d, filename_calibration_img])

    with open(filename_metadata) as f:
        metadata = json.load(f)
    color_intrinsics = metadata["color_intrinsics"]

    keypoints_3d_array = np.load(filename_keypoints_3d)

    pattern = CALIBRATION_PATTERN
    calibration_image = cv2.imread(filename_calibration_img, cv2.IMREAD_GRAYSCALE)

    calibration_result = get_image_points(
        calibration_image=calibration_image, calibration_pattern=pattern
    )
    if calibration_result is None:
        raise RuntimeError(f"chessboard not found in {filename_calibration_img}")
    image_points, debug_image = calibration_result
    if len(image_points) != len(pattern.pattern_points):
        raise RuntimeError(
            f"found {len(image_points)} chessboard corners, expected {len(pattern.pattern_points)}"
        )
    camera_matrix = np.array(
        [
            [color_intrinsics["fx"], 0, color_intrinsics["ppx"]],
            [0, color_intrinsics["fy"], color_intrinsics["ppy"]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    distortion_coeffs = np.array(color_intrinsics["coeffs"])
    success, rvec, t_vec = cv2.solvePnP(
        objectPoints=pattern.pattern_points,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion_coeffs,
        useExtrinsicGuess=False,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError(f"solvePnP failed for {filename_calibration_img}")
    R, jac = cv2.Rodrigues(rvec)

    camera_position_global_frame = (-R.T @ t_vec).ravel()

    print(f"distance: {round(np.linalg.norm(camera_position_global_frame), 2)} m")

    projected, _ = cv2.projectPoints(
        pattern.pattern_points, rvec, t_vec, camera_matrix, distortion_coeffs
    )
    error = np.linalg.norm(projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1)

    print("mean reprojection error:", error.mean())
    print("max reprojection error:", error.max())

    keypoints_3d_global = keypoints_3d_array @ R + camera_position_global_frame

    np.save(out_filename_keypoints_3d_global, keypoints_3d_global)


def resample(recording_name: str):
    """filter, resample to 100 Hz and map onto the Qualisys axes -> _stereo.npz"""
    filename_timestamps = get_path(recording_name, "_timestamps_color.bin")
    filename_metadata = get_path(recording_name, "_metadata.json")
    filename_keypoints_3d_global = get_path(recording_name, "_keypoints_3d_global.npy")
    filename_stereo_keypoints = get_path(recording_name, "_stereo.npz")
    check_files_exist([filename_timestamps, filename_metadata, filename_keypoints_3d_global])

    keypoints_3d_global = np.load(filename_keypoints_3d_global)
    labels = LABELS
    if len(labels) != keypoints_3d_global.shape[1]:
        raise ValueError(
            f"{len(labels)} labels but {keypoints_3d_global.shape[1]} keypoints, "
            "use the label list of the pose model that produced the 2D keypoints"
        )
    frames, times, fps = get_frame_times(filename_timestamps, n_frames=len(keypoints_3d_global))
    keypoints_3d_filtered = filter_keypoints_stereo(
        keypoints_3d_global[frames], sampling_rate=fps, cutoffs=(3, 3, 5)
    )
    keypoints_3d_resampled, _ = resample_keypoints_to_rate(keypoints_3d_filtered, times, 100)
    keypoints_3d_axis_swap = stereo_to_qualisys_axes(keypoints_3d_resampled)

    np.savez(filename_stereo_keypoints, labels=labels, keypoints=keypoints_3d_axis_swap)
    print(f"stereo keypoints saved to: {filename_stereo_keypoints} with shape: {keypoints_3d_axis_swap.shape}")


STAGE_FUNCTIONS = {
    "convert": convert,
    "keypoints_2d": keypoints_2d,
    "keypoints_3d": keypoints_3d,
    "calibration": calibration,
    "resample": resample,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo preprocessing pipeline")
    parser.add_argument("recording_names", nargs="+", help="e.g. tug2 (several can be given)")
    parser.add_argument(
        "--stages", nargs="+", choices=STAGES, default=STAGES, help="stages to run (default: all)"
    )
    args = parser.parse_args()

    for recording_name in args.recording_names:
        if not os.path.isdir(os.path.join(DATA_FOLDER, recording_name)):
            raise FileNotFoundError(f"Folder {os.path.join(DATA_FOLDER, recording_name)} does not exist")
        for stage in STAGES:
            if stage in args.stages:
                print(f"\n=== {recording_name}: {stage} ===")
                STAGE_FUNCTIONS[stage](recording_name)
