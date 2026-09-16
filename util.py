import cv2
import json
import numpy as np
import time
import importlib
from typing import Literal
import pyrealsense2 as rs

rtmlib_module = importlib.import_module("rtmlib")


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

    ROI = depth_map[x_min:x_max, y_min:y_max]
    valid = ROI[ROI > 0]

    depth = float(np.median(valid)) if valid.size > 0 else np.nan

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
        self.pattern_height = height
        self.pattern_width = width
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
