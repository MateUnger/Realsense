import cv2
import json
import numpy as np


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


def get_image_points(img, pattern_params: Pattern) -> tuple | None:
    """
    Find corners of chessboard, approximate them (subpixel precision)

    Args:
        img: input image
        pattern_params: parameters (height, width, ...) of the real world calibration pattern used

    Returns:
        image_points, object_points arrays for the input image
    """
    found = False
    corners = 0
    found, corners = cv2.findChessboardCorners(img, pattern_params.pattern_size)
    if found:
        print(f"corners found: {len(corners)}")
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.1)
        # refine corners
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        # points in camera/image space
        img_points = corners.reshape(-1, 2)

        # points in real world
        obj_points = pattern_params.pattern_points
    else:
        print("corners not found")
        return None

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(vis, pattern_params.pattern_size, corners, found)

    # draw real world origin on image
    vis = cv2.circle(
        vis,
        (int(img_points[0][0]), int(img_points[0][1])),
        10,
        (255, 0, 0),
        2,
    )

    return img_points, obj_points, vis


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
