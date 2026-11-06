# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Markerless 3D human motion capture with an Intel RealSense depth camera, validated against a Qualisys marker-based system. RGB frames go through 2D pose estimation (rtmlib / RTMPose / ViTPose, ONNX on CUDA), keypoints are lifted to 3D using the aligned depth map, transformed into a world frame defined by a chessboard, and then compared to Qualisys trajectories (positions, velocities, accelerations).

## Environment

- Python 3.11 venv in `realsense_env/` (gitignored). Run Python as `realsense_env\Scripts\python.exe`; do not use a system Python.
- `ffmpeg` must be on PATH (used through `subprocess` for color and depth encoding/decoding). Color encoding defaults to `h264_qsv` (Intel GPU).
- Pose models are downloaded by rtmlib from the URLs in `util.models` / `Custom.MODE`.
- No build, lint, or test setup. All work happens in Jupyter notebooks using the `ipympl` backend (`%matplotlib ipympl`).

## Code layout

- `Util/`: package holding the reusable modules (plus keypoint index reference images). Notebooks import from it as `from Util.util import ...` / `from Util.pulltest_util import ...`.
  - `Util/util.py`: the main module. All reusable pipeline functions live here. New reusable logic should go here rather than into notebook cells.
  - `Util/pulltest_util.py`: pull-test analysis helpers (pull onsets and segmentation, gait events, `Recording`), used by `PullTest.ipynb`.
- `test.ipynb`: the work-in-progress notebook where new code is developed before moving to `util.py`. It holds the main end-to-end pipeline (streaming, recording, conversion, 2D/3D keypoints, calibration, Qualisys preprocessing, comparison), followed by "Small tests" experiments (Kalman, Savitzky-Golay, outlier rejection, detector selection). Each section is a self-contained cell that sets `recording_name` and reads and writes files on disk, so stages can be rerun on their own. Passthrough stages, duplicated helpers and out-of-date functions are expected while work is in progress; don't clean them up unless asked.
- `Gait.ipynb`, `PullTest.ipynb`: analysis notebooks for gait (`w*`) and pull-test (`pq*`, `p*`) recordings. They are older than `util.py`: they redefine functions that now exist in `util.py` and call `load_keypoints_and_labels(npz_data_path=...)`, while the current parameter name is `filename`.

## Working style

- Work iteratively, starting from the simplest solution. Each step adds one change (e.g. one filter, one parameter, one new rule), and the user checks the result before the next step. Don't bundle several improvements into one edit, even if they all seem needed; list the other ideas as possible next steps instead.
- Use descriptive variable names everywhere, including loop counters and indices (e.g. `peak_number`, `peak_index`, `sample_index`, not `i`, `j`, `k`, `p`).

## Data pipeline (per recording in `data/<name>/`)

Each stage writes a file that the next stage reads. All files are named `<name>_<suffix>`:

1. `.db3`: raw RealSense recording (rosbag).
2. Filter, align and convert: depth is filtered by RealSense threshold → disparity → spatial → temporal filters, then aligned to color. Outputs `_color.mp4`, `_depth.mkv` (lossless FFV1, `gray16le`, raw units × `depth_scale_meters` = meters), `_timestamps_color.bin` and `_timestamps_depth.bin` (float64 ms, one per frame of each stream), and `_metadata.json` (fps, frame count, depth scale, color/depth intrinsics). Recordings converted before this change have a single `_timestamps.bin` holding the depth (frameset) timestamps instead, and were aligned before filtering.
3. `_keypoints_2d.npy`: pose estimation on the color video.
4. `_keypoints_2d_filtered.npy`: currently a passthrough (no filtering).
5. `_keypoints_3d.npy`: median depth patch (`sample_depth_patch`) + `rs2_deproject_pixel_to_point` using the **color** intrinsics, since depth is aligned to color.
6. `_keypoints_3d_filtered.npy`: currently a passthrough.
7. Calibration: `solvePnP` on `calibration_image.png` (chessboard `Pattern(height=6, width=4, square_size=0.15)`). Gives `_keypoints_3d_global.npy` and `_stereo.npz` (`keypoints`, `labels`).
8. Qualisys: `<name>.tsv` export → `get_qualisys_keypoints` (gap-filled samples become NaN, mm → m, adds `mid_shoulder` / `mid_hip`) → filtered at 100 Hz → `_qualisys.npz`.

Data files (`*.npy`, `*.npz`, `*.json`, `*.mp4`, `*.tsv`, ...) are gitignored.

## Conventions and gotchas

- Keypoint arrays are always shaped `(frames, keypoints, dims)` and use NaN for missing values. Filters must keep NaN gaps intact: `filter_keypoints_median_butter` interpolates gaps shorter than `gap_size` seconds and filters each contiguous valid segment separately.
- The camera is physically rotated 90°. Pose estimation runs on portrait frames, but depth and intrinsics are landscape. Step 5 maps `x_landscape = y_portrait`, `y_landscape = H - 1 - x_portrait`. Don't change this mapping or the axis order of the 3D output; the comment in the code warns that doing so breaks the spatial calibration.
- Sampling rates differ: RealSense nominally runs at 60/90 fps (`metadata["fps"]`), but actually at 59.755 / 89.868 fps; using the nominal rate drifts ~0.25 s/min against Qualisys (100 Hz). `get_frame_times` cleans `_timestamps_color.bin` (startup dropout, duplicated frames) and returns the frames to keep, their times and the actual fps; `resample_keypoints_to_rate` resamples onto a 100 Hz grid with those times (`resample_keypoints` assumes a round rate ratio). The two streams are aligned with hand-picked offsets (`stereo_cut`, `qualisys_cut`), not timestamps.
- Axis order: in the shared frame the axes are (horizontal, depth, vertical), so depth is index 1 and vertical (up positive) is index 2. The stereo chessboard frame always points its vertical axis down, but its horizontal axes depend on where the camera and chessboard are placed. At the end of stereo preprocessing, `stereo_to_qualisys_axes` maps the stereo keypoints onto the Qualisys axes with `axis_order` / `axis_signs`. The current setup is the `tug1` one, which is also the default: order `(0, 1, 2)`, signs `(-1, 1, -1)` (no swap, flip the first horizontal axis and the vertical axis). The `pq1` mapping (swap the two horizontal axes) belongs to an older, test-phase setup. After that, both systems share the same axes and can be indexed the same way.
- Label sets: `labels_17` and `labels_26` (Halpe26), each extended with `mid_shoulder` / `mid_hip`. Index keypoints with `labels.index(name)`.
- `Util.util.get_extrinsics` is stale: it reads `./calibration_metadata.json` and unpacks three values from `get_image_points`, which returns two. The working calibration code is the calibration cell in `test.ipynb`.
