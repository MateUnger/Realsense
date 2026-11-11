"""Qualisys preprocessing, taken from the "Qualisys preprocessing" cell in test.ipynb.

<name>.tsv -> get_qualisys_keypoints (gap-filled samples -> NaN, mm -> m, adds mid_shoulder / mid_hip)
           -> median + Butterworth filter at 100 Hz -> <name>_qualisys.npz

Usage (from the repo root):
    realsense_env\\Scripts\\python.exe preprocess_qualisys.py tug2
    realsense_env\\Scripts\\python.exe preprocess_qualisys.py tug1 tug2 tug3
"""

import os
import argparse
import numpy as np
from Util.util import get_qualisys_keypoints, filter_keypoints_median_butter


def preprocess_qualisys(recording_name: str):
    filename_qualisys_tsv = os.path.join("data", recording_name, recording_name + ".tsv")
    out_filename_qualisys_keypoints = os.path.join(
        "data", recording_name, recording_name + "_qualisys.npz"
    )
    if not os.path.exists(filename_qualisys_tsv):
        raise FileNotFoundError(f"File {filename_qualisys_tsv} does not exist")

    qualisys_keypoints, qualisys_labels = get_qualisys_keypoints(filename=filename_qualisys_tsv)
    qualisys_keypoints = filter_keypoints_median_butter(
        keypoints=qualisys_keypoints, sampling_rate=100, cutoff=7, order=2, gap_size=0.25
    )

    np.savez(out_filename_qualisys_keypoints, labels=qualisys_labels, keypoints=qualisys_keypoints)
    print(
        f"qualisys keypoints saved to: {out_filename_qualisys_keypoints} with shape: {qualisys_keypoints.shape}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qualisys preprocessing")
    parser.add_argument("recording_names", nargs="+", help="e.g. tug2 (several can be given)")
    args = parser.parse_args()

    for recording_name in args.recording_names:
        print(f"\n=== {recording_name}: qualisys ===")
        preprocess_qualisys(recording_name)
