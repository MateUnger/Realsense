from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from Util.util import (
    get_segment_vectors,
    angle_between_vectors,
    get_ankle_speed,
    get_gait_events,
    get_steps,
    Step,
)


def get_fft_frequencies_and_amplitudes(
    signal: np.ndarray, sampling_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided amplitude spectrum of each column of a signal, after removing its mean.
    NaN samples are treated as 0.

    Args:
        signal: shape (n_samples,) or (n_samples, n_channels)
        sampling_rate: sampling rate of the signal (Hz)
    Returns:
        frequencies (Hz), shape (n_frequencies,)
        amplitudes, same unit as the signal, shape (n_frequencies,) or (n_frequencies, n_channels)
    """
    signal_centered = signal - np.nanmean(signal, axis=0)
    frequencies = np.fft.rfftfreq(len(signal_centered), d=1 / sampling_rate)
    amplitudes = (
        2 / len(signal_centered) * np.abs(np.fft.rfft(np.nan_to_num(signal_centered), axis=0))
    )
    return frequencies, amplitudes


def get_psd(
    signal: np.ndarray, sampling_rate: float, segment_duration: float = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Power spectral density of each column of a signal (Welch's method, Hann window, 50% overlap),
    after removing its mean. NaN samples are treated as 0.

    Args:
        signal: shape (n_samples,) or (n_samples, n_channels)
        sampling_rate: sampling rate of the signal (Hz)
        segment_duration: length (s) of the Welch segments; the frequency resolution is
            1 / segment_duration. Signals shorter than this use one segment of their full length.
    Returns:
        frequencies (Hz), shape (n_frequencies,)
        psd, signal unit^2 / Hz, shape (n_frequencies,) or (n_frequencies, n_channels)
    """
    signal_centered = signal - np.nanmean(signal, axis=0)
    samples_per_segment = min(round(segment_duration * sampling_rate), len(signal_centered))
    frequencies, psd = welch(
        np.nan_to_num(signal_centered), fs=sampling_rate, nperseg=samples_per_segment, axis=0
    )
    return frequencies, psd


def get_spectral_parameters(
    frequencies: np.ndarray, psd: np.ndarray, max_frequency: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Peak frequency, the frequencies below which 50% (F50) and 80% (F80) of the power lies, the
    PSD value of the largest peak, and the mean frequency (power-weighted average frequency),
    using only frequencies up to max_frequency.

    Args:
        frequencies: output of get_psd
        psd: output of get_psd
        max_frequency: upper limit (Hz) of the band used
    Returns:
        peak frequency, F50, F80 (Hz), peak PSD (signal unit^2 / Hz), mean frequency (Hz),
        one value per channel
    """
    in_band = frequencies <= max_frequency
    frequencies = frequencies[in_band]
    power = psd[in_band]
    cumulative_power_fraction = np.cumsum(power, axis=0) / np.sum(power, axis=0)

    peak_frequency = frequencies[np.argmax(power, axis=0)]
    f50 = frequencies[np.argmax(cumulative_power_fraction >= 0.5, axis=0)]
    f80 = frequencies[np.argmax(cumulative_power_fraction >= 0.8, axis=0)]
    peak_psd = np.max(power, axis=0)
    # frequencies has shape (n_frequencies,), power may have a channel axis: broadcast over it
    frequency_column = frequencies.reshape(-1, *([1] * (power.ndim - 1)))
    mean_frequency = np.sum(frequency_column * power, axis=0) / np.sum(power, axis=0)
    return peak_frequency, f50, f80, peak_psd, mean_frequency


def get_trunk_angles(keypoints: np.ndarray, labels: list[str], baseline_end: int) -> np.ndarray:
    """
    Angle (degrees) of the trunk (mid_shoulder -> mid_hip) in every frame, relative to the
    mean trunk vector of the baseline.

    Args:
        keypoints: (frames, keypoints, 3) positions in meters
        labels: keypoint labels
        baseline_end: the baseline is frames 0 to baseline_end
    Returns:
        trunk angles, shape (n_frames,)
    """
    trunk_vectors = get_segment_vectors(keypoints, labels, "mid_shoulder", "mid_hip")
    trunk_baseline = np.nanmean(trunk_vectors[:baseline_end], axis=0)
    return np.array(
        [angle_between_vectors(trunk_vector, trunk_baseline) for trunk_vector in trunk_vectors]
    )


def get_arm_angles(
    keypoints: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shoulder angles (upper arm vs trunk, mid_shoulder -> mid_hip) and elbow angles (upper arm
    vs lower arm) in degrees, in every frame.

    Args:
        keypoints: (frames, keypoints, 3) positions in meters
        labels: keypoint labels
    Returns:
        left shoulder, right shoulder, left elbow, right elbow angles, each shape (n_frames,)
    """
    trunk_vectors = get_segment_vectors(keypoints, labels, "mid_shoulder", "mid_hip")
    left_upper_arm = get_segment_vectors(keypoints, labels, "left_shoulder", "left_elbow")
    left_lower_arm = get_segment_vectors(keypoints, labels, "left_elbow", "left_wrist")
    right_upper_arm = get_segment_vectors(keypoints, labels, "right_shoulder", "right_elbow")
    right_lower_arm = get_segment_vectors(keypoints, labels, "right_elbow", "right_wrist")

    left_shoulder_angles = np.array(
        [
            angle_between_vectors(upper_arm, trunk)
            for (upper_arm, trunk) in zip(left_upper_arm, trunk_vectors)
        ]
    )
    right_shoulder_angles = np.array(
        [
            angle_between_vectors(upper_arm, trunk)
            for (upper_arm, trunk) in zip(right_upper_arm, trunk_vectors)
        ]
    )
    left_elbow_angles = np.array(
        [
            angle_between_vectors(upper_arm, lower_arm)
            for (upper_arm, lower_arm) in zip(left_upper_arm, left_lower_arm)
        ]
    )
    right_elbow_angles = np.array(
        [
            angle_between_vectors(upper_arm, lower_arm)
            for (upper_arm, lower_arm) in zip(right_upper_arm, right_lower_arm)
        ]
    )
    return left_shoulder_angles, right_shoulder_angles, left_elbow_angles, right_elbow_angles


def get_com_positions(
    keypoints: np.ndarray, labels: list[str], eyes_open_end: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Horizontal center of mass position (medio-lateral, anterior-posterior) of the eyes open and
    the eyes closed phase, each centered on its own mean. Plotted against each other they give
    the statokinesigram, plotted against time the stabilogram.

    Args:
        keypoints: (frames, keypoints, 3) positions in meters
        labels: keypoint labels, must contain "center_of_mass"
        eyes_open_end: the eyes open phase is frames 0 to eyes_open_end, the eyes closed phase
            the rest
    Returns:
        eyes open positions, shape (n_eyes_open_frames, 2)
        eyes closed positions, shape (n_eyes_closed_frames, 2)
    """
    com_positions = keypoints[:, labels.index("center_of_mass"), :2]
    eyes_open_positions = com_positions[:eyes_open_end]
    eyes_closed_positions = com_positions[eyes_open_end:]
    eyes_open_positions = eyes_open_positions - np.nanmean(eyes_open_positions, axis=0)
    eyes_closed_positions = eyes_closed_positions - np.nanmean(eyes_closed_positions, axis=0)
    return eyes_open_positions, eyes_closed_positions


class BalanceRecording:
    """
    One balance recording from one system.

    Attributes:
        system: "qualisys" or "stereo"
        keypoints: (frames, keypoints, 3) positions in meters
        labels: keypoint labels
        fps: sampling rate of the keypoints
        eyes_open_baseline_duration: length (s) of the eyes open phase at the start of the
            recording; the eyes closed phase lasts from its end to the end of the recording
        trunk_angles: (frames,) trunk angle (deg), see get_trunk_angles
        left_shoulder_angles, right_shoulder_angles, left_elbow_angles, right_elbow_angles:
            (frames,) arm angles (deg), see get_arm_angles
        eyes_open_com_positions, eyes_closed_com_positions: (frames, 2) centered horizontal center
            of mass position of each phase, see get_com_positions
        eyes_open_frequencies, eyes_open_psd, eyes_closed_frequencies, eyes_closed_psd: center of
            mass PSD of each phase, see get_psd
        eyes_open_spectral_parameters, eyes_closed_spectral_parameters: (peak frequency, F50, F80,
            peak PSD, mean frequency) of each phase, one value per horizontal axis, see
            get_spectral_parameters
        com_std: (2,) root mean square deviation (m) of the eyes closed center of mass position
            from the mean position of the eyes open phase, per horizontal axis
        total_sway_path: length (m) of the 2D horizontal path the center of mass travels during
            the eyes closed phase
        steps: steps of both feet, see get_steps
    """

    def __init__(
        self,
        system: Literal["qualisys", "stereo"],
        keypoints: np.ndarray,
        labels: list[str],
        fps: float = 100,
        eyes_open_baseline_duration: float = 3,
    ):
        self.system = system
        self.keypoints = keypoints
        self.labels = labels
        self.fps = fps
        self.eyes_open_baseline_duration = eyes_open_baseline_duration
        self.trunk_angles: np.ndarray | None = None
        self.left_shoulder_angles: np.ndarray | None = None
        self.right_shoulder_angles: np.ndarray | None = None
        self.left_elbow_angles: np.ndarray | None = None
        self.right_elbow_angles: np.ndarray | None = None
        self.eyes_open_com_positions: np.ndarray | None = None
        self.eyes_closed_com_positions: np.ndarray | None = None
        self.eyes_open_frequencies: np.ndarray | None = None
        self.eyes_open_psd: np.ndarray | None = None
        self.eyes_closed_frequencies: np.ndarray | None = None
        self.eyes_closed_psd: np.ndarray | None = None
        self.eyes_open_spectral_parameters: (
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
        ) = None
        self.eyes_closed_spectral_parameters: (
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
        ) = None
        self.com_std: np.ndarray | None = None
        self.total_sway_path: float | None = None
        self.steps: list[Step] = []

    def analyze(
        self,
        segment_duration: float = 10,
        max_frequency: float = 5,
        speed_threshold: float = 0.5,
        min_swing_time: float = 0.2,
        min_stance_time: float = 0.2,
        min_step_length: float = 0.1,
    ):
        """
        Full pipeline: trunk and arm angles, center of mass positions -> PSD -> spectral
        parameters of each phase, and steps. Fills the result attributes.

        Args:
            segment_duration: length (s) of the Welch segments of the PSD
            max_frequency: upper limit (Hz) of the band used for the spectral parameters
            speed_threshold: ankle speed (m/s) below which the foot counts as on the ground
            min_swing_time: swing periods shorter than this (s) are treated as ground contact
            min_stance_time: stance periods shorter than this (s) are treated as swing
            min_step_length: steps shorter than this (m) are dropped as shuffles
        """
        eyes_open_end = round(self.eyes_open_baseline_duration * self.fps)

        self.trunk_angles = get_trunk_angles(self.keypoints, self.labels, eyes_open_end)
        (
            self.left_shoulder_angles,
            self.right_shoulder_angles,
            self.left_elbow_angles,
            self.right_elbow_angles,
        ) = get_arm_angles(self.keypoints, self.labels)

        self.eyes_open_com_positions, self.eyes_closed_com_positions = get_com_positions(
            self.keypoints, self.labels, eyes_open_end
        )
        # like a standard deviation, but around the eyes open mean instead of the eyes closed mean;
        # uses the raw positions, since the stored ones are centered on their own phase
        com_positions = self.keypoints[:, self.labels.index("center_of_mass"), :2]
        eyes_open_mean_position = np.nanmean(com_positions[:eyes_open_end], axis=0)
        self.com_std = np.sqrt(
            np.nanmean((com_positions[eyes_open_end:] - eyes_open_mean_position) ** 2, axis=0)
        )
        # sum of the 2D distances between consecutive frames; steps next to a NaN frame are skipped
        eyes_closed_relative_positions = com_positions[eyes_open_end:] - eyes_open_mean_position
        frame_to_frame_distances = np.linalg.norm(
            np.diff(eyes_closed_relative_positions, axis=0), axis=1
        )
        self.total_sway_path = float(np.nansum(frame_to_frame_distances))
        self.eyes_open_frequencies, self.eyes_open_psd = get_psd(
            self.eyes_open_com_positions, self.fps, segment_duration
        )
        self.eyes_closed_frequencies, self.eyes_closed_psd = get_psd(
            self.eyes_closed_com_positions, self.fps, segment_duration
        )
        self.eyes_open_spectral_parameters = get_spectral_parameters(
            self.eyes_open_frequencies, self.eyes_open_psd, max_frequency
        )
        self.eyes_closed_spectral_parameters = get_spectral_parameters(
            self.eyes_closed_frequencies, self.eyes_closed_psd, max_frequency
        )

        # steps typed relative to the mean ankle position of the eyes open phase
        ankle_indices = [self.labels.index("left_ankle"), self.labels.index("right_ankle")]
        baseline_ankle_position = np.nanmean(
            self.keypoints[:eyes_open_end, ankle_indices, :], axis=(0, 1)
        )
        gait_events = get_gait_events(
            self.keypoints,
            self.labels,
            fps=self.fps,
            speed_threshold=speed_threshold,
            min_swing_time=min_swing_time,
            min_stance_time=min_stance_time,
        )
        self.steps = get_steps(
            gait_events, self.keypoints, self.labels, min_step_length=min_step_length
        )
        for step in self.steps:
            step.set_type(self.keypoints, self.labels, baseline_ankle_position)

    def plot_angles(self):
        """
        Trunk angle, shoulder angles, elbow angles, and both ankle speeds with the steps shaded
        (left steps in the upper half of the panel, right steps in the lower half). The dashed
        line marks the end of the eyes open phase. Call analyze() first.

        Returns:
            the figure
        """
        if self.trunk_angles is None:
            raise ValueError("call analyze() before plot_angles()")
        step_colors = {"corrective": "tab:orange", "return": "tab:green"}
        frames = np.arange(len(self.keypoints))

        fig, axs = plt.subplots(4, 1, figsize=(15, 9), sharex=True)
        fig.suptitle(self.system)

        axs[0].set_title("trunk angle")
        axs[0].plot(frames, self.trunk_angles, label="trunk")
        axs[0].set_ylabel("deg")

        axs[1].set_title("shoulder angles")
        axs[1].plot(frames, self.left_shoulder_angles, label="left")
        axs[1].plot(frames, self.right_shoulder_angles, label="right")
        axs[1].set_ylabel("deg")

        axs[2].set_title("elbow angles")
        axs[2].plot(frames, self.left_elbow_angles, label="left")
        axs[2].plot(frames, self.right_elbow_angles, label="right")
        axs[2].set_ylabel("deg")

        axs[3].set_title("ankle speed and steps")
        axs[3].plot(
            frames, get_ankle_speed(self.keypoints, self.labels, "left", self.fps), label="left"
        )
        axs[3].plot(
            frames, get_ankle_speed(self.keypoints, self.labels, "right", self.fps), label="right"
        )
        for step in self.steps:
            step_label = f"{step.type} step" if step.type is not None else "step"
            span_bottom, span_top = (0.5, 1) if step.side == "left" else (0, 0.5)
            axs[3].axvspan(
                step.FC.frame,
                step.IC.frame,
                ymin=span_bottom,
                ymax=span_top,
                color=step_colors.get(step.type, "gray"),
                alpha=0.3,
                label=step_label,
            )
        axs[3].set_ylabel("m/s")

        eyes_open_end = round(self.eyes_open_baseline_duration * self.fps)
        for ax in axs:
            ax.axvline(eyes_open_end, color="k", linestyle="--", alpha=0.4, label="eyes closed")
            # one legend entry per label, outside the axes so it doesn't cover the data
            handles, labels = ax.get_legend_handles_labels()
            unique_entries = dict(zip(labels, handles))
            ax.legend(
                unique_entries.values(),
                unique_entries.keys(),
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                borderaxespad=0,
            )
            ax.grid(alpha=0.3)
        axs[-1].set_xlabel("frame")
        plt.tight_layout()
        return fig

    def plot_com(self):
        """
        Statokinesigram (anterior-posterior vs medio-lateral center of mass position) of both
        phases, and stabilogram (medio-lateral and anterior-posterior position over time). Each
        phase is centered on its own mean. The dashed line marks the end of the eyes open phase.
        Call analyze() first.

        Returns:
            the figure
        """
        if self.eyes_open_com_positions is None:
            raise ValueError("call analyze() before plot_com()")
        eyes_open_end = len(self.eyes_open_com_positions)
        eyes_open_frames = np.arange(eyes_open_end)
        eyes_closed_frames = eyes_open_end + np.arange(len(self.eyes_closed_com_positions))

        fig, axs = plt.subplots(1, 2, figsize=(15, 5), width_ratios=[1, 2])
        fig.suptitle(self.system)

        axs[0].set_title("statokinesigram")
        axs[0].plot(
            self.eyes_closed_com_positions[:, 0],
            self.eyes_closed_com_positions[:, 1],
            color="tab:blue",
            label="eyes closed",
        )
        axs[0].plot(
            self.eyes_open_com_positions[:, 0],
            self.eyes_open_com_positions[:, 1],
            color="tab:orange",
            label="eyes open",
        )
        axs[0].axhline(0, color="k", linestyle="--", alpha=0.4)
        axs[0].axvline(0, color="k", linestyle="--", alpha=0.4)
        axs[0].set_aspect("equal", adjustable="datalim")
        axs[0].set_xlabel("medio-lateral (m)")
        axs[0].set_ylabel("anterior-posterior (m)")

        axs[1].set_title("stabilogram")
        for axis_index, (axis_name, color) in enumerate(
            [("medio-lateral", "tab:green"), ("anterior-posterior", "tab:red")]
        ):
            axs[1].plot(
                eyes_open_frames,
                self.eyes_open_com_positions[:, axis_index],
                color=color,
                label=axis_name,
            )
            axs[1].plot(
                eyes_closed_frames, self.eyes_closed_com_positions[:, axis_index], color=color
            )
        axs[1].axhline(0, color="k", linestyle="--", alpha=0.4)
        axs[1].axvline(eyes_open_end, color="k", linestyle="--", alpha=0.4, label="eyes closed")
        axs[1].set_xlabel("frame")
        axs[1].set_ylabel("m")

        for ax in axs:
            # outside the axes, to the right, so it doesn't cover the data
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def __repr__(self):
        return (
            f"BalanceRecording(system={self.system!r}, frames={len(self.keypoints)}, "
            f"fps={self.fps}, eyes_open_baseline_duration={self.eyes_open_baseline_duration}, "
            f"steps={len(self.steps)})"
        )
