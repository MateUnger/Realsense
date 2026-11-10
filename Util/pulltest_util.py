import numpy as np
from dataclasses import dataclass, field
from typing import Literal
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from Util.util import fill_short_runs


def get_axis_velocity(keypoint, axis=None, fps=100):
    delta_t = 1 / fps  # frames/sec
    if axis is not None:
        keypoint = keypoint[:, axis]  # point, meters
    velocity = np.gradient(keypoint, delta_t, axis=0)  # vector, m/s
    return velocity


def get_pull_onsets(velocity, fps=100, height=0.5, min_interval=1.5):
    """Peaks of the velocity along the pull axis and the zero crossings before them.
    height: minimum peak velocity (m/s); min_interval: minimum time between pulls (s).
    Returns peaks and left_bases (pull onsets) as sample indices; each left base is the sample
    closest to the zero crossing, which peak_widths interpolates between samples."""
    peak_indices, _ = find_peaks(velocity, height=height, distance=int(min_interval * fps))

    # with prominence = peak velocity and rel_height=1, peak_widths measures the width at velocity 0;
    # the search limits are the whole signal, so it stops only at the first sample <= 0
    prominences = velocity[peak_indices]
    left_search_limits = np.zeros_like(peak_indices)
    right_search_limits = np.full_like(peak_indices, len(velocity) - 1)
    _, _, left_bases, right_bases = peak_widths(
        velocity,
        peak_indices,
        rel_height=1.0,
        prominence_data=(prominences, left_search_limits, right_search_limits),
    )
    left_base_indices = np.rint(left_bases).astype(int)
    right_base_indices = np.rint(right_bases).astype(int)
    return peak_indices, left_base_indices, right_base_indices


def segment_pulls(pull_onsets, sample_count, fps=100, baseline_time=0.5):
    """One Pull per onset. Each segment starts baseline_time before its onset and ends where
    the next segment starts; the last one ends at the end of the recording."""
    baseline_samples = round(baseline_time * fps)
    segment_starts = [max(0, int(onset) - baseline_samples) for onset in pull_onsets]
    segment_ends = segment_starts[1:] + [sample_count]
    return [
        Pull(start=segment_start, end=segment_end, pull_onset=int(onset))
        for onset, segment_start, segment_end in zip(pull_onsets, segment_starts, segment_ends)
    ]


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


class Pull:
    """
    One pull of a pull test: the segment of the recording from shortly before the pull onset up to
    the start of the next pull, with the steps taken in it and the per-pull metrics.

    Attributes:
        start: first frame of the segment (pull onset minus the baseline period)
        end: first frame after the segment (start of the next pull or end of the recording)
        pull_onset: frame where the pull starts
        pull_peak_frame: frame of the peak mid shoulder velocity along the pull axis, set by
            set_pull_magnitude
        pull_magnitude: peak mid shoulder velocity (m/s) along the pull axis, set by
            set_pull_magnitude
        baseline_ankle_position: (3,) mean position of both ankles from start up to pull_onset,
            set by set_baseline_ankle_position
        steps: steps whose FC falls between pull_onset and end, set by set_steps
        first_corrective_step_latency: time (s) from pull_onset to the FC of the first corrective
            step, set by set_first_corrective_step_latency
        corrective_step_count: number of corrective steps (both feet), set by
            set_corrective_step_count
        baseline_trunk: (3,) mean trunk vector (mid_hip -> mid_shoulder) from start up to
            pull_onset, set by set_baseline_trunk
        max_trunk_angle: largest angle (deg) between the trunk and baseline_trunk from pull_onset
            up to end, set by set_max_trunk_angle
        max_trunk_angle_frame: frame of max_trunk_angle, set by set_max_trunk_angle
        recovery_time: time (s) from pull_onset until the trunk angle falls back to a fraction
            (default 75%) of max_trunk_angle after the maximum, set by set_recovery_time
        first_corrective_step, first_corrective_step_length, first_corrective_step_velocity:
            properties computed from steps
    """

    def __init__(self, start, end, pull_onset):
        self.start = start
        self.end = end
        self.pull_onset = pull_onset
        self.pull_peak_frame: int | None = None  # set by set_pull_magnitude
        self.pull_magnitude: float | None = None  # m/s, set by set_pull_magnitude
        self.baseline_ankle_position: np.ndarray | None = (
            None  # (3,), set by set_baseline_ankle_position
        )
        self.steps: list[Step] = []  # steps whose FC falls after the pull onset, set by set_steps
        self.first_corrective_step_latency: float | None = (
            None  # s, set by set_first_corrective_step_latency
        )
        self.corrective_step_count: int | None = None  # set by set_corrective_step_count
        self.baseline_trunk: np.ndarray | None = None  # (3,), set by set_baseline_trunk
        self.max_trunk_angle: float | None = None  # deg, set by set_max_trunk_angle
        self.max_trunk_angle_frame: int | None = None  # set by set_max_trunk_angle
        self.recovery_time: float | None = None  # s, set by set_recovery_time

    def set_pull_magnitude(self, shoulder_velocity: np.ndarray, peak_frame: int):
        """
        Pull magnitude: mid shoulder velocity along the pull axis at the pull's velocity peak.

        Args:
            shoulder_velocity: (frames,) mid shoulder velocity (m/s) along the pull axis of the
                whole recording, e.g. from get_axis_velocity
            peak_frame: frame of this pull's velocity peak, from get_pull_onsets
        """
        self.pull_peak_frame = int(peak_frame)
        self.pull_magnitude = float(shoulder_velocity[peak_frame])

    def set_baseline_ankle_position(self, keypoints: np.ndarray, labels: list[str]):
        """
        Average position of the two ankles during the baseline period (start up to the pull onset).

        Args:
            keypoints: (frames, keypoints, 3) of the whole recording, indexed with recording frames
                like start and pull_onset
            labels: keypoint labels, must contain "left_ankle" and "right_ankle"
        """
        ankle_indices = [labels.index("left_ankle"), labels.index("right_ankle")]
        baseline_positions = keypoints[
            self.start : self.pull_onset, ankle_indices, :
        ]  # (frames, 2, 3)
        # nanmean, so frames where an ankle is missing don't make the whole baseline NaN
        self.baseline_ankle_position = np.nanmean(baseline_positions, axis=(0, 1))

    def set_steps(self, steps: list[Step], keypoints: np.ndarray, labels: list[str]):
        """
        Keep the steps whose FC falls after the pull onset (pull_onset <= FC frame < end) and
        classify each as corrective or return relative to the baseline ankle position. Steps during
        the baseline period are left out.

        Args:
            steps: steps of the whole recording, e.g. from get_steps
            keypoints: (frames, keypoints, 3) of the whole recording
            labels: keypoint labels, must contain "left_ankle" and "right_ankle"
        """
        if self.baseline_ankle_position is None:
            raise ValueError("call set_baseline_ankle_position before set_steps")
        self.steps = [step for step in steps if self.pull_onset <= step.FC.frame < self.end]
        for step in self.steps:
            step.set_type(keypoints, labels, self.baseline_ankle_position)

    def set_first_corrective_step_latency(self, fps: float) -> None:
        """Time (s) from the pull onset to the lift-off (FC) of the first corrective step;
        None if the pull has no corrective step."""
        if self.first_corrective_step is None:
            self.first_corrective_step_latency = None
            return
        self.first_corrective_step_latency = (
            self.first_corrective_step.FC.frame - self.pull_onset
        ) / fps

    def set_corrective_step_count(self) -> None:
        """Number of corrective steps in this pull (both feet)."""
        self.corrective_step_count = sum(step.type == "corrective" for step in self.steps)

    def set_baseline_trunk(self, keypoints: np.ndarray, labels: list[str]):
        """
        Mean trunk vector (mid_hip -> mid_shoulder) during the baseline period (start up to the
        pull onset).

        Args:
            keypoints: (frames, keypoints, 3) of the whole recording, indexed with recording frames
                like start and pull_onset
            labels: keypoint labels, must contain "mid_hip" and "mid_shoulder"
        """
        baseline_trunk_vectors = get_trunk_vectors(keypoints, labels)[self.start : self.pull_onset]
        # nanmean, so frames where a keypoint is missing don't make the whole baseline NaN
        self.baseline_trunk = np.nanmean(baseline_trunk_vectors, axis=0)

    def get_trunk_angles(self, keypoints: np.ndarray, labels: list[str]) -> np.ndarray:
        """
        3D angle (deg) between the trunk vector and the baseline trunk vector for every frame from
        the pull onset up to the end of the segment; NaN where the trunk is missing.

        Args:
            keypoints: (frames, keypoints, 3) of the whole recording, indexed with recording frames
                like pull_onset and end
            labels: keypoint labels, must contain "mid_hip" and "mid_shoulder"
        Returns:
            (end - pull_onset,) angles; index 0 is the pull onset frame
        """
        if self.baseline_trunk is None:
            raise ValueError("call set_baseline_trunk before get_trunk_angles")
        trunk_vectors = get_trunk_vectors(keypoints, labels)[self.pull_onset : self.end]
        cosines = (trunk_vectors @ self.baseline_trunk) / (
            np.linalg.norm(trunk_vectors, axis=1) * np.linalg.norm(self.baseline_trunk)
        )
        # clip, since rounding can push the cosine just outside [-1, 1] where arccos is NaN
        return np.degrees(np.arccos(np.clip(cosines, -1, 1)))

    def set_max_trunk_angle(self, keypoints: np.ndarray, labels: list[str]):
        """
        Largest trunk angle (deg, see get_trunk_angles) and its frame; None if the trunk is missing
        in every frame.

        Args:
            keypoints: (frames, keypoints, 3) of the whole recording, indexed with recording frames
                like pull_onset and end
            labels: keypoint labels, must contain "mid_hip" and "mid_shoulder"
        """
        trunk_angles = self.get_trunk_angles(keypoints, labels)
        if np.all(np.isnan(trunk_angles)):
            self.max_trunk_angle = None
            self.max_trunk_angle_frame = None
            return
        max_angle_index = int(np.nanargmax(trunk_angles))
        self.max_trunk_angle = float(trunk_angles[max_angle_index])
        self.max_trunk_angle_frame = self.pull_onset + max_angle_index

    def set_recovery_time(
        self,
        keypoints: np.ndarray,
        labels: list[str],
        fps: float,
        recovery_fraction: float = 0.75,
    ):
        """
        Time (s) from the pull onset to the first frame after the maximum where the trunk angle is
        back to recovery_fraction of the maximal angle or less; None if that doesn't happen before
        the end of the segment.

        Args:
            keypoints: (frames, keypoints, 3) of the whole recording, indexed with recording frames
                like pull_onset and end
            labels: keypoint labels, must contain "mid_hip" and "mid_shoulder"
            fps: sampling rate of the keypoints
            recovery_fraction: fraction of the maximal angle that counts as recovered
        """
        if self.max_trunk_angle_frame is None:
            self.recovery_time = None
            return
        trunk_angles = self.get_trunk_angles(keypoints, labels)
        max_angle_index = self.max_trunk_angle_frame - self.pull_onset
        # NaN angles fail the comparison, so missing frames never count as recovered
        recovered_indices = np.flatnonzero(
            trunk_angles[max_angle_index:] <= recovery_fraction * self.max_trunk_angle
        )
        if len(recovered_indices) == 0:
            self.recovery_time = None
            return
        recovery_index = max_angle_index + int(recovered_indices[0])
        self.recovery_time = recovery_index / fps

    @property
    def first_corrective_step(self) -> Step | None:
        """Corrective step with the earliest lift-off (FC); None if the pull has none."""
        corrective_steps = [step for step in self.steps if step.type == "corrective"]
        if not corrective_steps:
            return None
        return min(corrective_steps, key=lambda step: step.FC.frame)

    @property
    def first_corrective_step_length(self) -> float | None:
        """Step length (m) of the first corrective step; None if the pull has none."""
        if self.first_corrective_step is None:
            return None
        return self.first_corrective_step.step_length

    @property
    def first_corrective_step_velocity(self) -> float | None:
        """Step velocity (m/s) of the first corrective step; None if the pull has none."""
        if self.first_corrective_step is None:
            return None
        return self.first_corrective_step.step_velocity

    def __repr__(self):
        # short form, used when a list of pulls is displayed
        return f"Pull(start={self.start}, end={self.end}, pull_onset={self.pull_onset}, steps={len(self.steps)})"

    def __str__(self):
        # full form, used by print(pull): every attribute and property, then one row per step
        def format_value(value, unit=""):
            return "not set" if value is None else f"{value:.2f}{unit}"

        def format_vector(vector):
            return "not set" if vector is None else np.array2string(vector, precision=3)

        lines = [
            f"Pull: frames {self.start}-{self.end}, onset {self.pull_onset}",
            f"  pull magnitude: {format_value(self.pull_magnitude, ' m/s')}, frame {self.pull_peak_frame}",
            f"  baseline ankle position: {format_vector(self.baseline_ankle_position)}",
            f"  baseline trunk: {format_vector(self.baseline_trunk)}",
            f"  max trunk angle: {format_value(self.max_trunk_angle, ' deg')}, frame {self.max_trunk_angle_frame}",
            f"  recovery time: {format_value(self.recovery_time, ' s')}",
            f"  steps: {len(self.steps)}",
            f"  corrective step count: {self.corrective_step_count}",
            f"  first corrective step latency: {format_value(self.first_corrective_step_latency, ' s')}",
            f"  first corrective step length: {format_value(self.first_corrective_step_length, ' m')}",
            f"  first corrective step velocity: {format_value(self.first_corrective_step_velocity, ' m/s')}",
        ]
        if self.steps:
            lines.append(
                f"    {'type':<12}{'side':<7}{'FC':>6}{'IC':>6}{'length':>8}{'velocity':>10}"
            )
            for step in self.steps:
                length_text = "-" if step.step_length is None else f"{step.step_length:.2f}"
                velocity_text = "-" if step.step_velocity is None else f"{step.step_velocity:.2f}"
                lines.append(
                    f"    {str(step.type):<12}{step.side:<7}{step.FC.frame:>6}{step.IC.frame:>6}"
                    f"{length_text:>8}{velocity_text:>10}"
                )
        return "\n".join(lines)


class Recording:
    """
    One pull-test recording from one system, with everything derived from it.

    Attributes:
        system: "qualisys" or "stereo"
        keypoints: (frames, keypoints, 3) positions in meters; all frame numbers below index into it
        labels: keypoint labels
        fps: sampling rate of the keypoints
        pull_onsets: sample index of each pull onset
        pull_peaks: sample index of each pull's peak mid shoulder velocity
        gait_events: gait events of the whole recording
        steps: steps of the whole recording
        pulls: one Pull per pull onset
        pull_axis, speed_threshold, min_swing_time, min_stance_time: the values analyze() used,
            kept for plotting
    """

    def __init__(
        self,
        system: Literal["qualisys", "stereo"],
        keypoints: np.ndarray,
        labels: list[str],
        fps: float = 100,
    ):
        self.system = system
        self.keypoints = keypoints
        self.labels = labels
        self.fps = fps
        self.pull_onsets: np.ndarray | None = None
        self.pull_peaks: np.ndarray | None = None
        self.gait_events: list[GaitEvent] = []
        self.steps: list[Step] = []
        self.pulls: list[Pull] = []
        self.pull_axis: int = 0
        self.speed_threshold: float | None = None
        self.min_swing_time: float | None = None
        self.min_stance_time: float | None = None

    def analyze(
        self,
        pull_axis: int = 0,
        baseline_time: float = 0.5,
        speed_threshold: float = 0.3,
        min_swing_time: float = 0.3,
        min_stance_time: float = 0.3,
        min_step_length: float = 0.1,
    ):
        """
        Full pipeline: pull onsets -> pull segments -> gait events -> steps -> each pull's baseline
        and classified steps. Fills pull_onsets, gait_events, steps and pulls.

        Args:
            pull_axis: axis of the mid shoulder velocity used to detect pulls (0 = depth)
            baseline_time: time (s) before each pull onset used as baseline
            speed_threshold: ankle speed (m/s) below which the foot counts as on the ground
            min_swing_time: swing periods shorter than this (s) are treated as ground contact
            min_stance_time: stance periods shorter than this (s) are treated as swing
            min_step_length: steps shorter than this (m) are dropped as shuffles
        """
        self.pull_axis = pull_axis
        self.speed_threshold = speed_threshold
        self.min_swing_time = min_swing_time
        self.min_stance_time = min_stance_time

        shoulder_velocity = get_axis_velocity(
            self.keypoints[:, self.labels.index("mid_shoulder"), :], pull_axis, self.fps
        )
        self.pull_peaks, self.pull_onsets, _ = get_pull_onsets(shoulder_velocity, fps=self.fps)
        self.pulls = segment_pulls(
            self.pull_onsets, len(self.keypoints), fps=self.fps, baseline_time=baseline_time
        )
        # segment_pulls keeps the onset order, so pulls and peaks match one to one
        for pull, peak_frame in zip(self.pulls, self.pull_peaks):
            pull.set_pull_magnitude(shoulder_velocity, peak_frame)

        self.gait_events = get_gait_events(
            self.keypoints,
            self.labels,
            fps=self.fps,
            speed_threshold=speed_threshold,
            min_swing_time=min_swing_time,
            min_stance_time=min_stance_time,
        )
        self.steps = get_steps(
            self.gait_events, self.keypoints, self.labels, min_step_length=min_step_length
        )
        for step in self.steps:
            step.set_step_velocity(self.fps)

        for pull in self.pulls:
            pull.set_baseline_ankle_position(self.keypoints, self.labels)
            pull.set_steps(self.steps, self.keypoints, self.labels)
            pull.set_first_corrective_step_latency(self.fps)
            pull.set_corrective_step_count()
            pull.set_baseline_trunk(self.keypoints, self.labels)
            pull.set_max_trunk_angle(self.keypoints, self.labels)
            pull.set_recovery_time(self.keypoints, self.labels, self.fps)

    def plot(self):
        """
        Mid shoulder 3D position, mid shoulder velocity along the pull axis with the pull onsets,
        and for each ankle its speed with its steps shaded (corrective / return) and the ground
        contact signal after each cleaning stage of the step detection.
        Dashed lines mark the start of each pull segment. Call analyze() first to see the results.

        Returns:
            the figure
        """
        step_colors = {"corrective": "tab:orange", "return": "tab:green"}
        frames = np.arange(len(self.keypoints))

        fig, axs = plt.subplots(6, 1, figsize=(15, 9), sharex=True)
        fig.suptitle(self.system)

        shoulder = self.keypoints[:, self.labels.index("mid_shoulder"), :]
        axs[0].set_title("mid shoulder position")
        axis_labels = ["depth", "vertical", "horizontal"]
        for axis_index in range(3):
            axs[0].plot(frames, shoulder[:, axis_index], label=axis_labels[axis_index])
        axs[0].set_ylabel("m")

        shoulder_velocity = get_axis_velocity(shoulder, self.pull_axis, self.fps)
        axs[1].set_title(f"mid shoulder velocity along axis {self.pull_axis}")
        axs[1].plot(frames, shoulder_velocity, label="velocity")
        axs[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        if self.pull_onsets is not None:
            axs[1].plot(
                self.pull_onsets,
                shoulder_velocity[self.pull_onsets],
                "mo",
                markersize=4,
                label="pull onset",
            )
            axs[1].plot(
                self.pull_peaks,
                shoulder_velocity[self.pull_peaks],
                "c*",
                markersize=4,
                label="peak velocity",
            )
        axs[1].set_ylabel("m/s")

        # per ankle: speed panel, then ground contact stages panel
        for ax, stages_ax, side in [(axs[2], axs[3], "left"), (axs[4], axs[5], "right")]:
            # same speed and stages as get_gait_events uses, so the panels match the detection
            ankle_speed = get_ankle_speed(self.keypoints, self.labels, side, self.fps)
            if self.speed_threshold is not None:
                stages = get_ground_contact_stages(
                    ankle_speed,
                    self.fps,
                    self.speed_threshold,
                    self.min_swing_time,
                    self.min_stance_time,
                )
                stages_ax.set_title(
                    f"{side} ankle ground contact (1 = on the ground), offset per stage"
                )
                stage_count = len(stages)
                for stage_number, (stage_name, ground_contact) in enumerate(stages.items()):
                    stages_ax.plot(
                        frames,
                        ground_contact + 2 * (stage_count - 1 - stage_number),
                        label=stage_name,
                    )
                stages_ax.set_yticks([])

            ax.set_title(f"{side} ankle speed")
            ax.plot(frames, ankle_speed, color="k", linewidth=0.8, label="speed")
            if self.speed_threshold is not None:
                ax.axhline(
                    self.speed_threshold,
                    color="r",
                    linestyle="--",
                    linewidth=0.8,
                    label="velocity threshold",
                )
            for step in self.steps:
                if step.side != side:
                    continue
                step_label = f"{step.type} step" if step.type is not None else "step"
                ax.axvspan(
                    step.FC.frame,
                    step.IC.frame,
                    color=step_colors.get(step.type, "gray"),
                    alpha=0.3,
                    label=step_label,
                )
            ax.set_ylabel("m/s")

        for ax in axs:
            for pull in self.pulls:
                ax.axvline(
                    pull.start, color="k", linestyle="--", alpha=0.4, label="pull segment border"
                )
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

    def __repr__(self):
        return (
            f"Recording(system={self.system!r}, frames={len(self.keypoints)}, fps={self.fps}, "
            f"pulls={len(self.pulls)}, steps={len(self.steps)})"
        )


def get_trunk_vectors(keypoints: np.ndarray, labels: list[str]) -> np.ndarray:
    """trunk vector mid_hip -> mid_shoulder for every frame; (frames, 3)."""
    return keypoints[:, labels.index("mid_shoulder"), :] - keypoints[:, labels.index("mid_hip"), :]


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
