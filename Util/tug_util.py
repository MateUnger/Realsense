import numpy as np
import matplotlib.pyplot as plt
from Util.util import (
    get_segment_vectors,
    get_segment_borders,
    angle_between_vectors,
    fill_short_runs,
)
from Util.pulltest_util import GaitEvent, get_gait_events, get_ankle_speed


def get_keypoint_position_mask(
    keypoint: np.ndarray,
    baseline_end: float,
    direction: str,
    min_state_length: float,
    threshold: float,
    max_percentile: int,
):
    """
    Normalizes 1D keypoint coords, creates a mask indicating the position being close to the max (up, side) or min (down) of the whole recording.
    Assumes the coordinate rises after the baseline.
    """
    position = keypoint
    position_baseline = np.nanmean(position[: int(baseline_end)])
    position_baseline_diff = position - position_baseline
    position_normalized = position_baseline_diff / np.nanpercentile(
        position_baseline_diff, max_percentile
    )

    match direction:
        case "up":
            mask = position_normalized > 1 - threshold
            mask = fill_short_runs(mask, value=0, min_length=min_state_length)
        case "down" | "side":
            mask = position_normalized < threshold
            mask = fill_short_runs(mask, value=1, min_length=min_state_length)
            mask = fill_short_runs(mask, value=0, min_length=min_state_length)

    # mask = fill_short_runs(mask, value=1, min_length=min_state_length)
    return mask, position_normalized


def get_trunk_angle_and_movement_mask(
    keypoints: np.ndarray, labels: list, baseline_end: float, threshold: float, min_lean_time: float
):
    """
    Calculates the angle of the trunk vectors (mid_hip -> mid_shoulder) compared to their baseline mean (sitting).
    Also creates an upright mask indicating when the lean angle is less than threshold.

    Args:
        keypoints: keypoint data (n_frames, n_kpts, n_dims)
        labels: labels for the keypoints
        baseline_end: end of the baseline period (in samples)
        threshold: limit (in degrees) above which leaning/movement is detected
        min_lean_time: duration of time (in seconds) below which leaning/movement is considered noise
    """
    trunk_vectors = get_segment_vectors(
        keypoints, labels, start_kpt_name="mid_hip", end_kpt_name="mid_shoulder"
    )
    trunk_vector_baseline = np.nanmean(trunk_vectors[:baseline_end], axis=0)
    trunk_angles = np.array(
        [
            angle_between_vectors(trunk_vector, trunk_vector_baseline)
            for trunk_vector in trunk_vectors
        ]
    )
    trunk_angle_baseline = np.nanmean(trunk_angles[:baseline_end])
    upright_mask = (trunk_angles - trunk_angle_baseline) < threshold

    upright_mask = fill_short_runs(upright_mask, value=1, min_length=min_lean_time).astype(bool)
    upright_mask = fill_short_runs(upright_mask, value=0, min_length=min_lean_time).astype(bool)

    return trunk_angles, upright_mask


class TimedUpAndGo:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        # phase boundaries (frames) and durations (seconds), set by set_sit_stand_phases
        self.stand_up_start = None
        self.stand_up_end = None
        self.sit_down_start = None
        self.sit_down_end = None
        self.stand_up_duration = np.nan
        self.sit_down_duration = np.nan
        # trunk bending per phase (degrees, degrees/s), set by set_trunk_bending
        self.stand_up_max_trunk_angle = np.nan
        self.stand_up_max_trunk_angular_velocity = np.nan
        self.sit_down_max_trunk_angle = np.nan
        self.sit_down_max_trunk_angular_velocity = np.nan
        # mid_shoulder speed per phase (m/s), set by set_shoulder_velocity
        self.stand_up_max_shoulder_velocity = np.nan
        self.sit_down_max_shoulder_velocity = np.nan
        # turn at the far end and turn before sitting down: boundaries (frames), durations
        # (seconds) and number of steps, set by set_turn_phases and set_turn_steps
        self.turn_start = None
        self.turn_end = None
        self.turn_to_sit_start = None
        self.turn_to_sit_end = None
        self.turn_duration = np.nan
        self.turn_to_sit_duration = np.nan
        self.turn_steps = None
        self.turn_to_sit_steps = None

    def set_sit_stand_phases(self, sit_mask: np.ndarray, stand_mask: np.ndarray, fps: float):
        """
        Standing up: from the end of the sitting period at the start of the TUG to the first
        standing frame. Sitting down: from the end of the last standing period to the end of the
        TUG (start of the next sitting period).

        Args:
            sit_mask: sitting mask of the whole recording
            stand_mask: standing mask of the whole recording
            fps: sampling rate in Hz
        """
        sitting_segments = get_segment_borders(sit_mask[self.start : self.end])
        standing_segments = get_segment_borders(stand_mask[self.start : self.end])
        if not sitting_segments or not standing_segments:
            return  # no sitting or standing found in this TUG, durations stay NaN

        self.stand_up_start = self.start + sitting_segments[0][1]
        self.stand_up_end = self.start + standing_segments[0][0]
        self.sit_down_start = self.start + standing_segments[-1][1]
        self.sit_down_end = self.end

        self.stand_up_duration = (self.stand_up_end - self.stand_up_start) / fps
        self.sit_down_duration = (self.sit_down_end - self.sit_down_start) / fps

    def set_trunk_bending(self, trunk_angles: np.ndarray, fps: float):
        """
        Maximum trunk bending angle and maximum (absolute) trunk angular velocity during the
        standing up and sitting down phases. Call set_sit_stand_phases first.

        Args:
            trunk_angles: trunk angle (degrees) relative to the sitting baseline, whole recording
            fps: sampling rate in Hz
        """
        # velocity over the whole recording, so the phase edges get proper central differences
        trunk_angular_velocity = np.gradient(trunk_angles) * fps

        if self.stand_up_start is not None:
            stand_up_frames = slice(self.stand_up_start, self.stand_up_end)
            self.stand_up_max_trunk_angle = np.nanmax(trunk_angles[stand_up_frames])
            self.stand_up_max_trunk_angular_velocity = np.nanmax(
                np.abs(trunk_angular_velocity[stand_up_frames])
            )
        if self.sit_down_start is not None:
            sit_down_frames = slice(self.sit_down_start, self.sit_down_end)
            self.sit_down_max_trunk_angle = np.nanmax(trunk_angles[sit_down_frames])
            self.sit_down_max_trunk_angular_velocity = np.nanmax(
                np.abs(trunk_angular_velocity[sit_down_frames])
            )

    def set_shoulder_velocity(self, keypoints: np.ndarray, labels: list, fps: float):
        """
        Maximum 3D speed of the mid_shoulder keypoint during the standing up and sitting down
        phases. Call set_sit_stand_phases first.

        Args:
            keypoints: keypoint data of the whole recording (n_frames, n_kpts, n_dims)
            labels: labels for the keypoints
            fps: sampling rate in Hz
        """
        # speed over the whole recording, so the phase edges get proper central differences
        shoulder_position = keypoints[:, labels.index("mid_shoulder"), :]
        shoulder_speed = np.linalg.norm(np.gradient(shoulder_position, axis=0), axis=1) * fps

        if self.stand_up_start is not None:
            self.stand_up_max_shoulder_velocity = np.nanmax(
                shoulder_speed[self.stand_up_start : self.stand_up_end]
            )
        if self.sit_down_start is not None:
            self.sit_down_max_shoulder_velocity = np.nanmax(
                shoulder_speed[self.sit_down_start : self.sit_down_end]
            )

    def set_turn_phases(self, turn_mask: np.ndarray, fps: float):
        """
        Turn: the first turning segment after standing up (the 180 deg turn at the far end).
        Turn to sit: the last turning segment of the TUG (the turn before sitting down).
        With only one turning segment, only the turn is set. Call set_sit_stand_phases first.

        Args:
            turn_mask: turning mask of the whole recording
            fps: sampling rate in Hz
        """
        if self.stand_up_end is None:
            return
        turning_segments = get_segment_borders(turn_mask[self.stand_up_end : self.end])
        if not turning_segments:
            return

        self.turn_start = self.stand_up_end + turning_segments[0][0]
        self.turn_end = self.stand_up_end + turning_segments[0][1]
        self.turn_duration = (self.turn_end - self.turn_start) / fps

        if len(turning_segments) >= 2:
            self.turn_to_sit_start = self.stand_up_end + turning_segments[-1][0]
            self.turn_to_sit_end = self.stand_up_end + turning_segments[-1][1]
            self.turn_to_sit_duration = (self.turn_to_sit_end - self.turn_to_sit_start) / fps

    def set_turn_steps(self, gait_events: list[GaitEvent]):
        """
        Number of steps in the turn and the turn to sit: the initial contacts (a foot landing)
        that fall inside each phase. Call set_turn_phases first.

        Args:
            gait_events: gait events of the whole recording
        """
        initial_contact_frames = [event.frame for event in gait_events if event.type == "IC"]
        if self.turn_start is not None:
            self.turn_steps = sum(
                self.turn_start <= frame < self.turn_end for frame in initial_contact_frames
            )
        if self.turn_to_sit_start is not None:
            self.turn_to_sit_steps = sum(
                self.turn_to_sit_start <= frame < self.turn_to_sit_end
                for frame in initial_contact_frames
            )

    def __repr__(self):
        # short form, used when a list of TUGs is displayed
        return f"TUG(start={self.start}, end={self.end})"


MIN_SIT_STAND_DURATION = 0.75  # seconds
HIP_DOWN_THR = 0.1  # fraction (normalized)
HIP_UP_THR = 0.1  # fraction (normalized)
SHOULDER_UP_THR = 0.05
TRUNK_ANGLE_THR = (
    7.5  # degrees TODO: assuming they return to a sitting position within thr degrees of baseline
)
MIN_LEAN_TIME = 1  # sec
TURN_THR = 0.15
MIN_TURN_DURATION = 0.5  # seconds
STEP_SPEED_THR = 0.5  # m/s, ankle speed below which the foot counts as on the ground
MIN_SWING_TIME = 0.3  # seconds
MIN_STANCE_TIME = 0.3  # seconds


class TimedUpAndGoRecording:
    def __init__(self, system, keypoints, labels, fps=100, baseline_duration=3):
        self.system = system
        self.keypoints = keypoints
        self.labels = labels
        self.fps = fps
        self.baseline_end = int(baseline_duration * self.fps)  # baseline_duration in seconds
        self.sit_mask = None
        self.stand_mask = None
        self.turn_mask = None
        self.heading = None
        self.gait_events: list[GaitEvent] = []
        self.tugs: list[TimedUpAndGo] = []

    def segment_tugs(self):
        # hips
        hip_vertical = self.keypoints[:, self.labels.index("mid_hip"), 2]
        hip_down, _ = get_keypoint_position_mask(
            hip_vertical,
            self.baseline_end,
            "down",
            MIN_SIT_STAND_DURATION * self.fps,
            HIP_DOWN_THR,
            95,
        )
        hip_up, hip_vertical = get_keypoint_position_mask(
            hip_vertical, self.baseline_end, "up", MIN_SIT_STAND_DURATION * self.fps, HIP_UP_THR, 95
        )
        # shoulders
        shoulder_vertical = self.keypoints[:, self.labels.index("mid_shoulder"), 2]
        shoulder_up, shoulder_vertical = get_keypoint_position_mask(
            shoulder_vertical,
            self.baseline_end,
            "up",
            MIN_SIT_STAND_DURATION * self.fps,
            SHOULDER_UP_THR,
            95,
        )
        # tunk angles
        self.trunk_angles, upright_mask = get_trunk_angle_and_movement_mask(
            self.keypoints,
            self.labels,
            self.baseline_end,
            TRUNK_ANGLE_THR,
            MIN_LEAN_TIME * self.fps,
        )
        # turninig
        shoulder_horizontal_velocity = np.abs(
            np.gradient(
                get_segment_vectors(
                    self.keypoints,
                    self.labels,
                    start_kpt_name="left_shoulder",
                    end_kpt_name="right_shoulder",
                )[:, 0]
            )
        )
        straight_mask, shoulder_horizontal_velocity = get_keypoint_position_mask(
            shoulder_horizontal_velocity,
            self.baseline_end,
            "side",
            MIN_TURN_DURATION * self.fps,
            TURN_THR,
            100,
        )

        # heading: direction of the left -> right shoulder line in the horizontal plane
        # (axes 0 and 1), degrees, wrapped to [-180, 180]; facing direction is 90 deg off
        shoulder_line = get_segment_vectors(
            self.keypoints, self.labels, start_kpt_name="left_shoulder", end_kpt_name="right_shoulder"
        )
        self.heading = np.degrees(np.arctan2(shoulder_line[:, 1], shoulder_line[:, 0]))

        # final masks
        self.sit_mask = hip_down & upright_mask  # hips down and trunk straight
        self.stand_mask = hip_up & shoulder_up  # hips up and shoulders up
        self.turn_mask = ~straight_mask  # shoulders in line (horizontal)

        segment_borders = get_segment_borders(self.sit_mask)
        self.tugs = [
            TimedUpAndGo(start=current_segment_start, end=next_segment_start)
            for (current_segment_start, _), (next_segment_start, _) in zip(
                segment_borders, segment_borders[1:]
            )
        ]
        self.gait_events = get_gait_events(
            self.keypoints,
            self.labels,
            fps=self.fps,
            speed_threshold=STEP_SPEED_THR,
            min_swing_time=MIN_SWING_TIME,
            min_stance_time=MIN_STANCE_TIME,
        )
        for tug in self.tugs:
            tug.set_sit_stand_phases(self.sit_mask, self.stand_mask, self.fps)
            tug.set_trunk_bending(self.trunk_angles, self.fps)
            tug.set_shoulder_velocity(self.keypoints, self.labels, self.fps)
            tug.set_turn_phases(self.turn_mask, self.fps)
            tug.set_turn_steps(self.gait_events)

    def plot(self):
        """
        Whole recording: mid_shoulder position, trunk bending angle, trunk angular velocity,
        mid_shoulder speed, heading and ankle speeds with the step speed threshold. Dashed
        lines mark the start of each TUG; the standing up, sitting down and turning phases are
        shaded. Call segment_tugs() first.

        Returns:
            the figure
        """
        phase_colors = {
            "stand up": "tab:green",
            "sit down": "tab:orange",
            "turn": "tab:red",
            "turn to sit": "tab:brown",
        }
        side_colors = {"left": "tab:blue", "right": "tab:olive"}

        shoulder_position = self.keypoints[:, self.labels.index("mid_shoulder"), :]
        shoulder_speed = np.linalg.norm(np.gradient(shoulder_position, axis=0), axis=1) * self.fps
        trunk_angular_velocity = np.gradient(self.trunk_angles) * self.fps

        fig, axs = plt.subplots(7, 1, figsize=(15, 14), sharex=True)
        fig.suptitle(self.system)

        axs[0].set_title("mid_shoulder position")
        axs[0].plot(shoulder_position, label=["horizontal", "depth", "vertical"])
        axs[0].set_ylabel("m")

        axs[1].set_title("trunk bending angle")
        axs[1].plot(self.trunk_angles, "tab:purple")
        axs[1].set_ylabel("deg")

        axs[2].set_title("trunk angular velocity")
        axs[2].plot(trunk_angular_velocity, "tab:purple")
        axs[2].set_ylabel("deg/s")

        axs[3].set_title("mid_shoulder speed")
        axs[3].plot(shoulder_speed, "tab:blue")
        axs[3].set_ylabel("m/s")

        axs[4].set_title("heading (shoulder line, horizontal plane)")
        axs[4].plot(self.heading, "tab:cyan")
        axs[4].set_ylabel("deg")
        axs[4].set_yticks([-180, -90, 0, 90, 180])

        # one row per ankle: left in axs[5], right in axs[6]
        ankle_legend_handles = {}
        for ankle_ax, (side, color) in zip(axs[5:], side_colors.items()):
            ankle_speed = get_ankle_speed(self.keypoints, self.labels, side, self.fps)
            ankle_ax.set_title(f"{side} ankle speed")
            ankle_ax.plot(ankle_speed, color=color, label=f"{side} ankle")
            initial_contact_frames = [
                event.frame
                for event in self.gait_events
                if event.type == "IC" and event.side == side
            ]
            ankle_ax.plot(
                initial_contact_frames,
                ankle_speed[initial_contact_frames],
                linestyle="none",
                marker="d",
                markersize=5,
                color="#940909",
                label="step (initial contact)",
            )
            ankle_ax.axhline(STEP_SPEED_THR, color="#00000049", label="speed threshold")
            ankle_ax.set_ylabel("m/s")
            # only the ankle lines in these legends, the phases are already in the top ones
            ankle_legend_handles[ankle_ax] = list(ankle_ax.get_lines())
        axs[6].set_xlabel("frame")

        for tug_number, tug in enumerate(self.tugs):
            for ax in axs:
                ax.axvline(
                    tug.start,
                    color="k",
                    linestyle="--",
                    linewidth=1,
                    label="tug start" if tug_number == 0 else None,
                )
                # label the phases only once, so the legend has one entry each
                if tug.stand_up_start is not None:
                    ax.axvspan(
                        tug.stand_up_start,
                        tug.stand_up_end,
                        color=phase_colors["stand up"],
                        alpha=0.25,
                        label="stand up" if tug_number == 0 else None,
                    )
                if tug.sit_down_start is not None:
                    ax.axvspan(
                        tug.sit_down_start,
                        tug.sit_down_end,
                        color=phase_colors["sit down"],
                        alpha=0.25,
                        label="sit down" if tug_number == 0 else None,
                    )
                if tug.turn_start is not None:
                    ax.axvspan(
                        tug.turn_start,
                        tug.turn_end,
                        color=phase_colors["turn"],
                        alpha=0.25,
                        label="turn" if tug_number == 0 else None,
                    )
                if tug.turn_to_sit_start is not None:
                    ax.axvspan(
                        tug.turn_to_sit_start,
                        tug.turn_to_sit_end,
                        color=phase_colors["turn to sit"],
                        alpha=0.25,
                        label="turn to sit" if tug_number == 0 else None,
                    )

        axs[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        axs[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        for ankle_ax, handles in ankle_legend_handles.items():
            ankle_ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                borderaxespad=0,
            )
        plt.tight_layout()
        return fig
