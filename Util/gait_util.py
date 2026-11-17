import numpy as np
import matplotlib.pyplot as plt
from Util.util import (
    GaitEvent,
    GaitCycle,
    get_gait_cycles,
    estimate_speed_threshold,
    get_gait_events,
    get_ankle_speed,
    get_segment_vectors,
    get_segment_borders,
    get_keypoint_position_mask,
)
from typing import Literal


class GaitRecording:
    """
    One gait (walking) recording from one system: turns, perspective, gait events and gait cycles.

    Attributes:
        system: "qualisys" or "stereo"
        keypoints: (frames, keypoints, 3) positions in meters; all frame numbers below index into it
        labels: keypoint labels
        fps: sampling rate of the keypoints
        baseline_end: end (frame) of the baseline at the start of the recording, used by the turn
            detection
        turn_signal: (frames,) normalized speed of the horizontal component of the shoulder vector,
            see get_keypoint_position_mask in util
        turn_mask: (frames,) bool, True while turning
        perspective: (frames,) "front" / "back", from the sign of the horizontal component of the
            shoulder vector (the participant walks along the depth axis)
        speed_threshold: ankle speed threshold (m/s) of the final gait event detection
        gait_events: gait events of the whole recording, with perspective and segment set
        gait_cycles: one GaitCycle per IC -> next IC of the same foot
        ankle_thr, initial_speed_threshold, max_stride_time, min_swing_time, min_stance_time,
            turn_threshold, min_turn_duration: the values analyze() used, kept for plotting
    """

    def __init__(
        self,
        system: Literal["qualisys", "stereo"],
        keypoints: np.ndarray,
        labels: list[str],
        fps: float = 100,
        baseline_duration: float = 3,
    ):
        self.system = system
        self.keypoints = keypoints
        self.labels = labels
        self.fps = fps
        self.baseline_end = round(baseline_duration * self.fps)  # baseline_duration in seconds
        self.turn_signal: np.ndarray | None = None
        self.turn_mask: np.ndarray | None = None
        self.perspective: np.ndarray | None = None
        self.speed_threshold: float | None = None
        self.gait_events: list[GaitEvent] = []
        self.gait_cycles: list[GaitCycle] = []
        self.ankle_thr: float | None = None
        self.initial_speed_threshold: float | None = None
        self.max_stride_time: float | None = None
        self.min_swing_time: float | None = None
        self.min_stance_time: float | None = None
        self.turn_threshold: float | None = None
        self.min_turn_duration: float | None = None

    def analyze(
        self,
        ankle_thr: float = 0.8,
        initial_speed_threshold: float = 1.0,
        max_stride_time: float = 2.0,
        min_swing_time: float = 0.3,
        min_stance_time: float = 0.3,
        turn_threshold: float = 0.15,
        min_turn_duration: float = 0.5,
    ):
        """
        Full pipeline: turns and perspective -> speed threshold from the walking speed ->
        gait events (outside the turns) ->
        gait cycles. Fills the result attributes.

        Args:
            ankle_thr: multiplier of the stride velocity, gives the ankle speed threshold
            initial_speed_threshold: velocity (m/s) of the first gait event detection stage
            max_stride_time: longer IC -> IC intervals (s) are not gait cycles
            min_swing_time: swing periods shorter than this (s) are treated as ground contact
            min_stance_time: stance periods shorter than this (s) are treated as swing
            turn_threshold: fraction of the max turn signal above which the subject is turning
            min_turn_duration: turning and straight periods shorter than this (s) are merged into
                their neighbours
        """
        self.ankle_thr = ankle_thr
        self.initial_speed_threshold = initial_speed_threshold
        self.max_stride_time = max_stride_time
        self.min_swing_time = min_swing_time
        self.min_stance_time = min_stance_time
        self.turn_threshold = turn_threshold
        self.min_turn_duration = min_turn_duration

        # turns, same as in the TUG: the horizontal component of the shoulder vector changes
        # fastest while turning
        shoulder_vector_horizontal = get_segment_vectors(
            self.keypoints,
            self.labels,
            start_kpt_name="left_shoulder",
            end_kpt_name="right_shoulder",
        )[:, 0]
        shoulder_horizontal_speed = np.abs(np.gradient(shoulder_vector_horizontal))
        straight_mask, self.turn_signal = get_keypoint_position_mask(
            shoulder_horizontal_speed,
            self.baseline_end,
            "side",
            min_turn_duration * self.fps,
            turn_threshold,
            100,
        )
        self.turn_mask = ~straight_mask

        # walking along the depth axis: the shoulder vector's horizontal component flips sign at each turn
        self.perspective = np.where(shoulder_vector_horizontal > 0, "front", "back")

        self.speed_threshold = estimate_speed_threshold(
            self.keypoints,
            self.labels,
            fps=self.fps,
            ankle_thr=ankle_thr,
            initial_speed_threshold=initial_speed_threshold,
            max_stride_time=max_stride_time,
            min_swing_time=min_swing_time,
            min_stance_time=min_stance_time,
            event_mask=straight_mask,
        )
        self.gait_events = get_gait_events(
            self.keypoints,
            self.labels,
            fps=self.fps,
            speed_threshold=self.speed_threshold,
            min_swing_time=min_swing_time,
            min_stance_time=min_stance_time,
            event_mask=straight_mask,
        )
        # the shared detector only sets type, frame and side: add the walking context
        for event in self.gait_events:
            event.perspective = self.perspective[event.frame]
            event.segment = "turn" if self.turn_mask[event.frame] else "straight"

        self.gait_cycles = get_gait_cycles(
            self.gait_events, self.keypoints, self.labels, self.fps, max_stride_time
        )

    def plot(self):
        """
        Whole recording: turn signal with the turn threshold, horizontal component of the shoulder
        vector (its sign is the perspective), ankle speeds with the initial contacts and the
        speed threshold, and the gait events found for each gait cycle (one column per cycle, at
        its IC0). Turns are shaded. Call analyze() first.

        Returns:
            the figure
        """
        side_colors = {"left": "tab:blue", "right": "tab:olive"}

        fig, axs = plt.subplots(5, 1, figsize=(15, 11), sharex=True)
        fig.suptitle(self.system)

        axs[0].set_title("turn signal (normalized shoulder vector change)")
        axs[0].plot(self.turn_signal, "tab:purple", label="turn signal")
        axs[0].axhline(self.turn_threshold, color="#00000049", label="turn threshold")

        axs[1].set_title("shoulder vector, horizontal component (> 0 = front, < 0 = back)")
        shoulder_vector_horizontal = get_segment_vectors(
            self.keypoints,
            self.labels,
            start_kpt_name="left_shoulder",
            end_kpt_name="right_shoulder",
        )[:, 0]
        axs[1].plot(shoulder_vector_horizontal, "tab:blue", label="left -> right shoulder")
        axs[1].axhline(0, color="#00000049")
        axs[1].set_ylabel("m")

        # one row per ankle: left in axs[2], right in axs[3]
        for ankle_ax, (side, color) in zip(axs[2:4], side_colors.items()):
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
                label="initial contact",
            )
            ankle_ax.axhline(self.speed_threshold, color="#00000049", label="speed threshold")
            ankle_ax.set_ylabel("m/s")

        # one column per gait cycle, at its IC0: which gait events were found; invalid cycles faded
        event_colors = {"IC0": "c", "IC2": "m", "IC1": "r", "FC1": "b", "FC0": "g"}
        for gait_cycle in self.gait_cycles:
            alpha = 1 if gait_cycle.valid else 0.25
            for row_number, (event_name, color) in enumerate(event_colors.items()):
                if getattr(gait_cycle, event_name) is not None:
                    axs[4].vlines(
                        gait_cycle.IC0.frame,
                        row_number + 0.1,
                        row_number + 0.9,
                        color,
                        linewidth=3,
                        alpha=alpha,
                    )
        axs[4].set_yticks(
            [row_number + 0.5 for row_number in range(len(event_colors))], list(event_colors)
        )
        valid_cycle_count = sum(gait_cycle.valid for gait_cycle in self.gait_cycles)
        axs[4].set_title(
            f"gait cycle composition ({valid_cycle_count} of {len(self.gait_cycles)} valid, "
            "invalid ones faded)"
        )
        axs[4].set_xlabel("frame")

        for turn_number, (turn_start, turn_end) in enumerate(get_segment_borders(self.turn_mask)):
            for ax in axs:
                # label only once, so the legend has one entry
                ax.axvspan(
                    turn_start,
                    turn_end,
                    color="tab:red",
                    alpha=0.15,
                    label="turn" if turn_number == 0 else None,
                )

        for ax in axs[:4]:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        plt.tight_layout()
        return fig
