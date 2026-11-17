import numpy as np
import matplotlib.pyplot as plt
from dataclasses import make_dataclass
from Util.util import (
    angle_between_vectors,
    get_projection,
    GaitEvent,
    get_gait_events_two_stage,
    get_ankle_speed,
    get_segment_vectors,
    get_segment_borders,
    get_keypoint_position_mask,
)
from typing import Literal

# gait parameters to be calculated later
GAIT_PARAMETERS = [
    "stride_length",
    "stride_time",
    "stride_velocity",
    "base_of_support",
    "double_support_time",
    "swing_time",
    "knee_ext_rom",
    "shoulder_flex_rom",
    "shoulder_abd_rom",
    "elbow_ext_rom",
]

# create dataclass for storing gait parameters
parameters = [(parameter_name, float, np.nan) for parameter_name in GAIT_PARAMETERS]
Parameters = make_dataclass(cls_name="Parameters", fields=parameters)


class GaitCycle:
    """
    Class to representing a Gait Cycle. Calculates, checks and stores the propeprties of a GC.

    Attributes:
        possible_gait_events: GEs which might belong to this GC. They will be evaluated and if fit, the final GEs will be picked from this list.
        keypoints: 3D keypoint (preprocessed) data in (frames, num_kpts, dims) shape.
        labels: labels describing each keypoint
        fps: recording frequency in Hz.
        IC0: 1st IC type ipsilateral GE, regarded as the start of GC
        IC2: last IC type ipsilateral GE, regarded as end of GC
        IC1, FC0, FC1: rest of the GEs of the segment
        ipsi: ipsilateral side (left/right)
        contra: contralateral side
        perspective: perspective of the segment to which the GEs belong
        segment: whether the segment is straight/turn type
        gait_events: the GEs that have been selected and verified to be part of this GC
        parameters: Parameters type container, storing the results of calculated gait-cycle parameters
        valid: see the valid property

    """

    def __init__(
        self,
        IC0: GaitEvent,
        IC2: GaitEvent,
        input_gait_events: list,
        keypoint_data: np.ndarray,
        labels: list,
        fps,
    ):

        self.possible_gait_events = sorted(input_gait_events, key=lambda event: event.frame)
        self.keypoints = keypoint_data
        self.labels = labels
        self.fps = fps

        # IC0, IC2 are determined at the Recording level (checks are also performed there)
        self.IC0 = IC0
        self.IC2 = IC2

        self.ipsi = self.IC0.side
        self.contra = "left" if self.ipsi == "right" else "right"
        self.perspective = self.IC0.perspective

        # GE priorities (based on parameters): IC0-IC2, IC1, FC0-FC1 -> select IC1 first, then FC0, FC1
        self.IC1 = self.select_gait_event(type="IC", side=self.contra)
        self.FC0 = self.select_gait_event(
            type="FC", side=self.contra, prev_event=self.IC0, next_event=self.IC1
        )
        self.FC1 = self.select_gait_event(
            type="FC", side=self.ipsi, prev_event=self.IC1, next_event=self.IC2
        )

        # get rid of not found / non-existent GEs
        self.gait_events = [ge for ge in [self.IC0, self.FC0, self.IC1, self.FC1, self.IC2] if ge]

        self.parameters = Parameters()
        self.calculate_stride_parameters()
        self.calculate_knee_rom()
        self.calculate_elbow_rom()
        self.calculate_shoulder_rom()
        if None not in [self.FC1]:
            self.calculate_swing_time()
        if None not in [self.IC1]:
            self.calculate_base_of_support()
            if None not in [self.FC0, self.FC1]:
                self.calculate_double_support_time()

    @property
    def valid(self) -> bool:
        """
        True if all of these hold (IC1, FC0 and FC1 may be missing, only the found ones are checked):
            IC0 and IC2 found
            types: IC0, IC1, IC2 are ICs; FC0, FC1 are FCs
            order: IC0 <= FC0 <= IC1 <= FC1 <= IC2 by frame, IC0 < IC2
            sides: IC0, FC1, IC2 are ipsilateral; FC0, IC1 are contralateral
            no turn: no found event has segment "turn" (events without a segment pass)
        """
        if self.IC0 is None or self.IC2 is None:
            return False

        expected = {
            "IC0": ("IC", self.ipsi),
            "FC0": ("FC", self.contra),
            "IC1": ("IC", self.contra),
            "FC1": ("FC", self.ipsi),
            "IC2": ("IC", self.ipsi),
        }
        found_events = [
            (getattr(self, name), event_type, side)
            for name, (event_type, side) in expected.items()
            if getattr(self, name) is not None
        ]
        types_ok = all(event.type == event_type for event, event_type, _ in found_events)
        sides_ok = all(event.side == side for event, _, side in found_events)

        frames = [event.frame for event, _, _ in found_events]
        order_ok = self.IC0.frame < self.IC2.frame and all(
            frame <= next_frame for frame, next_frame in zip(frames, frames[1:])
        )

        no_turn = all(event.segment != "turn" for event, _, _ in found_events)

        return bool(types_ok and sides_ok and order_ok and no_turn)

    # str() so that a None perspective (e.g. TUG events) can be padded
    def __str__(self):
        return f"{str(self.perspective):<10}{self.ipsi:<10}{[ge.frame for ge in self.gait_events]}"

    def __repr__(self):
        return f"{str(self.perspective):<10}{self.ipsi:<10}{self.contra:<10}{[ge.frame for ge in self.gait_events]}"

    def calculate_stride_parameters(self) -> None:  # IC0, IC2
        """
        Computes stride_time, stride_lenght and stride_velocity.
            Gait events needed: IC0, IC2

        Sets:
            parameters.stride_time
            parameters.stride_length
            parameters.stride_velocity
        """
        self.parameters.stride_time = (self.IC2.frame - self.IC0.frame) / self.fps

        IC0_ankle = self.keypoints[self.IC0.frame, self.labels.index(f"{self.ipsi}_ankle"), :]
        IC2_ankle = self.keypoints[self.IC2.frame, self.labels.index(f"{self.ipsi}_ankle"), :]
        self.parameters.stride_length = np.linalg.norm(IC2_ankle - IC0_ankle)

        self.parameters.stride_velocity = (
            self.parameters.stride_length / self.parameters.stride_time
            if self.parameters.stride_time > 0
            else np.nan
        )

    def calculate_swing_time(self) -> None:  # FC1, IC2
        """
        Computes swing_time.
            Gait events needed: FC1, IC2

        Sets:
            parameters.swing_time
        """
        if self.fps > 0:
            self.parameters.swing_time = (self.IC2.frame - self.FC1.frame) / self.fps

    def calculate_double_support_time(self) -> None:  # FC0, FC1, IC0, IC1
        """
        Computes double_support_time.
            Gait events needed: FC0, FC1, IC0, IC1

        Sets:
            parameters.double_support_time
        """
        if self.fps > 0:
            self.parameters.double_support_time = (
                (self.FC0.frame - self.IC0.frame) + (self.FC1.frame - self.IC1.frame)
            ) / self.fps

    def calculate_base_of_support(self) -> None:  # IC0, IC1, IC2
        """
        Computes base_of_support.
            Gait events needed: IC0, IC1, IC2

        Sets:
            parameters.base_of_support
        """
        # using ankles
        IC1_ankle = self.keypoints[self.IC1.frame, self.labels.index(f"{self.contra}_ankle"), :]
        IC0_ankle = self.keypoints[self.IC0.frame, self.labels.index(f"{self.ipsi}_ankle"), :]
        IC2_ankle = self.keypoints[self.IC2.frame, self.labels.index(f"{self.ipsi}_ankle"), :]

        IC2_IC0_line = IC2_ankle - IC0_ankle
        IC0_IC1_line = IC0_ankle - IC1_ankle

        self.parameters.base_of_support = np.linalg.norm(
            np.cross(IC2_IC0_line, IC0_IC1_line)
        ) / np.linalg.norm(IC2_IC0_line)

    def calculate_knee_rom(self) -> None:  # IC0, IC2
        """
        Computes knee range of motion based on extension angle.
            Gait events needed: IC0, IC2

        Sets:
            parameters.knee_ext_rom
        """

        hip_position = self.keypoints[
            self.IC0.frame : self.IC2.frame, self.labels.index(f"{self.ipsi}_hip"), :
        ]

        knee_position = self.keypoints[
            self.IC0.frame : self.IC2.frame, self.labels.index(f"{self.ipsi}_knee"), :
        ]
        ankle_position = self.keypoints[
            self.IC0.frame : self.IC2.frame, self.labels.index(f"{self.ipsi}_ankle"), :
        ]

        # knee ROM
        upper_leg_vector = knee_position - hip_position
        lower_leg_vector = ankle_position - knee_position

        # knee EXTENSION angle for all frames of the current gait-cycle, no projection
        knee_angles = [
            angle_between_vectors(lower_leg, upper_leg)
            for lower_leg, upper_leg in zip(lower_leg_vector, upper_leg_vector)
        ]
        # knee range of motion = max(knee angle) - min(knee angle) in given gait cycle
        self.parameters.knee_ext_rom = np.nanmax(knee_angles) - np.nanmin(knee_angles)
        self.knee_ext_angles = knee_angles

    def calculate_elbow_rom(self) -> None:  # IC0, IC2
        """
        Computes elbow range of motion based on flexion angle.
            Gait events needed: IC0, IC2

        Sets:
            parameters.elbow_ext_rom
        """
        shoulder_position = self.keypoints[
            self.IC0.frame : self.IC2.frame, self.labels.index(f"{self.contra}_shoulder"), :
        ]
        elbow_position = self.keypoints[
            self.IC0.frame : self.IC2.frame, self.labels.index(f"{self.contra}_elbow"), :
        ]
        wrist_position = self.keypoints[
            self.IC0.frame : self.IC2.frame, self.labels.index(f"{self.contra}_wrist"), :
        ]

        # arm rom (elbow flexion, shoulder flexion, shoulder adduction)
        upper_arm_vector = elbow_position - shoulder_position
        lower_arm_vector = wrist_position - elbow_position

        # elbow flexion angle w/o projection
        elbow_angles = [
            angle_between_vectors(upper_arm, lower_arm)
            for upper_arm, lower_arm in zip(upper_arm_vector, lower_arm_vector)
        ]
        self.elbow_ext_angles = elbow_angles
        self.parameters.elbow_ext_rom = np.nanmax(elbow_angles) - np.nanmin(elbow_angles)

    def calculate_shoulder_rom(self) -> None:  # IC0, IC2
        """
        Computes shoulder range of motion based on flexion/extension angle.
            Shoulder flexion/extension: angle between torso vector (mid_shoulder-mid_hip) and upper arm (shoulder-elbow) projections in sagittal plane.
            Shoulder adduction/abduction: angle between shoulder vector (left_shoulder-right_shoulder) and upper arm (shoulder-elbow) projections in coronal plane.
        Gait events needed: IC0, IC2

        Sets:
            parameters.shoulder_flex_rom
            parameters.shoulder_abd_rom
            self.shoulder_flex_angles
            self.shoulder_abd_angles
        """
        contra_shoulder_position = self.keypoints[
            self.IC0.frame : self.IC2.frame, self.labels.index(f"{self.contra}_shoulder"), :
        ]
        ipsi_shoulder_position = self.keypoints[
            self.IC0.frame : self.IC2.frame, self.labels.index(f"{self.ipsi}_shoulder"), :
        ]
        elbow_position = self.keypoints[
            self.IC0.frame : self.IC2.frame, self.labels.index(f"{self.contra}_elbow"), :
        ]
        mid_shoulder_position = self.keypoints[
            self.IC0.frame : self.IC2.frame, self.labels.index(f"mid_shoulder"), :
        ]

        mid_hip_position = self.keypoints[
            self.IC0.frame : self.IC2.frame, self.labels.index(f"mid_hip"), :
        ]

        upper_arm_vector = elbow_position - contra_shoulder_position
        torso_vector = mid_hip_position - mid_shoulder_position

        sagittal_normal = contra_shoulder_position - ipsi_shoulder_position
        coronal_normal = np.cross(sagittal_normal, torso_vector)

        # shoulder flexion/extension angle with projection (torso and upper arm on sagittal plane)
        sagittal_upper_arm_proj = get_projection(sagittal_normal, upper_arm_vector)
        sagittal_torso_proj = get_projection(sagittal_normal, torso_vector)
        shoulder_flex_angles = np.array(
            [
                angle_between_vectors(upper_arm_proj, torso_proj)
                for upper_arm_proj, torso_proj in zip(sagittal_upper_arm_proj, sagittal_torso_proj)
            ]
        )
        # arccos gives unsigned angles: sign them by which side of the torso line the arm is on
        cross_product = np.cross(sagittal_torso_proj, sagittal_upper_arm_proj)
        sign = np.sign(np.sum(cross_product * sagittal_normal, axis=1))
        shoulder_flex_angles = sign * shoulder_flex_angles

        self.parameters.shoulder_flex_rom = np.nanmax(shoulder_flex_angles) - np.nanmin(
            shoulder_flex_angles
        )
        self.shoulder_flex_angles = shoulder_flex_angles

        # shoulder abduction/adduction angle with projection (torso and upper arm on coronal plane)
        coronal_torso_proj = get_projection(coronal_normal, torso_vector)
        coronal_arm_proj = get_projection(coronal_normal, upper_arm_vector)
        shoulder_abd_angles = [
            angle_between_vectors(torso_proj, upper_arm_proj)
            for torso_proj, upper_arm_proj in zip(coronal_torso_proj, coronal_arm_proj)
        ]
        self.parameters.shoulder_abd_rom = np.nanmax(shoulder_abd_angles) - np.nanmin(
            shoulder_abd_angles
        )
        self.shoulder_abd_angles = shoulder_abd_angles

    def select_gait_event(
        self,
        type: Literal["IC", "FC"],
        side: Literal["left", "right"],
        prev_event: GaitEvent = None,
        next_event: GaitEvent = None,
    ) -> GaitEvent | None:
        """
        Select the appropriate gait event from the possible_gait_events belonging to this gait cycle.

        Args:
            type: type of gait event
            side: side of gait event
            prev_event: previoius GE event (based on frame). Selected GE can only come after this
            next_event: next GE event. Selected one must come before this.

        Returns:
            selected_gait_event: the first event (ranked by frame) that satisfies the criteria, otherwise None
        """

        frame_min = prev_event.frame if prev_event else self.IC0.frame
        frame_max = next_event.frame if next_event else self.IC2.frame

        specified_gait_events = [
            ge
            for ge in self.possible_gait_events
            if (
                ge.type == type
                and ge.side == side
                and ge.perspective == self.perspective
                and frame_min <= ge.frame <= frame_max
            )
        ]
        selected_gait_event = next(iter(specified_gait_events), None)
        return selected_gait_event


def get_gait_cycles(
    gait_events: list[GaitEvent],
    keypoints: np.ndarray,
    labels: list[str],
    fps: float,
    max_stride_time: float,
) -> list[GaitCycle]:
    """
    One GaitCycle per IC -> next IC of the same foot, if that IC comes within max_stride_time and
    has the same perspective and segment (events without these, e.g. in the TUG, have None for
    both, so they always match). Invalid cycles are kept, see GaitCycle.valid.

    Args:
        gait_events: events of the whole recording, e.g. from get_gait_events
        keypoints: (frames, keypoints, 3) of the whole recording, indexed with recording frames
        labels: keypoint labels
        fps: sampling rate of the keypoints
        max_stride_time: longer IC -> IC intervals (s) are not gait cycles
    Returns:
        gait cycles, in the order of their IC0 in gait_events
    """
    max_stride_frames = max_stride_time * fps
    gait_cycles = []
    for event in gait_events:
        if event.type == "IC":
            next_ipsi_IC = next(
                iter(
                    [
                        ge
                        for ge in gait_events
                        if (
                            ge.type == "IC"
                            and ge.side == event.side
                            and ge.frame > event.frame
                            and ge.frame <= event.frame + max_stride_frames
                            and ge.perspective == event.perspective
                            and ge.segment == event.segment
                        )
                    ]
                ),
                None,
            )
            if next_ipsi_IC:
                possible_events = [
                    ge
                    for ge in gait_events
                    if (ge.frame >= event.frame and ge.frame <= next_ipsi_IC.frame)
                ]
                current_gait_cycle = GaitCycle(
                    event, next_ipsi_IC, possible_events, keypoints, labels, fps
                )
                gait_cycles.append(current_gait_cycle)

    return gait_cycles


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
        Full pipeline: turns and perspective -> gait events (two-stage, outside the turns) ->
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

        self.gait_events, self.speed_threshold = get_gait_events_two_stage(
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
