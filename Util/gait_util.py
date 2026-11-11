import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, envelope
from dataclasses import dataclass, make_dataclass
from Util.util import angle_between_vectors, get_projection, GaitEvent
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


@dataclass
class Flags:
    """
    Dataclass to store the result of various checks performed on GaitCycle objects.

    Attributes:
        complete_cycle: all gait events are available
        types_ok: all gait events are available & have the correct type (IC/FC)
        sides_ok: all gait events are available & have the correct sides (right/left)
        perspectives_ok: at least IC0, IC2 are available & have the same perspective (front/back)
        segments_ok: at least IC0, IC2 are available & have the same segment type (straight, turn) and same segment_index
        valid: GC has not been deemed useless, there are no critical mistakes with the GEs
    """

    complete_cycle: bool = True
    types_ok: bool = True
    sides_ok: bool = True
    perspectives_ok: bool = True
    segments_ok: bool = True
    valid: bool = True

    def __repr__(self):
        return (
            f"valid: {str(self.valid):<6}persp: {str(self.perspectives_ok):<6}"
            f"segments: {str(self.segments_ok):<6}complete cycle: {str(self.complete_cycle):<6}"
            f"types: {str(self.types_ok):<6}sides: {str(self.sides_ok):<6}"
        )


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
        segment_index: index of segment, starting from 0 at the beginning of recording. Changes with perspective
        relative_position: relative position of the start of GC compared to the beginning of current segment
        gait_events: the GEs that have been selected and verified to be part of this GC
        flags: Flags type container, storing the results of GE checks. Indicates the overall quality of GC
        parameters: Parameters type container, storing the results of calculated gait-cycle parameters

    """

    def __init__(self, input_gait_events: list, keypoint_data: np.ndarray, labels: list, fps):

        self.possible_gait_events = sorted(input_gait_events, key=lambda event: event.frame)
        self.keypoints = keypoint_data
        self.labels = labels
        self.fps = fps

        # IC0, IC2 are determined at the Recording level (checks are also performed there)
        self.IC0 = self.possible_gait_events[0] if len(self.possible_gait_events) >= 2 else None
        self.IC2 = self.possible_gait_events[-1] if len(self.possible_gait_events) >= 2 else None

        self.ipsi = self.IC0.side
        self.contra = "left" if self.ipsi == "right" else "right"
        self.perspective = self.IC0.perspective
        self.segment = self.IC0.segment
        self.segment_index = self.IC0.segment_index
        self.relative_position = self.IC0.relative_position

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

        self.flags = Flags()
        self.set_flags()

        self.parameters = Parameters()
        if None not in [self.IC0, self.IC2]:
            self.calculate_stride_parameters()

            if self.flags.valid:
                self.calculate_knee_rom()
                self.calculate_elbow_rom()
                self.calculate_shoulder_rom()
                if None not in [self.FC1]:
                    self.calculate_swing_time()
                if None not in [self.IC1]:
                    self.calculate_base_of_support()
                    if None not in [self.FC0, self.FC1]:
                        self.calculate_double_support_time()

    def __str__(self):
        return f"{self.perspective:<10}{self.ipsi:<10}{self.flags.valid:<10}{[ge.frame for ge in self.gait_events]}"

    def __repr__(self):
        return f"{self.perspective:<10}{self.ipsi:<10}{self.contra:<10}{self.flags.valid:<10}{[ge.frame for ge in self.gait_events]}"

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
        # if it is qualisys recording use the front/back markers according to perspective
        if self.keypoints.shape[0] != 27:
            hip_front_position = self.keypoints[
                self.IC0.frame : self.IC2.frame, self.labels.index(f"{self.ipsi}_hip_front"), :
            ]
            hip_back_position = self.keypoints[
                self.IC0.frame : self.IC2.frame, self.labels.index(f"{self.ipsi}_hip_back"), :
            ]
            hip_position = np.nanmean([hip_front_position, hip_back_position], axis=0)
        # it is stereo
        else:
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

        # if it is qualisys recording use the front/back markers according to perspective
        if self.keypoints.shape[0] != 27:
            mid_hip_position = self.keypoints[
                self.IC0.frame : self.IC2.frame, self.labels.index(f"mid_hip_{self.perspective}"), :
            ]
        # it is stereo
        else:
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

    def set_flags(self) -> None:
        """
        Checks the perspectives (front/back), segments (straight/turn), types (IC/FC), sides (left/right) of gait cycle.

        Sets:
            flags.perspectives_ok
            flags.segments_ok
            flags.types_ok
            flags.sides_ok
            flags.complete_cycle
            flags.valid
        """

        # check if perspectives are all the same (front/back)
        self.flags.perspectives_ok = False not in [
            ge.perspective == self.perspective for ge in self.gait_events
        ]
        # check if segments are all the same (straight/turn)
        self.flags.segments_ok = False not in [
            ge.segment == self.segment and ge.segment_index == self.segment_index
            for ge in self.gait_events
        ]

        self.flags.types_ok = False not in [
            self.IC0.type == "IC" if self.IC0 else True,
            self.FC0.type == "FC" if self.FC0 else True,
            self.IC1.type == "IC" if self.IC1 else True,
            self.FC1.type == "FC" if self.FC1 else True,
            self.IC2.type == "IC" if self.IC2 else True,
        ]
        self.flags.sides_ok = False not in [
            self.IC0.side == self.ipsi if self.IC0 else True,
            self.FC0.side == self.contra if self.FC0 else True,
            self.IC1.side == self.contra if self.IC1 else True,
            self.FC1.side == self.ipsi if self.FC1 else True,
            self.IC2.side == self.ipsi if self.IC2 else True,
        ]

        # if we have all gait events
        if None not in [self.IC0, self.IC1, self.IC2, self.FC0, self.FC1]:
            # get rid of the rest (extra events)
            self.gait_events = sorted(
                [self.IC0, self.IC1, self.IC2, self.FC0, self.FC1], key=lambda event: event.frame
            )
        else:
            self.flags.complete_cycle = False

        if (
            not self.flags.types_ok
            or not self.flags.sides_ok
            or not self.flags.perspectives_ok
            or not self.flags.segments_ok
        ):
            self.flags.valid = False

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
                and ge.segment == self.segment
                and frame_min <= ge.frame <= frame_max
            )
        ]
        selected_gait_event = next(iter(specified_gait_events), None)
        return selected_gait_event


class GaitRecording:
    """
    Class representing the functionality and data of a single recording system (stereo/qualisys).
    Ascertains segment types, indices and perspectives. Performs GE detection and arranges GEs into GCs.

    Attributes:
        keypoints: 3D (prefiltered) keypoint data with shape (frames, num_kpts, dims)
        labels: labels describing each keypoint
        recording_type: type of the system used to obtain the keypoint data (stereo/qualisys
        gait_analysis_properties: properties used for GE detection and preprocessing
        fps: sampling frequency in Hz
        heel_thr: velocity threshold below which the ankle keypoint is considered stationary
        max_stride_time: maximum time below which a stride sequence is considered valid
        min_swing_time: minimal time above which a swing phase is considered valid
        min_stance_time: minaimal time above which a stance phase is considered valid
        debug_folder: folder that should containe the figures and graphs used mainly for debugging
        turn_mask: mask indicating whether a given point in time belongs to a straight or turn segment
        perspective: mask indicating whether the given point in time belongs to a front of back perspective
        segment_indices: mask indicating the index of segments
        segment_boundaries: dictionary linking segment indices to the start and end points of each segment
        initial_gait_events: GEs detected by the first pass of a 2-stage detection algorithm
        initial_gait_cycles: GCs constructed from the initial_gait_events
        velocity_estimate: stride velocity estimate caluclated using initial_gait_cycles
        gait_events: final GEs detected by 2nd stage of the 2-stage algorithm
        all_gait_cycles: all the GCs detected (valid and invalid)
        gait_cycles: only the valid GCs (the ones which passed all the checks)
    """

    def __init__(
        self,
        keypoints: np.ndarray,
        labels,
        recording_type: Literal["stereo", "qualisys"],
        gait_analysis_properties: dict,
        debug_folder: str = None,
    ):
        self.keypoints = keypoints
        self.labels = labels
        self.recording_type = recording_type

        # gait analysis properties
        gait_analysis_properties = gait_analysis_properties[self.recording_type]
        self.fps = gait_analysis_properties["fps"]

        self.heel_thr = gait_analysis_properties["heel_thr"]

        self.max_stride_time = gait_analysis_properties["stride_time_max"] * self.fps
        self.min_swing_time = gait_analysis_properties["swing_time_min"] * self.fps
        self.min_stance_time = gait_analysis_properties["stance_time_min"] * self.fps

        if not os.path.exists(debug_folder):
            os.mkdir(debug_folder)
        self.debug_folder = debug_folder

        self.turn_mask, self.perspective, self.segment_indices, self.segment_boundaries = (
            self.get_turns_and_perspective_and_segments(relative_height=0.7)
        )

        self.initial_gait_events = self.get_gait_events(velocity=1, create_debug_fig=False)
        self.initial_gait_cycles = self.get_gait_cycles(self.initial_gait_events)
        self.velocity_estimate = np.nanmean(
            [gc.parameters.stride_velocity for gc in self.initial_gait_cycles if gc.flags.valid]
        )
        self.gait_events = self.get_gait_events(
            velocity=self.velocity_estimate, create_debug_fig=True
        )
        # make distinction between valid and all-gait cycles
        self.all_gait_cycles = self.get_gait_cycles(self.gait_events)
        self.gait_cycles = [gc for gc in self.all_gait_cycles if gc.flags.valid]
        if self.debug_folder:
            self.create_gait_cycles_figure()

    def get_gait_cycles(self, gait_events: list):
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
                                and ge.frame <= event.frame + self.max_stride_time
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
                        possible_events, self.keypoints, self.labels, self.fps
                    )
                    gait_cycles.append(current_gait_cycle)

        return gait_cycles

    def get_turns_and_perspective_and_segments(
        self,
        min_peak_height: float = 0.25,
        relative_height: float = 0.5,
        min_walk_duration: float = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Identifies turning segments (straight/turn) and perspective (front / back) based on shoulder coordinates using peak detection method.
        Also assigns numeric indices to segments (0,1,2...) and records segment boundaries based on perspective.

        Args:
            min_peak_height:    only peaks above this height will be considered
            relative_height:    height at which the bases of a peak will be taken. It is MEASURED FROM APEX
            min_walk_duration:  min elapsed time (in seconds) between turns (1 turn = 180 deg)

        Returns:
            turn_mask: boolean array indicating straight turning segments (1) and straight segments (0).

            perspective: bollean array indicating frontal facing (1) and back facing (0) segments.

            segment_indices: array containing the segment index for every time point.

            segment_boundaries: dict linking segent indices to segment bondaries. (idx 1-> 0-1500)
        """

        # min distance from peak to peak (2 seconds of straight movement) in samples
        min_peak_distance = min_walk_duration * self.fps

        # get the y coord of shoulders (for qualisys x-y plane is horizontal, y coord was forwards/backwards movement)
        right_shoulder_position = self.keypoints[:, self.labels.index("right_shoulder"), 1]
        left_shoulder_position = self.keypoints[:, self.labels.index("left_shoulder"), 1]

        # get rate of change in positional difference
        shoulder_diff_change = np.abs(
            np.diff(
                right_shoulder_position - left_shoulder_position,
                prepend=(right_shoulder_position - left_shoulder_position)[0],
            )
        )
        # Interpolate NaNs in shoulder_diff
        if np.any(np.isnan(shoulder_diff_change)):
            nans = np.isnan(shoulder_diff_change)
            shoulder_diff_change[nans] = np.interp(
                np.flatnonzero(nans), np.flatnonzero(~nans), shoulder_diff_change[~nans]
            )
        # normalize
        shoulder_diff_change = shoulder_diff_change / np.nanmax(shoulder_diff_change)

        numeric_perspective = np.where(left_shoulder_position > right_shoulder_position, 1, 0)
        perspective_change = np.diff(a=numeric_perspective, prepend=[numeric_perspective[0]])

        # search for peaks in shoulder difference change by looking at position crossover of shoulders
        init_peak_indices = np.flatnonzero(perspective_change)
        init_peaks = shoulder_diff_change[init_peak_indices]

        # assume most peaks are detected normally, pick average representation as median
        median_peak_height = np.median(init_peaks)

        # if there is diff larger than 20% between median and any other peak, consider it outlier
        if True in [abs(peak - median_peak_height) > 0.7 for peak in init_peaks]:
            # clip signal with median
            shoulder_diff_change = shoulder_diff_change.clip(0, median_peak_height)

            # normalize again since maximum changed
            shoulder_diff_change = shoulder_diff_change / np.nanmax(shoulder_diff_change)

            # calculate upper envelope of signal to get rid of noisy parts (one big peak divided into several smaller)
            shoulder_diff_change = envelope(
                z=shoulder_diff_change, bp_in=(None, 100), squared=True
            )[0]

        # find peaks in corrected signal
        peaks, peak_properties = find_peaks(
            shoulder_diff_change, height=min_peak_height, distance=min_peak_distance
        )
        widths, width_heights, left_ips, right_ips = peak_widths(
            x=shoulder_diff_change, peaks=peaks, rel_height=relative_height
        )

        # create mask to indicate straight walking segments (straight=False, turn=True)
        turn_mask = np.zeros_like(right_shoulder_position)
        for left_base_idx, right_base_idx in zip(
            np.floor(left_ips).astype(int), np.ceil(right_ips).astype(int)
        ):
            turn_mask[left_base_idx:right_base_idx] = 1

        # create mask to indicate front vs back perspectives (front=True, back=False)
        perspective = np.where(left_shoulder_position > right_shoulder_position, "front", "back")
        turn_mask = np.where(turn_mask == 1, "turn", "straight")

        # assign indices to straight segments
        segment_idx = 0
        segment_idices = np.empty_like(perspective_change)
        for frame_idx, change in enumerate(perspective_change):
            if change != 0:
                segment_idx += 1
            segment_idices[frame_idx] = segment_idx

        # record boundary indices of segments
        segment_boundary_indices = np.concat(
            ([0], np.flatnonzero(perspective_change), [len(perspective_change)])
        )
        segment_boundaries = {}
        for idx, boundary in enumerate(segment_boundary_indices):
            if idx < len(segment_boundary_indices) - 1:
                segment_boundaries[idx] = [boundary, segment_boundary_indices[idx + 1]]

        # PLOT
        if self.debug_folder:

            plt.close("all")
            fig, axs = plt.subplots(4, 1, figsize=(14, 6))

            axs[0].plot(shoulder_diff_change, label="shoulder diff")
            axs[0].plot(left_ips, width_heights, "o", label="turn start", markersize=4)
            axs[0].plot(right_ips, width_heights, "o", label="turn end", markersize=4)
            axs[0].plot(turn_mask, label="straight/turn")
            axs[0].set_title("turn mask")

            axs[1].plot(left_shoulder_position, label="left")
            axs[1].plot(right_shoulder_position, label="right")
            axs[1].set_title("shoulder position (horizontal)")

            axs[2].plot(perspective, label="perspective")
            axs[2].set_title("Perspective")

            axs[3].plot(segment_idices, label="segment indices")
            axs[3].set_title("segment indices")
            axs[3].set_xlabel("Time [samples]")

            for ax in axs[:-2]:
                ax.legend(loc="upper left")
                ax.grid()

            plt.tight_layout()

            out_filename = os.path.join(
                self.debug_folder, f"{self.recording_type}_turns_and_perspective.png"
            )
            plt.savefig(out_filename)

            print(f"TPS figure saved to: \n{out_filename}")

        return (turn_mask, perspective, segment_idices, segment_boundaries)

    def get_gait_events(self, velocity, create_debug_fig: bool = False):
        ankle_thr = self.heel_thr * velocity

        all_events = []

        for side in ["left", "right"]:
            ankle = self.keypoints[:, self.labels.index(f"{side}_ankle"), :]

            ankle_vel = (
                np.linalg.norm(np.diff(ankle, axis=0, prepend=ankle[[0], :]), axis=1) * self.fps
            )

            ground_contact_thrs = (ankle_vel < ankle_thr).astype(int)

            # remove events from turn segments
            ground_contact_straight = np.where(self.turn_mask == "straight", ground_contact_thrs, 1)

            contact_diff_straight = np.diff(
                ground_contact_straight, prepend=ground_contact_straight[0]
            )

            # remove too short swing periods
            ground_conact_no_short_swing = np.copy(ground_contact_straight)
            starts = np.nonzero(contact_diff_straight == -1)[0]
            ends = np.nonzero(contact_diff_straight == 1)[0]

            for start_frame, end_frame in zip(starts, ends):
                if end_frame - start_frame < self.min_swing_time:
                    ground_conact_no_short_swing[start_frame:end_frame] = 1

            contact_diff_no_short_swing = np.diff(
                ground_conact_no_short_swing, prepend=ground_conact_no_short_swing[0]
            )

            # remove too short contact periods
            ground_contact_no_short_contact = np.copy(ground_conact_no_short_swing)
            starts = np.nonzero(contact_diff_no_short_swing == 1)[0]
            ends = np.nonzero(contact_diff_no_short_swing == -1)[0]
            for idx, [start_frame, end_frame] in enumerate(zip(starts, ends)):
                if idx + 1 < ends.shape[0]:
                    if ends[idx + 1] - start_frame < self.min_stance_time:
                        ground_contact_no_short_contact[start_frame : ends[idx + 1]] = 0

            contact_diff_no_short_contact = np.diff(
                ground_contact_no_short_contact, prepend=ground_contact_no_short_contact[0]
            )

            IC_frames = np.nonzero(contact_diff_no_short_contact == 1)[0]
            FC_frames = np.nonzero(contact_diff_no_short_contact == -1)[0]

            IC_events = [
                GaitEvent(
                    type="IC",
                    frame=frame,
                    side=side,
                    perspective=self.perspective[frame],
                    segment=self.turn_mask[frame],
                    segment_index=self.segment_indices[frame],
                    relative_position=self.calculate_relative_position(frame_idx=frame),
                )
                for frame in IC_frames
            ]

            FC_events = [
                GaitEvent(
                    type="FC",
                    frame=frame,
                    side=side,
                    perspective=self.perspective[frame],
                    segment=self.turn_mask[frame],
                    segment_index=self.segment_indices[frame],
                    relative_position=self.calculate_relative_position(frame_idx=frame),
                )
                for frame in FC_frames
                if frame < len(self.turn_mask)
            ]

            all_events.extend(IC_events + FC_events)

            # PLOT
            if create_debug_fig:
                plt.close("all")
                fig, axs = plt.subplots(4, 1, figsize=(14, 7))

                axs[0].plot(ankle_vel, label="velocity")
                axs[0].plot(ground_contact_thrs * ankle_thr, label="ground contact at thr")
                # axs[0].plot(self.turn_mask,'r--')

                axs[1].plot(ground_contact_thrs + 6, label="raw", color="#ff8800ff")
                axs[1].plot(ground_contact_straight + 4, label="no turns", color="#ffd000ff")
                axs[1].plot(
                    ground_conact_no_short_swing + 2, label="no short swings", color="#d9e717"
                )
                axs[1].plot(
                    ground_contact_no_short_contact, label="no short contacts", color="#6dc916"
                )

                axs[2].plot(contact_diff_straight + 4, label="no turns", color="#ffd000ff")
                axs[2].plot(
                    contact_diff_no_short_swing + 2, label="no short swings", color="#d9e717"
                )
                axs[2].plot(
                    contact_diff_no_short_contact, label="no short contact", color="#6dc916"
                )

                axs[3].plot(self.perspective, label="perspective")

                # fmt: off
                axs[0].set_title("ankle velocity")
                axs[1].set_title("ground contact")
                axs[2].set_title("ground contact diff (GEs)")

                axs[0].set_yticks([0, 3, ankle_thr])
                axs[0].set_ylim([0,3])
                # fmt: on
                for ax in axs:
                    ax.legend(loc="upper left")
                    ax.grid()

                plt.tight_layout()
                out_filename = os.path.join(
                    self.debug_folder, f"{self.recording_type}_gait_events_{side}.png"
                )
                plt.savefig(out_filename)
                plt.close("all")
                print(f"GE figure saved to: \n{out_filename}")
        all_events_sorted = sorted(all_events, key=lambda event: event.frame)

        return all_events_sorted

    def create_gait_cycles_figure(self):
        if self.debug_folder:
            plt.close("all")
            fig, axs = plt.subplots(3, 1, figsize=(14, 5))

            # show GEs of all gait cycles
            for gc in self.all_gait_cycles:
                alpha = 1 if gc.flags.valid else 0.25
                if gc.IC0:
                    axs[0].vlines(gc.IC0.frame, 0, 0.9, "c", linewidth=3, alpha=alpha)
                if gc.IC2:
                    axs[0].vlines(gc.IC0.frame, 1.1, 1.9, "m", linewidth=3, alpha=alpha)
                if gc.IC1:
                    axs[0].vlines(gc.IC0.frame, 2.1, 2.9, "r", linewidth=3, alpha=alpha)
                if gc.FC1:
                    axs[0].vlines(gc.IC0.frame, 3.1, 3.9, "b", linewidth=3, alpha=alpha)
                if gc.FC0:
                    axs[0].vlines(gc.IC0.frame, 4.1, 4.9, "g", linewidth=3, alpha=alpha)

            axs[0].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5], ["IC0", "IC2", "IC1", "FC1", "FC0"])
            axs[0].set_xlim([0, len(self.perspective)])
            axs[0].set_title("gait cycle composition")

            # show flags of all gait cycles
            for gc in self.all_gait_cycles:
                if gc.flags.valid:
                    axs[1].vlines(gc.IC0.frame, 0, 0.9, "gray", linewidth=3)
                if gc.flags.perspectives_ok:
                    axs[1].vlines(gc.IC0.frame, 1.1, 1.9, "gray", linewidth=3)
                if gc.flags.segments_ok:
                    axs[1].vlines(gc.IC0.frame, 2.1, 2.9, "gray", linewidth=3)
                if gc.flags.types_ok:
                    axs[1].vlines(gc.IC0.frame, 3.1, 3.9, "gray", linewidth=3)
                if gc.flags.sides_ok:
                    axs[1].vlines(gc.IC0.frame, 4.1, 4.9, "gray", linewidth=3)

            axs[1].set_yticks(
                [0.5, 1.5, 2.5, 3.5, 4.5], ["valid", "persp", "segments", "types", "sides"]
            )
            axs[1].set_xlim([0, len(self.perspective)])
            axs[1].set_title("gait cycle flags")

            axs[2].plot(self.perspective)
            axs[2].plot(self.turn_mask)
            axs[2].set_xlim([0, len(self.perspective)])

            plt.tight_layout()
            # plt.show()

            out_filename = os.path.join(self.debug_folder, f"{self.recording_type}_gait_cycles.png")
            plt.savefig(out_filename)
            print(f"Gait cycles figure saved to: \n{out_filename}")

    def calculate_relative_position(self, frame_idx: int) -> float:
        """
        Calculates the relative (compared to segment start) position of GE within the segment it belongs to.

        Args:
            frame_idx: frame of the GE

        Returns:
            relative_position: position of GE's frame within the segment defined by self.segment_boundaries
        """
        for segment_idx, (start, end) in self.segment_boundaries.items():
            if start < frame_idx < end:
                segment_length = end - start
                relative_position = (frame_idx - start) / segment_length
                return relative_position
