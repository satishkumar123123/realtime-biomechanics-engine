"""Neutral-zero geometric joint estimates from BlazePose world landmarks.

Body-frame mode constructs orthonormal left/down/posterior axes from the torso;
its X-Y and Y-Z projections follow subject orientation. Camera-frame mode retains
fixed X-Y/Y-Z planes and requires a front-aligned subject. Both are geometric
proxies: inferred depth, trunk motion and landmark error affect clinical accuracy.
"""

from typing import Optional

import numpy as np


class BiomechanicsEngine:
    """Stateless, visibility-gated bilateral angle calculations in degrees."""

    EPSILON = 1e-7
    VISIBILITY_THRESHOLD = 0.65
    # Reject directions close to a plane normal, where tiny noise dominates angle.
    MIN_PROJECTION_RATIO = 0.05
    FRAME_LANDMARKS = (11, 12, 23, 24)

    @classmethod
    def visibility_reliable(cls, visibilities, indices):
        """All requested confidence values must be finite and in (0.65, 1]."""
        for index in indices:
            try:
                confidence = float(visibilities[index])
            except (KeyError, IndexError, TypeError, ValueError, OverflowError):
                return False
            if not np.isfinite(confidence) or not cls.VISIBILITY_THRESHOLD < confidence <= 1:
                return False
        return True

    @classmethod
    def body_frame(cls, world_landmarks, visibilities):
        """Return (hip midpoint, basis) or None for an unreliable torso.

        Basis columns are subject-left, torso-down and posterior. Down follows
        hip midpoint minus shoulder midpoint. A Gram-Schmidt lateral axis uses
        both bilateral spans; cross(left, down) defines posterior. Landmark
        identities determine signs, without camera-facing/yaw assumptions.
        Four visible torso anchors are required. Contradictory spans and nearly
        collinear torso/lateral directions are rejected instead of inventing axes.
        This frame estimates torso orientation, not clinical pelvic/scapular axes.
        """
        points = np.asarray(world_landmarks, dtype=np.float64)
        if points.shape != (33, 3):
            raise ValueError('world_landmarks must have shape (33, 3)')
        if (not cls.visibility_reliable(visibilities, cls.FRAME_LANDMARKS)
                or not np.all(np.isfinite(points[list(cls.FRAME_LANDMARKS)]))):
            return None
        with np.errstate(over='ignore', invalid='ignore'):
            shoulder_mid = points[11] / 2 + points[12] / 2
            hip_mid = points[23] / 2 + points[24] / 2
            shoulder_left = cls.normalize(points[11] - points[12])
            hip_left = cls.normalize(points[23] - points[24])
            down = cls.normalize(hip_mid - shoulder_mid)
        if shoulder_left is None or hip_left is None or down is None:
            return None
        # >60-degree disagreement suggests twisted/incorrect torso estimates.
        if cls.normalized_dot_product(shoulder_left, hip_left) < 0.5:
            return None
        lateral = cls.normalize(shoulder_left + hip_left)
        orthogonal = lateral - np.dot(lateral, down) * down
        if cls.euclidean_norm(orthogonal) < 0.2:
            return None
        left = cls.normalize(orthogonal)
        posterior = cls.normalize(cls.cross_product(left, down))
        return hip_mid, np.column_stack((left, down, posterior))

    @staticmethod
    def _vector(value):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,):
            raise ValueError("Expected a three-component vector")
        return vector

    @classmethod
    def euclidean_norm(cls, vector) -> Optional[float]:
        """Return finite Euclidean length, or None for nonfinite components."""
        vector = cls._vector(vector)
        if not np.all(np.isfinite(vector)):
            return None
        # hypot avoids overflow/underflow from squaring raw coordinates.
        norm = float(np.hypot.reduce(vector))
        return norm if np.isfinite(norm) else None

    @classmethod
    def normalize(cls, vector) -> Optional[np.ndarray]:
        """Return unit vector; lengths <= epsilon have no defined direction."""
        vector = cls._vector(vector)
        norm = cls.euclidean_norm(vector)
        if norm is None or norm <= cls.EPSILON:
            return None
        return vector / norm

    @classmethod
    def normalized_dot_product(cls, first, second) -> Optional[float]:
        """Cosine of the angle, clamped to [-1, 1]; None if degenerate."""
        first, second = cls.normalize(first), cls.normalize(second)
        if first is None or second is None:
            return None
        return float(np.clip(np.dot(first, second), -1.0, 1.0))

    @classmethod
    def cross_product(cls, first, second) -> Optional[np.ndarray]:
        """Finite 3D cross product; None if inputs or result are nonfinite."""
        first, second = cls._vector(first), cls._vector(second)
        if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
            return None
        with np.errstate(over='ignore', invalid='ignore'):
            result = np.cross(first, second)
        return result if np.all(np.isfinite(result)) else None

    @classmethod
    def project_coronal(cls, vector):
        """Project a vector onto X-Y by removing its Z component."""
        vector = cls._vector(vector).copy()
        vector[2] = 0.0
        return vector

    @classmethod
    def project_sagittal(cls, vector):
        """Project a vector onto Y-Z by removing its X component."""
        vector = cls._vector(vector).copy()
        vector[0] = 0.0
        return vector

    @classmethod
    def calculate_angle_3d(cls, a, b, c) -> Optional[float]:
        """Interior angle at b in [0, 180]; None for invalid/zero segments."""
        with np.errstate(over='ignore', invalid='ignore'):
            cosine = cls.normalized_dot_product(cls._vector(a) - cls._vector(b),
                                                cls._vector(c) - cls._vector(b))
        return None if cosine is None else float(np.degrees(np.arccos(cosine)))

    @classmethod
    def _signed_projected_angle(cls, reference, moving, plane, sign):
        """Oriented angle after projection, using unit-vector dot and cross."""
        project = cls.project_coronal if plane == 'coronal' else cls.project_sagittal
        projected = []
        for vector in (reference, moving):
            length = cls.euclidean_norm(vector)
            in_plane = project(vector)
            in_plane_length = cls.euclidean_norm(in_plane)
            if (length is None or in_plane_length is None or length <= cls.EPSILON
                    or in_plane_length / length < cls.MIN_PROJECTION_RATIO):
                return None
            projected.append(cls.normalize(in_plane))
        reference, moving = projected
        if reference is None or moving is None:
            return None
        cosine = cls.normalized_dot_product(reference, moving)
        cross = cls.cross_product(reference, moving)
        normal_axis = 2 if plane == 'coronal' else 0
        sine = float(np.clip(sign * cross[normal_axis], -1.0, 1.0))
        # The opposite ray has no directional sign; avoid signed-zero -180 ties.
        if cosine < 0 and abs(sine) <= cls.EPSILON:
            return 180.0
        angle = float(np.degrees(np.arctan2(sine, cosine)))
        return 0.0 if abs(angle) < cls.EPSILON else angle

    @classmethod
    def compute_joint_metrics(cls, world_landmarks, visibilities, *,
                              anterior_z_sign=-1, left_x_sign=1,
                              coordinate_frame='camera'):
        """Return 12 bilateral neutral-zero angles, or None per invalid joint.

        Inputs: numeric (33, 3) world coordinates in meters and a visibility
        mapping or 33-element sequence. Missing/nonfinite visibility is rejected;
        every triplet landmark must have confidence in (0.65, 1]. Nonfinite or
        coincident coordinates and zero projected segments also return None.

        Existing *_Elbow_Flex, *_Knee_Flex and *_Hip_Flex keys are retained.
        Elbow/knee report unsigned flexion (180 - interior); triplet geometry
        alone cannot distinguish hyperextension. Shoulder/hip flexion is positive
        anteriorly and negative in extension. Shoulder abduction is positive
        outward, negative across the trunk (adduction). Ankle is interior - 90:
        positive plantarflexion, negative dorsiflexion. Knee-ankle-foot-index is
        a geometric proxy and does not locate the clinical ankle axes.
        In body mode, shoulder/hip projections require four reliable torso
        anchors; elbow/knee/ankle remain independent of that frame. Coordinate
        signs apply only to camera mode; body signs follow landmark identities.
        Body-mode shoulder abduction uses torso-down as its coronal reference;
        a diagonal ipsilateral hip ray would introduce a neutral offset solely
        from differing shoulder and hip widths.
        """
        points = np.asarray(world_landmarks, dtype=np.float64)
        if points.shape != (33, 3):
            raise ValueError("world_landmarks must have shape (33, 3)")
        if anterior_z_sign not in (-1, 1) or left_x_sign not in (-1, 1):
            raise ValueError("Coordinate signs must be -1 or +1")
        if coordinate_frame not in ('camera', 'body'):
            raise ValueError("coordinate_frame must be 'camera' or 'body'")
        projected_points = points
        if coordinate_frame == 'body':
            frame = cls.body_frame(points, visibilities)
            projected_points = None
            if frame is not None:
                origin, basis = frame
                with np.errstate(over='ignore', invalid='ignore'):
                    projected_points = (points - origin) @ basis
            anterior_z_sign, left_x_sign = -1, 1

        def reliable(indices):
            if not np.all(np.isfinite(points[list(indices)])):
                return False
            return cls.visibility_reliable(visibilities, indices)

        metrics = {}
        for side, shoulder, elbow, wrist, hip, knee, ankle, foot, outward in (
                ('L', 11, 13, 15, 23, 25, 27, 31, left_x_sign),
                ('R', 12, 14, 16, 24, 26, 28, 32, -left_x_sign)):
            triplets = {
                'Elbow_Flex': (shoulder, elbow, wrist),
                'Knee_Flex': (hip, knee, ankle),
                'Shoulder_Flex': (hip, shoulder, elbow),
                'Shoulder_Abd': (hip, shoulder, elbow),
                'Hip_Flex': (shoulder, hip, knee),
                'Ankle_Dorsi_Plantar': (knee, ankle, foot),
            }
            for name, indices in triplets.items():
                value = None
                if reliable(indices):
                    a, b, c = points[list(indices)]
                    if name in ('Elbow_Flex', 'Knee_Flex', 'Ankle_Dorsi_Plantar'):
                        interior = cls.calculate_angle_3d(a, b, c)
                        if interior is not None:
                            value = (interior - 90.0 if name == 'Ankle_Dorsi_Plantar'
                                     else 180.0 - interior)
                    elif projected_points is None:
                        value = None
                    elif name == 'Hip_Flex':
                        a, b, c = projected_points[list(indices)]
                        # Reverse the trunk ray: downwards is neutral thigh direction.
                        with np.errstate(over='ignore', invalid='ignore'):
                            value = cls._signed_projected_angle(b - a, c - b,
                                                               'sagittal', anterior_z_sign)
                    else:
                        a, b, c = projected_points[list(indices)]
                        with np.errstate(over='ignore', invalid='ignore'):
                            reference = a - b
                            if coordinate_frame == 'body' and name == 'Shoulder_Abd':
                                reference = np.array((0.0, reference[1], 0.0))
                            value = cls._signed_projected_angle(
                                reference, c - b,
                                'coronal' if name == 'Shoulder_Abd' else 'sagittal',
                                -outward if name == 'Shoulder_Abd' else anterior_z_sign)
                metrics[f'{side}_{name}'] = value
        return metrics
