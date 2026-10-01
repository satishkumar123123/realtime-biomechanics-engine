import numpy as np


class BiomechanicsEngine:
    @staticmethod
    def calculate_angle_3d(a, b, c):
        """
        Calculates 3D interior angle at vertex b between vectors ba and bc.
        """
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine_angle)))

    @classmethod
    def compute_joint_metrics(cls, world_landmarks, visibilities):
        """
        Calculates mandatory angles following Goniometric standard (0 deg = neutral).
        Returns None for a joint if visibility is below threshold.
        """
        metrics = {}
        conf_thresh = 0.65

        def is_reliable(*indices):
            return all(visibilities.get(i, 0.0) >= conf_thresh for i in indices)

        # (name, shoulder/hip/knee landmark triplet)
        joints = {
            # Elbow Flexion (Shoulder, Elbow, Wrist)
            'L_Elbow_Flex': (11, 13, 15),
            'R_Elbow_Flex': (12, 14, 16),
            # Knee Flexion (Hip, Knee, Ankle)
            'L_Knee_Flex': (23, 25, 27),
            'R_Knee_Flex': (24, 26, 28),
            # Hip Flexion (Shoulder, Hip, Knee)
            'L_Hip_Flex': (11, 23, 25),
            'R_Hip_Flex': (12, 24, 26),
        }

        for name, (i, j, k) in joints.items():
            if is_reliable(i, j, k):
                angle = cls.calculate_angle_3d(
                    world_landmarks[i], world_landmarks[j], world_landmarks[k]
                )
                metrics[name] = max(0.0, 180.0 - angle)
            else:
                metrics[name] = None

        return metrics
