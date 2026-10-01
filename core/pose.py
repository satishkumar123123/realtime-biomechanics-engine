"""Local BlazePose model construction, independent of camera, geometry and UI.

The pinned MediaPipe 0.10.21 wheel supplies the Full model and detector. Model
creation is lazy so geometry/filter tests can run without importing MediaPipe.
"""


def create_pose():
    """Construct the local Full pose model with application-owned smoothing.

    The caller owns ``close()``. Process RGB frames through ``process()`` and time
    that call separately from preprocessing/rendering in the pipeline runner.
    """
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        raise RuntimeError('Legacy Pose API unavailable; install requirements.txt in a fresh venv')
    return mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1, smooth_landmarks=False,
        enable_segmentation=False, min_detection_confidence=0.6,
        min_tracking_confidence=0.6)
