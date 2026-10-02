"""Local BlazePose model construction, independent of camera, geometry and UI.

The pinned MediaPipe 0.10.21 wheel supplies the Full model and detector. Model
creation is lazy so geometry/filter tests can run without importing MediaPipe.
"""
import importlib
import warnings


def load_mediapipe():
    """Import the pinned runtime with a scoped protobuf 4.x compatibility rule.

    Python 3.12 deprecates the native map-container metaclass used by protobuf
    <5 (required by MediaPipe 0.10.21). Only those two exact import warnings are
    ignored; other warnings retain the caller's policy, including -W error.
    Upstream issue: https://github.com/protocolbuffers/protobuf/issues/15077
    No inference warnings are suppressed and no global filters are installed.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', category=DeprecationWarning,
            message=(r'^Type google\._upb\._message\.(MessageMapContainer|ScalarMapContainer) '
                     r'uses PyType_Spec with a metaclass that has custom tp_new\. '
                     r'This is deprecated and will no longer be allowed in Python 3\.14\.$'))
        return importlib.import_module('mediapipe')


def create_pose():
    """Construct the local Full pose model with application-owned smoothing.

    The caller owns ``close()``. Process RGB frames through ``process()`` and time
    that call separately from preprocessing/rendering in the pipeline runner.
    """
    mp = load_mediapipe()
    if not hasattr(mp, 'solutions'):
        raise RuntimeError('Legacy Pose API unavailable; install requirements.txt in a fresh venv')
    return mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1, smooth_landmarks=False,
        enable_segmentation=False, min_detection_confidence=0.6,
        min_tracking_confidence=0.6)
