"""Pinned local runtime compatibility, including actual protobuf output packets."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import warnings

from core.pose import load_mediapipe


class PoseRuntimeTests(unittest.TestCase):
    def test_only_known_native_container_warnings_are_scoped_out(self):
        def import_with_known_warnings(name):
            for container in ('MessageMapContainer', 'ScalarMapContainer'):
                warnings.warn(
                    f'Type google._upb._message.{container} uses PyType_Spec with a metaclass '
                    'that has custom tp_new. This is deprecated and will no longer be allowed '
                    'in Python 3.14.', DeprecationWarning)
            return 'loaded'
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            original_filters = warnings.filters[:]
            with patch('core.pose.importlib.import_module', import_with_known_warnings):
                self.assertEqual(load_mediapipe(), 'loaded')
            self.assertEqual(warnings.filters, original_filters)

    def test_unrelated_import_warnings_still_fail_strict_policy(self):
        def import_with_unrelated_warning(name):
            warnings.warn('Unrelated dependency deprecation', DeprecationWarning)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with patch('core.pose.importlib.import_module', import_with_unrelated_warning):
                with self.assertRaisesRegex(DeprecationWarning, 'Unrelated dependency'):
                    load_mediapipe()

    @unittest.skipUnless(importlib.util.find_spec('mediapipe'), 'MediaPipe not installed')
    def test_fresh_process_strict_import_and_landmark_packet_roundtrip(self):
        # A new interpreter must hit the C-extension import, not sys.modules.
        # Nonempty packets exercise the protobuf path that blank pose frames omit.
        result = subprocess.run(
            [sys.executable, '-W', 'error', '-c', '''
from core.pose import load_mediapipe
mp = load_mediapipe()
from mediapipe.framework.formats import landmark_pb2
for message_type in (landmark_pb2.LandmarkList, landmark_pb2.NormalizedLandmarkList):
    landmarks = message_type()
    for i in range(33):
        landmarks.landmark.add(x=i / 33, y=0.5, z=-0.2, visibility=0.9)
    packet = mp.packet_creator.create_proto(landmarks)
    assert mp.packet_getter.get_proto(packet) == landmarks
'''], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
