"""Build an explicitly augmented single-photo replay for real model timing.

This is image-space motion, not a recording of changing human joint angles.
The original image is an official checksum-pinned MediaPipe test asset. It is
downloaded only with --download; inference itself never accesses the network.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import cv2
import numpy as np


IMAGE_URL = 'https://storage.googleapis.com/mediapipe-assets/pose.jpg?generation=1678737494661975'
IMAGE_SHA256 = 'c8a830ed683c0276d713dd5aeda28f415f10cd6291972084a40d0d8b934ed62b'
ASSET_MANIFEST = ('https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/'
                  'third_party/external_files.bzl')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def replay_frame(image, index, frames, width=640, height=480):
    """Letterbox, then apply a cyclic small translation, scale and image roll."""
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height-resized.shape[0])//2, (width-resized.shape[1])//2
    canvas[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
    phase = 2*np.pi*index/frames
    matrix = cv2.getRotationMatrix2D((width/2, height/2), 2*np.sin(phase),
                                   1 + .03*np.sin(2*phase))
    matrix[:, 2] += (12*np.sin(2*phase), 8*np.sin(phase))
    return cv2.warpAffine(canvas, matrix, (width, height), flags=cv2.INTER_LINEAR)


def prepare(image_path, video_path, manifest_path, *, download=False, frames=720):
    """Create an MJPEG replay and provenance manifest; verify source checksum."""
    image_path, video_path, manifest_path = map(Path, (image_path, video_path, manifest_path))
    if len({p.resolve() for p in (image_path, video_path, manifest_path)}) != 3:
        raise ValueError('Image, video and manifest paths must be distinct')
    if not isinstance(frames, int) or frames < 2:
        raise ValueError('At least two replay frames are required')
    if download:
        with urllib.request.urlopen(IMAGE_URL, timeout=30) as response:
            data = response.read(2 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != IMAGE_SHA256:
            raise ValueError('Downloaded image checksum differs from the pinned source')
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(data)
    if not image_path.is_file() or sha256_file(image_path) != IMAGE_SHA256:
        raise ValueError('Pinned source image missing/mismatched; use --download or --image')
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError('Cannot decode source image')
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'MJPG'), 60, (640, 480))
    try:
        if not writer.isOpened():
            raise OSError('MJPEG video writer unavailable')
        for index in range(frames):
            writer.write(replay_frame(image, index, frames))
    finally:
        writer.release()
    manifest = {
        'schema_version': 1, 'workload_kind': 'augmented_single_photo_replay',
        'source': {'url': IMAGE_URL, 'sha256': IMAGE_SHA256, 'upstream_manifest': ASSET_MANIFEST},
        'video': {'file': video_path.name, 'sha256': sha256_file(video_path),
                  'width': 640, 'height': 480, 'fps': 60, 'frames': frames, 'codec': 'MJPG'},
        'recipe': {'phase': '2*pi*frame_index/frame_count', 'translation_x_px': '12*sin(2*phase)',
                   'translation_y_px': '8*sin(phase)', 'image_roll_deg': '2*sin(phase)',
                   'scale': '1+0.03*sin(2*phase)', 'resize': 'letterbox', 'opencv': cv2.__version__},
        'limitations': ['One source photograph; no changing human articulation or subject diversity',
                        'Image-space augmentation does not establish a real webcam workload',
                        'No calibrated ground-truth joint angles accompany the photograph'],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--image', type=Path, default=Path('benchmarks/assets/pose.jpg'))
    parser.add_argument('--video', type=Path, default=Path('benchmarks/assets/human_replay.avi'))
    parser.add_argument('--manifest', type=Path, default=Path('benchmarks/results/replay_manifest.json'))
    parser.add_argument('--frames', type=int, default=720)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.image, args.video, args.manifest, download=args.download, frames=args.frames)
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f'Replay preparation failed: {exc}\n')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
