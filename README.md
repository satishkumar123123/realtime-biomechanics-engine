# Real-Time Biomechanical Analysis System

A real-time monocular biomechanical analysis system computing physiological joint angles using MediaPipe BlazePose and Goniometric neutral-zero standards.

## Project Structure
- `core/biomechanics.py`: Joint angle vector computation and goniometric calibration.
- `main.py`: Webcam ingestion, model inference, and real-time visualization HUD.

## Setup Instructions

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the baseline engine:
```bash
python main.py
```

## Asynchronous capture

`core/capture.py` provides a daemon producer with a lock-protected latest-frame
mailbox. It uses only OpenCV, NumPy, and the Python standard library; the existing
MediaPipe dependency is still required by the baseline analysis application.
The baseline `main.py` remains a synchronous reference; use this module for the
next processing-pipeline integration.

```python
from core.capture import VideoCaptureAsync

with VideoCaptureAsync(width=640, height=480, fps=60) as camera:
    sequence = 0
    while True:
        sample = camera.wait_for_frame(sequence, timeout=1.0)
        if sample is None:
            raise RuntimeError(f"Camera unavailable: {camera.error}")
        sequence = sample.sequence
        frame = sample.frame  # Writable copy, safe for overlays/inference.
        # Process frame here; capture continues independently.
```

`read()` returns `(available, frame)` immediately and may repeat the latest frame.
`snapshot()` adds a sequence and monotonic timestamp; `wait_for_frame()` avoids
processing duplicates. Copies occur outside the mailbox lock. `copy=False`
borrows a read-only frame; consumers must not change it. Width, height, FPS and
backend are configurable at construction (create a new instance to reconfigure).
Inspect `actual_settings` for driver negotiation. The FPS request does not force
a device to support 60 FPS. Timestamps measure host receipt, not sensor exposure.

This bounded mailbox intentionally replaces unconsumed frames to avoid backlog.
It continuously polls the camera but does not promise zero hardware drops or
lossless processing of every frame. `stop()` joins the worker and releases the
camera. If native camera I/O blocks longer than the shutdown timeout, it raises
`TimeoutError`; cleanup completes when the read returns. Python cannot safely
cancel every camera backend's native read.

Run hardware-independent tests and a paced synthetic 300-frame benchmark:

```bash
python -m unittest tests.test_capture -v
python -m tests.benchmark_capture --synthetic --frames 300
```

Verify the actual webcam separately:

```bash
python -m tests.benchmark_capture --camera 0 --frames 300 --fps 60
```

The benchmark reports producer/consumer FPS, superseded frames, host frame age,
and mean/p99/max mailbox access time. It fails if measured maximum mailbox access
is at least 1 ms. Sensor wait and frame-copy costs are excluded from that lock
overhead measurement; it is a measured check, not a hard real-time guarantee.
