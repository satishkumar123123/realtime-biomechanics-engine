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
