# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nail-biting detection application using computer vision. Runs in the background and alerts with sound + fullscreen red flash when nail-biting is detected via webcam.

## Dependencies

- mediapipe: Hand and face landmark detection
- opencv-python (cv2): Webcam capture and image processing
- shapely: Geometric polygon operations for containment checks
- pygame: Sound playback for alerts
- tkinter: Fullscreen red flash overlay

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py            # Main application - live webcam detection with alerts
python landmark_image_maker.py  # Visualize landmarks with indices (debugging)
```

## Architecture

Detection pipeline:

```
Webcam Frame → Hand Detection → Face Detection (if hands found) → Lip Polygon → Z-Depth + 2D Check → Temporal Filter → Alert
```

### Detection Logic

1. **Hand detection first** - Skip face detection if no hands (performance optimization)
2. **2D containment** - Check if finger points fall inside lip polygon
3. **Z-depth filtering** - Verify finger is at similar depth to lips (filters out fingers passing in front of face)
4. **Temporal consistency** - Require N consecutive frames before triggering (filters momentary glitches)
5. **Cooldown** - Keep alert active for a period after detection stops (prevents flickering)

### Key Configuration (main.py)

```python
SENSITIVITY = 0.01        # Lip polygon buffer size
Z_DEPTH_THRESHOLD = 0.1   # Max z-difference for valid detection
FRAMES_REQUIRED = 3       # Consecutive frames before alert triggers
COOLDOWN_PERIOD = 1.5     # Seconds to keep alert after biting stops
```

### Alert System

- `RedFlashAlert` class: Fullscreen topmost red window with warning text
- `SoundManager` class: Loops alert sound while detection active

### Model Files

Pre-trained MediaPipe models in `models/`:
- `hand_landmarker.task` - Hand landmark detection
- `face_landmarker.task` - Face landmark detection

### Landmark Indices

- `HAND_INDICES = [4, 3, 8, 7, 12, 11, 16, 15, 20, 19]` - Finger tips and adjacent joints
- `LIP_INDICES` - 21 points forming mouth polygon boundary

## Development Notes

- Test images go in `assets/` directory (gitignored)
- Alert sound file: `assets/noise.mp3`
- Webcam development done on Windows (WSL has hardware access limitations)
- Use `landmark_image_maker.py` to visualize landmark positions for debugging
