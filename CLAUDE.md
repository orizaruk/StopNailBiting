# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stop Nail Biting is a Windows desktop application that detects nail-biting behavior via webcam and triggers alerts (visual flash + audio) to help users break the habit. It runs in the system tray and uses MediaPipe for real-time hand and face tracking.

## Commands

### Run from source
```powershell
python main.py          # or: python -m stopnailbiting
```

### Build Windows executable
```powershell
pip install pyinstaller
pyinstaller stopnailbiting.spec
# Output: dist/StopNailBiting.exe
```

### Install dependencies
```powershell
pip install -r requirements.txt
```

## Architecture

The app is a Python package (`stopnailbiting/`) with a thin `main.py` entry-point
shim that calls `stopnailbiting.app.main()`. Each module owns one concern:

| Module | Responsibility |
|---|---|
| `app.py` | `main()` — builds the components (dependency injection) and runs the engine |
| `config.py` | `ConfigManager` — persistent settings |
| `camera.py` | `CameraManager` — webcam enumeration (ffmpeg/WMI) + capture |
| `audio.py` | `SoundManager` — alert sound |
| `alert.py` | `RedFlashAlert` — fullscreen red flash windows |
| `media.py` | `MediaController` — pause/resume system media during alerts |
| `controller.py` | `AppController` — tray icon, menu, app state, startup integration |
| `detection.py` | `DetectionEngine` — the per-frame loop and alert state machine |
| `geometry.py` | `polygon_contains_buffered` — point-in-polygon hit test |
| `constants.py` | tuning constants, model paths, landmark indices, MediaPipe option builders |
| `resources.py` | `resource_path`, `find_sound_file` (dev + PyInstaller paths) |
| `platform_windows.py` | Windows-specific helpers (DPI awareness) |

Detection state lives on the `DetectionEngine` instance (not module globals), and
components are wired together in `app.main()`.

### Classes
- **ConfigManager** (`config.py`): Persistent settings stored in `%APPDATA%\StopNailBiting\config.json`
- **RedFlashAlert** (`alert.py`): Fullscreen red warning windows on all monitors using tkinter
- **SoundManager** (`audio.py`): Audio alerts via miniaudio (looped playback with software volume control), falling back to a winsound beep on Windows
- **MediaController** (`media.py`): Pauses/resumes system media (Windows SMTC + pycaw) during alerts
- **AppController** (`controller.py`): App state, system tray icon (pystray), and Windows startup integration
- **DetectionEngine** (`detection.py`): Owns the webcam loop, MediaPipe inference, and the alert state machine

### Detection Pipeline
1. Capture webcam frame with OpenCV
2. Run MediaPipe hand detection (`hand_landmarker.task` model)
3. If hands found, run face detection (`face_landmarker.task` model)
4. Check if fingertip landmarks intersect with the lip polygon (custom NumPy/stdlib point-in-polygon test via `polygon_contains_buffered`, replacing Shapely)
5. Apply z-depth check to filter false positives (fingers passing in front of face)
6. Require `FRAMES_REQUIRED` consecutive positive frames before triggering alert
7. Apply `COOLDOWN_PERIOD` before hiding alert after detection stops

### Key Constants (in `stopnailbiting/constants.py`)
- `SENSITIVITY`: Lip polygon buffer size
- `Z_DEPTH_THRESHOLD`: Max z-difference for valid detection
- `FRAMES_REQUIRED`: Consecutive frames needed before alert
- `TARGET_FPS`: Frame rate cap (15 FPS)
- `COOLDOWN_PERIOD`: Seconds alert stays after detection stops

### External Assets
- `models/hand_landmarker.task`: MediaPipe hand detection model
- `models/face_landmarker.task`: MediaPipe face detection model
- `assets/noise.wav`: Alert sound file

### Landmark Indices
- `LIP_INDICES`: MediaPipe face landmark indices forming the mouth polygon
- `HAND_INDICES`: MediaPipe hand landmark indices for fingertips and adjacent joints
