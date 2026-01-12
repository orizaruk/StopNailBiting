# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nail-biting detection application using computer vision. Runs silently in the background with a system tray icon and alerts with sound + fullscreen red flash when nail-biting is detected via webcam. Designed to help people break their nail-biting habits through immediate feedback.

## Features (for README)

- **Real-time Detection**: Uses MediaPipe hand and face tracking to detect when fingers approach mouth
- **Dual Alert System**: Configurable fullscreen red flash + audio alert
- **System Tray Integration**: Runs quietly in background, all controls via tray icon
- **Persistent Settings**: Preferences saved and restored between sessions
- **Volume Control**: Adjustable alert volume (25%, 50%, 75%, 100%)
- **Auto-Start**: Optional launch on Windows startup
- **Pause/Resume**: Temporarily disable detection without closing app
- **Low Resource Usage**: 15 FPS cap, hand-detection-first optimization
- **Standalone Executable**: No Python installation required for end users

## System Requirements (for README)

- Windows 10/11
- Webcam
- ~200MB disk space (for standalone .exe)
- ~300-400MB RAM while running

## Dependencies

- mediapipe: Hand and face landmark detection
- opencv-python (cv2): Webcam capture and image processing
- shapely: Geometric polygon operations for containment checks
- pygame: Sound playback for alerts
- tkinter: Fullscreen red flash overlay (included with Python)
- pystray: System tray icon integration
- Pillow: Icon image generation
- pyinstaller: Building standalone Windows executable

## Setup (Development)

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

## Building Standalone Executable

PyInstaller is used to create a standalone Windows .exe:

```bash
# Must be run on Windows (not WSL) with native Windows paths
pyinstaller stopnailbiting.spec
# Output: dist/StopNailBiting.exe (~150-200MB)
```

Note: Build from a native Windows directory (e.g., `C:\projects\stopnailbiting`), not WSL filesystem paths, to avoid DLL resolution issues with Shapely.

## User Configuration

### Config File Location

- **Windows**: `%APPDATA%\StopNailBiting\config.json`
- **Linux**: `~/.config/StopNailBiting/config.json`

### Config File Format

```json
{
  "flash_enabled": true,
  "sound_enabled": true,
  "start_with_windows": false,
  "volume": 0.75
}
```

### Settings Descriptions

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `flash_enabled` | bool | true | Show fullscreen red flash on detection |
| `sound_enabled` | bool | true | Play alert sound on detection |
| `start_with_windows` | bool | false | Launch app automatically on Windows login |
| `volume` | float | 0.75 | Alert volume (0.25, 0.50, 0.75, or 1.0) |

## Architecture

### Detection Pipeline

```
Webcam Frame → Hand Detection → Face Detection (if hands found) → Lip Polygon → Z-Depth + 2D Check → Temporal Filter → Alert
```

### Core Classes

| Class | Purpose |
|-------|---------|
| `ConfigManager` | Loads/saves settings to JSON, handles defaults |
| `RedFlashAlert` | Fullscreen topmost red window with warning text |
| `SoundManager` | Alert sound playback with volume control |
| `AppController` | App state, system tray, and user settings UI |

### System Tray Menu

```
Right-click menu:
├── ✓ Enable Flash      (checkbox)
├── ✓ Enable Sound      (checkbox)
├── Volume ►
│       ├── ○ 25%
│       ├── ○ 50%
│       ├── ● 75%       (radio selection)
│       └── ○ 100%
├── ○ Start with Windows (checkbox)
├── ─────────────────────
├── Pause / Resume
└── Quit
```

### Tray Icon States

| Icon | Meaning |
|------|---------|
| Red circle | Actively monitoring |
| Gray circle | Paused |

### Threading Model

- **Main thread**: Detection loop + tkinter window updates
- **Background thread**: System tray icon (pystray)
- Shared state via `AppController` (running, paused flags)

### Detection Logic

1. **Hand detection first** - Skip face detection if no hands (performance optimization)
2. **2D containment** - Check if finger points fall inside lip polygon
3. **Z-depth filtering** - Verify finger is at similar depth to lips (filters out fingers passing in front of face)
4. **Temporal consistency** - Require N consecutive frames before triggering (filters momentary glitches)
5. **Cooldown** - Keep alert active for a period after detection stops (prevents flickering)

### Key Constants (main.py)

```python
SENSITIVITY = 0.01        # Lip polygon buffer size
Z_DEPTH_THRESHOLD = 0.1   # Max z-difference for valid detection
FRAMES_REQUIRED = 3       # Consecutive frames before alert triggers
TARGET_FPS = 15           # Frame rate cap to reduce CPU usage
COOLDOWN_PERIOD = 1.5     # Seconds to keep alert after biting stops
```

### Model Files

Pre-trained MediaPipe models in `models/`:
- `hand_landmarker.task` - Hand landmark detection (~7.5 MB)
- `face_landmarker.task` - Face landmark detection (~3.6 MB)

### Landmark Indices

- `HAND_INDICES = [4, 3, 8, 7, 12, 11, 16, 15, 20, 19]` - Finger tips and adjacent joints
- `LIP_INDICES` - 21 points forming mouth polygon boundary

## File Structure

```
stopnailbiting/
├── main.py                 # Main application (~550 lines)
├── stopnailbiting.spec     # PyInstaller build configuration
├── requirements.txt        # Python dependencies
├── CLAUDE.md               # This file
├── models/                 # MediaPipe model files
│   ├── hand_landmarker.task
│   └── face_landmarker.task
├── assets/
│   └── noise.wav           # Alert sound
└── landmark_image_maker.py # Debug tool for visualizing landmarks
```

## Auto-Start Implementation

When "Start with Windows" is enabled:
- Creates shortcut: `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\StopNailBiting.lnk`
- Uses PowerShell COM object to create .lnk file
- Points to the .exe location with working directory set

## Development Notes

- Test images go in `assets/` directory (gitignored)
- Alert sound file: `assets/noise.wav`
- Webcam development done on Windows (WSL has hardware access limitations)
- Use `landmark_image_maker.py` to visualize landmark positions for debugging
- `resource_path()` helper ensures paths work both in dev and PyInstaller bundles
- Config changes are saved immediately when toggled via tray menu

## Known Limitations

- Windows only (system tray and auto-start are Windows-specific)
- Requires webcam access
- First launch of .exe may be slow (~5-10s) as PyInstaller extracts files
- Some antivirus software may flag PyInstaller bundles (false positive)
