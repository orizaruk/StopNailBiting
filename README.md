# Stop Nail Biting

Your guardian against nail-biting. This is a lightweight app that uses Google's MediaPipe models to spot nail-biting in real-time and make you STOP. It runs silently in the background, keeping you accountable at all times.

## How It Works

The app runs silently in your system tray, monitoring your webcam. When it detects nail biting, it triggers an irritating alert to make you stop.

By default, the alert is a flashing red screen and a BEEP noise but you can choose to disable either or both. 
LEAVING BOTH ON IS HIGHLY RECOMMENDED! :)

**Detection pipeline:**
1. Detects hands in the webcam frame using MediaPipe
2. If hands are found, detects face and lip position
3. Checks if fingertips are inside the mouth region (2D + depth check)
4. Filters out false positives with temporal consistency (multiple frames required)
5. Triggers alert (sound + visual flash)

## Features

- **Real-time Detection** - Uses MediaPipe hand and face tracking (thanks Google!)
- **Dual Alert System** - Fullscreen red flash + audio alert (both configurable, designed to be annoying and effective)
- **System Tray Integration** - Runs quietly in background
- **Persistent Settings** - Preferences saved between sessions
- **Volume Control** - Adjustable alert volume (25%, 50%, 75%, 100%)
- **Auto-Start** - Optional launch on Windows startup
- **Pause/Resume** - Temporarily disable without closing
- **Low Resource Usage** - Optimized to ~15 FPS with smart detection skipping

## Installation

### Option 1: Download Executable (Recommended)

1. Go to [Releases](../../releases)
2. Download `StopNailBiting.exe`
3. Run the executable - no installation required

### Option 2: Run from Source

```powershell
# Clone the repository
git clone https://github.com/orizaruk/stopnailbiting.git
cd stopnailbiting

# Create virtual environment
python -m venv venv

# Activate virtual environment
# For PowerShell:
.\venv\Scripts\Activate.ps1
# For Command Prompt (cmd):
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py
```

### Option 3: Build Executable from Source

```powershell
# Follow Option 2 steps first, then:

# Install PyInstaller
pip install pyinstaller

# Build (must be run on Windows)
pyinstaller stopnailbiting.spec

# Output: dist/StopNailBiting.exe
```

## Usage

1. **Launch the app** - A red circle icon appears in your system tray
2. **Position your webcam** - Ensure your face and hands are visible
3. **Work normally** - The app monitors in the background
4. **Get alerted** - When nail-biting is detected, you'll see/hear an alert
5. **Auto-start** - Make sure to mark "Start with Windows" if you want it to launch automatically on startup (RECOMMENDED!)

### System Tray Menu

Right-click the tray icon to access settings:

| Option | Description |
|--------|-------------|
| Enable Flash | Toggle fullscreen red flash alert |
| Enable Sound | Toggle audio alert |
| Volume | Set alert volume (25%, 50%, 75%, 100%) |
| Start with Windows | Auto-launch on login |
| Pause / Resume | Temporarily disable detection |
| Quit | Exit the application |

### Tray Icon States

| Icon | Status |
|------|--------|
| Red circle | Actively monitoring |
| Gray circle | Paused |

## Configuration

Settings are automatically saved to:
- **Windows**: `%APPDATA%\StopNailBiting\config.json`

```json
{
  "flash_enabled": true,
  "sound_enabled": true,
  "start_with_windows": false,
  "volume": 0.75
}
```

## Technical Details

### Dependencies

- [MediaPipe](https://mediapipe.dev/) - Hand and face landmark detection
- [OpenCV](https://opencv.org/) - Webcam capture and image processing
- [Shapely](https://shapely.readthedocs.io/) - Geometric polygon operations
- [Pygame](https://www.pygame.org/) - Audio playback
- [pystray](https://github.com/moses-palmer/pystray) - System tray integration

### Detection Parameters

These can be adjusted in `main.py`:

```python
SENSITIVITY = 0.01        # Lip polygon buffer size
Z_DEPTH_THRESHOLD = 0.1   # Max z-difference for valid detection
FRAMES_REQUIRED = 3       # Consecutive frames before alert
TARGET_FPS = 15           # Frame rate cap
COOLDOWN_PERIOD = 1.5     # Seconds alert stays after detection stops
```

Feel free to experiment with the values and see if they yield better results for you, the default settings should work well enough. 

## Limitations

- Windows only
- False positives are possible if hands are near mouth
- First launch may be slow (~5-10s) as files are extracted

## Privacy

- **All processing is local** - No data leaves your computer
- **No recording** - Webcam frames are processed in memory and immediately discarded
- **No network access** - The app works entirely offline

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
