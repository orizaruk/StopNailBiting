"""
Stop Nail Biting - Real-time nail biting detection and alert system.

This application uses computer vision (MediaPipe) to detect when fingers are near
the mouth, indicating potential nail-biting behavior. When detected, it triggers
visual (red screen flash) and audio alerts to help break the habit.

The app runs silently in the system tray and supports:
- Multi-monitor visual alerts
- Configurable audio alerts with volume control
- Drinking detection to reduce false positives (cups, bottles, glasses)
- Persistent settings saved to user config directory
- Windows startup integration

This file is a thin entry-point shim. The implementation lives in the
`stopnailbiting` package; see `stopnailbiting/app.py` for the wiring and
`stopnailbiting/detection.py` for the detection loop.
"""

from stopnailbiting.app import main

if __name__ == "__main__":
    main()
