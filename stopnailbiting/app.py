"""Application entry point: build the components and run the detection loop.

This wires the singletons together (dependency injection) and hands control to
the DetectionEngine, which owns the main loop. Importing this module has no
side effects; call main() to run the app.
"""

from .alert import RedFlashAlert
from .audio import SoundManager
from .camera import CameraManager
from .config import ConfigManager
from .controller import AppController
from .detection import DetectionEngine
from .media import MediaController
from .platform_windows import set_windows_dpi_awareness
from .resources import find_sound_file


def main():
    """Build the app's components and run the detection loop until exit."""
    # Initialize config first (needed by other components)
    config = ConfigManager()
    camera_manager = CameraManager()
    set_windows_dpi_awareness()

    # Initialize sound, alert, and media managers
    sound_manager = SoundManager(
        find_sound_file("noise"), volume=config.get("volume")
    )
    red_flash = RedFlashAlert()
    media_controller = MediaController()
    app_controller = AppController(config, sound_manager, camera_manager)

    engine = DetectionEngine(
        config,
        sound_manager,
        red_flash,
        media_controller,
        camera_manager,
        app_controller,
    )
    engine.run()
