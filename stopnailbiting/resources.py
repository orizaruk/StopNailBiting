"""Resource path resolution for dev and PyInstaller-frozen runs."""

import os
import sys


def resource_path(relative_path):
    """Get absolute path to a bundled resource, for dev and PyInstaller.

    When frozen, resources live under sys._MEIPASS. In development, they live
    in the project root (the parent of this `stopnailbiting/` package directory),
    so we resolve relative to that — not to this module's location.
    """
    if hasattr(sys, "_MEIPASS"):
        # Running as a PyInstaller bundle
        return os.path.join(sys._MEIPASS, relative_path)
    # Running in development: project root is the parent of the package dir
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, relative_path)


def find_sound_file(base_name="noise"):
    """Find a sound file with a supported extension in the assets folder.

    Searches for base_name with .mp3, .wav, .ogg extensions.
    Returns the path if found, None otherwise.
    """
    assets_dir = resource_path("assets")
    supported_extensions = [".mp3", ".wav", ".ogg"]

    for ext in supported_extensions:
        path = os.path.join(assets_dir, base_name + ext)
        if os.path.exists(path):
            return path
    return None
