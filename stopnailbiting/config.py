"""Persistent application configuration in the user config directory."""

import json
import os
import sys


class ConfigManager:
    """Manages persistent configuration settings"""

    DEFAULT_CONFIG = {
        "flash_enabled": True,
        "sound_enabled": True,
        "start_with_windows": False,
        "volume": 0.75,
        "drinking_detection_enabled": True,
        "pause_media_on_alert": True,
        "camera_name": None,
    }

    def __init__(self):
        """Initialize config manager and load settings from disk.

        Loads existing config from %APPDATA%/StopNailBiting/config.json on Windows,
        or creates a new config file with defaults if none exists.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.config_dir = self._get_config_dir()
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.load()

    def _get_config_dir(self):
        """Get the config directory path (platform-specific)"""
        if sys.platform == "win32":
            base = os.environ.get("APPDATA", os.path.expanduser("~"))
        else:
            base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
        return os.path.join(base, "StopNailBiting")

    def load(self):
        """Load config from file, create default if doesn't exist"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    loaded = json.load(f)
                    # Merge with defaults (in case new settings are added)
                    self.config = {**self.DEFAULT_CONFIG, **loaded}
                print(f"[Config] Loaded from {self.config_file}")
            else:
                # Create default config
                self.save()
                print(f"[Config] Created default config at {self.config_file}")
        except Exception as e:
            print(f"[Config] Error loading config: {e}, using defaults")
            self.config = self.DEFAULT_CONFIG.copy()

    def save(self):
        """Save current config to file"""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"[Config] Error saving config: {e}")

    def get(self, key):
        """Get a config value"""
        return self.config.get(key, self.DEFAULT_CONFIG.get(key))

    def set(self, key, value):
        """Set a config value and save"""
        self.config[key] = value
        self.save()

