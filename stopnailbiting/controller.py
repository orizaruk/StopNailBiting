"""System tray icon, menu, app state, and Windows startup integration."""

import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta

import pystray
from PIL import Image, ImageDraw


class AppController:
    """Controls app state and system tray integration."""

    def __init__(self, config, sound_manager, camera_manager):
        """Initialize app controller with config and integration managers."""
        self.config = config
        self.sound_manager = sound_manager
        self.camera_manager = camera_manager
        self.running = True
        self.paused = False
        self.pause_until = None
        self.pause_timer = None
        self.icon = None
        self.icon_active = self._create_icon_image(active=True)
        self.icon_paused = self._create_icon_image(active=False)
        self.camera_unavailable = False
        self.camera_choices = self.camera_manager.list_camera_choices()
        self.active_camera_selection = self.config.get("camera_name")
        self._pending_camera_switch = None
        self._has_pending_camera_switch = False
        self._camera_switch_lock = threading.Lock()

    def _create_icon_image(self, active=True):
        """Create a 64x64 tray icon image."""
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        fill_color = (220, 53, 69) if active else (128, 128, 128)
        margin = 4
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=fill_color,
            outline=(255, 255, 255),
            width=2,
        )
        center = size // 2
        draw.rectangle(
            [center - 8, center - 12, center + 8, center + 12],
            fill=(255, 255, 255),
        )
        return img

    def _current_status_text(self):
        if self.paused:
            return "Paused"
        if self.camera_unavailable:
            return "No camera"
        return "Monitoring"

    def _update_icon_and_title(self):
        if not self.icon:
            return

        status = self._current_status_text()
        is_paused_visual = self.paused or self.camera_unavailable
        self.icon.icon = self.icon_paused if is_paused_visual else self.icon_active
        camera_label = self.camera_manager.selection_to_label(
            self.active_camera_selection
        )
        self.icon.title = f"Stop Nail Biting - {status} ({camera_label})"

    def _update_menu(self):
        if not self.icon:
            return
        self.icon.menu = self._build_main_menu()
        try:
            self.icon.update_menu()
        except Exception:
            pass

    def set_camera_unavailable(self, unavailable):
        self.camera_unavailable = unavailable
        self._update_icon_and_title()
        self._update_menu()

    def set_active_camera(self, selection):
        self.active_camera_selection = selection
        self._update_icon_and_title()
        self._update_menu()

    def refresh_camera_choices(self, icon=None, item=None):
        self.camera_choices = self.camera_manager.list_camera_choices()
        print(f"[Tray] Cameras refreshed ({len(self.camera_choices)} found)")
        self._update_menu()

    def request_camera_switch(self, selection):
        with self._camera_switch_lock:
            self._pending_camera_switch = selection
            self._has_pending_camera_switch = True
        self.config.set("camera_name", selection)
        print(
            f"[Tray] Camera switch requested: {self.camera_manager.selection_to_log_label(selection)}"
        )

    def consume_camera_switch_request(self):
        with self._camera_switch_lock:
            if not self._has_pending_camera_switch:
                return False, None
            selection = self._pending_camera_switch
            self._pending_camera_switch = None
            self._has_pending_camera_switch = False
        return True, selection

    def toggle_pause(self, icon, item):
        """Toggle pause state and update icon."""
        if self.paused:
            self._cancel_timed_pause()
        self.paused = not self.paused
        status = "Paused" if self.paused else "Monitoring"
        print(f"[Tray] {status}")
        self._update_icon_and_title()

    def pause_for_interval(self, minutes):
        """Pause detection for a specific duration."""
        self._cancel_timed_pause()
        self.paused = True
        self.pause_until = datetime.now() + timedelta(minutes=minutes)
        self.pause_timer = threading.Timer(minutes * 60, self._resume_from_timer)
        self.pause_timer.daemon = True
        self.pause_timer.start()
        print(
            f"[Tray] Paused for {minutes} minutes (until {self.pause_until.strftime('%H:%M')})"
        )
        self._update_icon_and_title()

    def _resume_from_timer(self):
        """Called when timed pause expires to resume detection."""
        self.paused = False
        self.pause_until = None
        self.pause_timer = None
        print("[Tray] Timed pause expired - Monitoring")
        self._update_icon_and_title()

    def _cancel_timed_pause(self):
        """Cancel any active timed pause timer."""
        if self.pause_timer is not None:
            self.pause_timer.cancel()
            self.pause_timer = None
        self.pause_until = None

    def quit_app(self, icon, item):
        """Signal the app to quit."""
        print("[Tray] Quit requested")
        self._cancel_timed_pause()
        self.running = False
        if self.icon:
            self.icon.stop()

    def get_pause_text(self, item):
        """Dynamic menu item text for pause/resume."""
        if not self.paused:
            return "Pause"

        if self.pause_until is not None:
            remaining = self.pause_until - datetime.now()
            if remaining.total_seconds() > 0:
                total_minutes = int(remaining.total_seconds() / 60)
                hours = total_minutes // 60
                minutes = total_minutes % 60
                if hours > 0:
                    return f"Resume ({hours}h {minutes}m remaining)"
                return f"Resume ({minutes}m remaining)"

        return "Resume"

    def is_flash_enabled(self, item):
        return self.config.get("flash_enabled")

    def is_sound_enabled(self, item):
        return self.config.get("sound_enabled")

    def toggle_flash(self, icon, item):
        new_value = not self.config.get("flash_enabled")
        self.config.set("flash_enabled", new_value)
        print(f"[Tray] Flash {'enabled' if new_value else 'disabled'}")

    def toggle_sound(self, icon, item):
        new_value = not self.config.get("sound_enabled")
        self.config.set("sound_enabled", new_value)
        if not new_value:
            self.sound_manager.stop_sound()
        print(f"[Tray] Sound {'enabled' if new_value else 'disabled'}")

    def is_drinking_detection_enabled(self, item):
        return self.config.get("drinking_detection_enabled")

    def toggle_drinking_detection(self, icon, item):
        new_value = not self.config.get("drinking_detection_enabled")
        self.config.set("drinking_detection_enabled", new_value)
        print(f"[Tray] Drinking detection {'enabled' if new_value else 'disabled'}")

    def is_pause_media_enabled(self, item):
        return self.config.get("pause_media_on_alert")

    def toggle_pause_media(self, icon, item):
        new_value = not self.config.get("pause_media_on_alert")
        self.config.set("pause_media_on_alert", new_value)
        print(f"[Tray] Pause media on alert {'enabled' if new_value else 'disabled'}")

    def is_start_with_windows(self, item):
        return self.config.get("start_with_windows")

    def toggle_start_with_windows(self, icon, item):
        new_value = not self.config.get("start_with_windows")
        if new_value:
            success = self._create_startup_shortcut()
        else:
            success = self._remove_startup_shortcut()
        if success:
            self.config.set("start_with_windows", new_value)
            print(f"[Tray] Start with Windows {'enabled' if new_value else 'disabled'}")
        else:
            print("[Tray] Failed to update startup setting")

    def _get_startup_folder(self):
        if sys.platform != "win32":
            return None
        return os.path.join(
            os.environ.get("APPDATA", ""),
            "Microsoft",
            "Windows",
            "Start Menu",
            "Programs",
            "Startup",
        )

    def _get_shortcut_path(self):
        startup_folder = self._get_startup_folder()
        if startup_folder:
            return os.path.join(startup_folder, "StopNailBiting.lnk")
        return None

    def _get_exe_path(self):
        if getattr(sys, "frozen", False):
            return sys.executable
        return f'"{sys.executable}" "{os.path.abspath(__file__)}"'

    def _create_startup_shortcut(self):
        if sys.platform != "win32":
            print("[Startup] Not on Windows, skipping")
            return False

        shortcut_path = self._get_shortcut_path()
        exe_path = self._get_exe_path()
        if not shortcut_path:
            return False

        try:
            ps_script = f'''
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{exe_path}"
$Shortcut.WorkingDirectory = "{os.path.dirname(exe_path)}"
$Shortcut.Description = "Stop Nail Biting Detection"
$Shortcut.Save()
'''
            subprocess.run(
                ["powershell", "-Command", ps_script], capture_output=True, check=True
            )
            print(f"[Startup] Created shortcut at {shortcut_path}")
            return True
        except Exception as e:
            print(f"[Startup] Failed to create shortcut: {e}")
            return False

    def _remove_startup_shortcut(self):
        shortcut_path = self._get_shortcut_path()
        if not shortcut_path:
            return False

        try:
            if os.path.exists(shortcut_path):
                os.remove(shortcut_path)
                print(f"[Startup] Removed shortcut from {shortcut_path}")
            return True
        except Exception as e:
            print(f"[Startup] Failed to remove shortcut: {e}")
            return False

    def _volume_menu_item(self, label, level):
        def is_checked(item):
            return abs(self.config.get("volume") - level) < 0.01

        def set_level(icon, item):
            self.config.set("volume", level)
            self.sound_manager.set_volume(level)

        return pystray.MenuItem(label, set_level, checked=is_checked)

    def _camera_menu_item(self, selection, label):
        def set_camera(icon, item):
            self.request_camera_switch(selection)

        def is_selected(item):
            return self.active_camera_selection == selection

        return pystray.MenuItem(label, set_camera, checked=is_selected)

    def _build_camera_menu(self):
        items = [
            pystray.MenuItem("Refresh Cameras", self.refresh_camera_choices),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Auto / Default",
                lambda icon, item: self.request_camera_switch(None),
                checked=lambda item: self.active_camera_selection is None,
            ),
        ]

        if self.camera_choices:
            for selection, label in self.camera_choices:
                items.append(self._camera_menu_item(selection, label))
        else:
            items.append(
                pystray.MenuItem(
                    "No cameras found",
                    lambda icon, item: None,
                    enabled=False,
                )
            )
        return pystray.Menu(*items)

    def _build_main_menu(self):
        volume_menu = pystray.Menu(
            self._volume_menu_item("100%", 1.0),
            self._volume_menu_item("75%", 0.75),
            self._volume_menu_item("50%", 0.50),
            self._volume_menu_item("25%", 0.25),
        )

        pause_interval_menu = pystray.Menu(
            pystray.MenuItem(
                "30 minutes",
                lambda icon, item: self.pause_for_interval(30),
            ),
            pystray.MenuItem(
                "1 hour",
                lambda icon, item: self.pause_for_interval(60),
            ),
            pystray.MenuItem(
                "2 hours",
                lambda icon, item: self.pause_for_interval(120),
            ),
        )

        alert_settings_menu = pystray.Menu(
            pystray.MenuItem(
                "Enable Flash",
                self.toggle_flash,
                checked=self.is_flash_enabled,
            ),
            pystray.MenuItem(
                "Enable Sound",
                self.toggle_sound,
                checked=self.is_sound_enabled,
            ),
            pystray.MenuItem(
                "Pause Media on Alert",
                self.toggle_pause_media,
                checked=self.is_pause_media_enabled,
            ),
            pystray.MenuItem(
                "Drinking Detection",
                self.toggle_drinking_detection,
                checked=self.is_drinking_detection_enabled,
            ),
        )

        return pystray.Menu(
            pystray.MenuItem("Choose Camera", self._build_camera_menu()),
            pystray.MenuItem("Alert Settings", alert_settings_menu),
            pystray.MenuItem("Volume", volume_menu),
            pystray.MenuItem(
                "Start with Windows",
                self.toggle_start_with_windows,
                checked=self.is_start_with_windows,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Pause for...", pause_interval_menu),
            pystray.MenuItem(
                self.get_pause_text,
                self.toggle_pause,
            ),
            pystray.MenuItem("Quit", self.quit_app),
        )

    def setup_tray(self):
        """Create and configure the system tray icon with menu."""
        self.icon = pystray.Icon(
            "StopNailBiting",
            self.icon_active,
            "Stop Nail Biting - Monitoring",
            self._build_main_menu(),
        )
        self._update_icon_and_title()
        return self.icon

    def run_tray(self):
        """Run the tray icon (call from a separate thread)."""
        icon = self.setup_tray()
        icon.run()
