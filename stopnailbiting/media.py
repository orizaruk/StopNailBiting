"""Pause/resume system media during alerts (Windows SMTC + pycaw fallback)."""

import os
import queue
import sys
import threading

# Optional WinRT import for media control (graceful degradation if unavailable)
GlobalSystemMediaTransportControlsSessionManager = None
GlobalSystemMediaTransportControlsSessionPlaybackStatus = None
try:
    from winrt.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager,
        GlobalSystemMediaTransportControlsSessionPlaybackStatus,
    )

    WINRT_AVAILABLE = True
except ImportError:
    WINRT_AVAILABLE = False

# Optional CoreAudio (pycaw) import for browser audio muting fallback
comtypes = None
AudioUtilities = None
try:
    if sys.platform == "win32":
        _comtypes_base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        _comtypes_base = os.environ.get(
            "XDG_CONFIG_HOME", os.path.expanduser("~/.config")
        )
    os.environ.setdefault(
        "COMTYPES_GEN_DIR",
        os.path.join(_comtypes_base, "StopNailBiting", "comtypes_gen"),
    )

    import comtypes
    from pycaw.pycaw import AudioUtilities

    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False


class MediaController:
    """Pauses and resumes system media during alerts using Windows SMTC.

    Uses the Windows System Media Transport Controls API to pause any currently
    playing media (Spotify, YouTube, etc.) when an alert triggers, and resume
    it when the alert ends. Gracefully degrades if WinRT is unavailable.

    Also includes an optional CoreAudio fallback (pycaw) to temporarily mute
    browser audio during alerts. This is necessary because Chromium browsers
    typically expose only a single SMTC session even when multiple tabs play
    audio.

    Note: WinRT async operations and CoreAudio COM calls are executed on a
    dedicated worker thread to avoid blocking the main detection loop.
    """

    def __init__(self):
        """Initialize the media controller."""
        self.paused_session_ids = set()
        self.enabled = WINRT_AVAILABLE
        self._muted_browser_session_states = {}
        self._muted_browser_pids = set()
        self._browser_process_names = {
            "brave.exe",
            "chrome.exe",
            "msedge.exe",
            "firefox.exe",
            "opera.exe",
            "opera_gx.exe",
            "vivaldi.exe",
        }

        self._queue = queue.Queue()
        self._worker_thread = None
        if self.enabled or PYCAW_AVAILABLE:
            self._worker_thread = threading.Thread(
                target=self._worker_loop, daemon=True
            )
            self._worker_thread.start()

        if self.enabled and PYCAW_AVAILABLE:
            print("[MediaController] SMTC + CoreAudio fallback enabled")
        elif self.enabled:
            print("[MediaController] SMTC enabled (CoreAudio fallback unavailable)")
        elif PYCAW_AVAILABLE:
            print("[MediaController] CoreAudio fallback enabled (SMTC unavailable)")
        else:
            print("[MediaController] Media control unavailable (WinRT + pycaw missing)")

    def _worker_loop(self):
        """Run queued media operations on a dedicated worker thread."""
        if PYCAW_AVAILABLE and comtypes is not None:
            try:
                # WinRT blocking waits (op.get()) require an MTA thread.
                # CoreAudio (pycaw) is also fine in MTA.
                comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
                print("[MediaController] Worker initialized (MTA)")
            except Exception:
                # Fall back to STA if MTA init fails; CoreAudio may still work,
                # but WinRT SMTC blocking waits may fail in this mode.
                try:
                    comtypes.CoInitialize()
                    print("[MediaController] Worker initialized (STA fallback)")
                except Exception:
                    pass

        try:
            while True:
                fn = self._queue.get()
                if fn is None:
                    break
                try:
                    fn()
                except Exception as e:
                    print(f"[MediaController] Worker error: {e}")
                finally:
                    self._queue.task_done()
        finally:
            if PYCAW_AVAILABLE and comtypes is not None:
                try:
                    comtypes.CoUninitialize()
                except Exception:
                    pass

    def _submit(self, fn, wait=False, timeout=2.0):
        """Submit a media operation to the worker thread."""
        if self._worker_thread is None:
            return

        done = threading.Event()

        def _wrapped():
            try:
                fn()
            finally:
                done.set()

        self._queue.put(_wrapped)
        if wait:
            done.wait(timeout=timeout)

    def _get_manager(self):
        """Get the SMTC session manager (must be called from worker thread)."""
        return GlobalSystemMediaTransportControlsSessionManager.request_async().get()

    def _audio_session_key(self, audio_session):
        """Generate a stable-ish key for an audio session.

        Uses (pid, session instance id) when available, and falls back to
        (pid, display name) when necessary.
        """
        pid = None
        try:
            if getattr(audio_session, "Process", None) is not None:
                pid = audio_session.Process.pid
        except Exception:
            pid = None

        if pid is None:
            try:
                pid = audio_session._ctl.GetProcessId()
            except Exception:
                pid = -1

        instance_id = None
        try:
            instance_id = audio_session._ctl.GetSessionInstanceIdentifier()
        except Exception:
            instance_id = None

        if instance_id:
            return (pid, instance_id)

        session_id = None
        try:
            session_id = audio_session._ctl.GetSessionIdentifier()
        except Exception:
            session_id = None

        if session_id:
            return (pid, session_id)

        display_name = ""
        try:
            display_name = getattr(audio_session, "DisplayName", "") or ""
        except Exception:
            display_name = ""

        return (pid, display_name)

    def _mute_browser_audio_sessions(self):
        """Mute browser CoreAudio sessions and record previous mute state."""
        if not PYCAW_AVAILABLE:
            return

        try:
            sessions = AudioUtilities.GetAllSessions()
        except Exception as e:
            print(f"[MediaController] Error enumerating audio sessions: {e}")
            return

        muted_count = 0
        for s in sessions:
            try:
                proc = getattr(s, "Process", None)
                if proc is None:
                    continue
                name = (proc.name() or "").lower()
                if name not in self._browser_process_names:
                    continue

                vol = s.SimpleAudioVolume
                prev_muted = bool(vol.GetMute())
                if prev_muted:
                    continue

                key = self._audio_session_key(s)
                if key in self._muted_browser_session_states:
                    continue

                vol.SetMute(True, None)
                self._muted_browser_session_states[key] = prev_muted
                self._muted_browser_pids.add(proc.pid)
                muted_count += 1
            except Exception:
                continue

        if muted_count:
            print(f"[MediaController] Muted {muted_count} browser audio session(s)")

    def _restore_browser_audio_sessions(self):
        """Restore browser CoreAudio sessions muted by this controller."""
        if not PYCAW_AVAILABLE or not self._muted_browser_session_states:
            return

        states = self._muted_browser_session_states.copy()
        muted_pids = self._muted_browser_pids.copy()

        try:
            sessions = AudioUtilities.GetAllSessions()
        except Exception as e:
            print(f"[MediaController] Error enumerating audio sessions: {e}")
            return

        restored = 0
        remaining = set(states.keys())
        for s in sessions:
            try:
                key = self._audio_session_key(s)
                if key not in states:
                    continue
                s.SimpleAudioVolume.SetMute(states[key], None)
                restored += 1
                remaining.discard(key)
            except Exception:
                continue

        # If no sessions matched (keys can be unstable for some drivers/apps),
        # fall back to unmuting browser sessions for pids we muted.
        if restored == 0 and muted_pids:
            for s in sessions:
                try:
                    proc = getattr(s, "Process", None)
                    if proc is None or proc.pid not in muted_pids:
                        continue
                    name = (proc.name() or "").lower()
                    if name not in self._browser_process_names:
                        continue
                    vol = s.SimpleAudioVolume
                    if bool(vol.GetMute()):
                        vol.SetMute(False, None)
                except Exception:
                    continue

        # Clear state after restore attempt to avoid unmuting later unrelated sessions.
        self._muted_browser_session_states.clear()
        self._muted_browser_pids.clear()

        if restored:
            print(f"[MediaController] Restored {restored} browser audio session(s)")

    def pause_all(self):
        """Pause all currently playing media sessions and store their IDs."""
        if not self.enabled and not PYCAW_AVAILABLE:
            return

        def _do_pause():
            if self.enabled:
                self.paused_session_ids.clear()
                try:
                    manager = self._get_manager()
                    for session in manager.get_sessions():
                        info = session.get_playback_info()
                        if (
                            info.playback_status
                            == GlobalSystemMediaTransportControlsSessionPlaybackStatus.PLAYING
                        ):
                            app_id = session.source_app_user_model_id
                            if session.try_pause_async().get():
                                self.paused_session_ids.add(app_id)
                                print(f"[MediaController] Paused: {app_id}")
                except Exception as e:
                    print(f"[MediaController] Error pausing media: {e}")

            # Browser CoreAudio fallback: silences additional tabs that SMTC can't pause.
            self._mute_browser_audio_sessions()

        self._submit(_do_pause)

    def resume_all(self, wait=False):
        """Resume only the sessions that were paused by this controller."""
        if not self.enabled and not PYCAW_AVAILABLE:
            return

        if not self.paused_session_ids and not self._muted_browser_session_states:
            return

        def _do_resume():
            if self.enabled and self.paused_session_ids:
                session_ids_to_resume = self.paused_session_ids.copy()
                self.paused_session_ids.clear()
                try:
                    manager = self._get_manager()
                    for session in manager.get_sessions():
                        if session.source_app_user_model_id in session_ids_to_resume:
                            info = session.get_playback_info()
                            if (
                                info.playback_status
                                == GlobalSystemMediaTransportControlsSessionPlaybackStatus.PAUSED
                            ):
                                session.try_play_async().get()
                                print(
                                    f"[MediaController] Resumed: {session.source_app_user_model_id}"
                                )
                except Exception as e:
                    print(f"[MediaController] Error resuming media: {e}")

            self._restore_browser_audio_sessions()

        self._submit(_do_resume, wait=wait)

