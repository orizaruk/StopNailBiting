"""Alert sound playback via miniaudio, with a winsound beep fallback."""

import array
import os
import sys
import threading

import numpy as np

try:
    import winsound

    WINSOUND_AVAILABLE = True
except ImportError:
    winsound = None
    WINSOUND_AVAILABLE = False

# miniaudio: tiny cross-platform looped WAV/MP3/OGG playback with volume control.
try:
    import miniaudio

    MINIAUDIO_AVAILABLE = True
except ImportError:
    miniaudio = None
    MINIAUDIO_AVAILABLE = False


class SoundManager:
    """Manages sound alerts with anti-flicker protection and graceful degradation.

    Primary backend is miniaudio: it decodes the alert file (wav/mp3/ogg) once and
    streams it on a loop with software volume control. If miniaudio or the sound
    file is unavailable, it falls back to a winsound beep loop on Windows, and
    otherwise runs in visual-only mode.
    """

    def __init__(self, sound_file, volume=0.75):
        """Initialize sound manager with optional audio file.

        Args:
            sound_file: Path to the alert sound file (.mp3, .wav, or .ogg).
                       If None or file doesn't exist, runs in visual-only mode.
            volume: Initial volume level from 0.0 (silent) to 1.0 (full).
        """
        self.enabled = False
        self.sound_playing = False
        self.volume = max(0.0, min(1.0, volume))
        self.backend = "none"

        # miniaudio playback state
        self._device = None
        self._source_samples = None  # np.int16 view of the decoded, pre-volume audio
        self._nchannels = 1
        self._scaled = None  # array.array('h') rendered at current volume
        self._play_pos = 0  # read cursor into self._scaled (loops)
        self._stop_requested = False

        # winsound beep fallback state
        self._beep_stop_event = threading.Event()
        self._beep_thread = None

        if sound_file is not None and os.path.exists(sound_file):
            if MINIAUDIO_AVAILABLE:
                try:
                    decoded = miniaudio.decode_file(
                        sound_file, output_format=miniaudio.SampleFormat.SIGNED16
                    )
                    self._source_samples = np.frombuffer(
                        decoded.samples.tobytes(), dtype=np.int16
                    )
                    self._nchannels = decoded.nchannels
                    self._device = miniaudio.PlaybackDevice(
                        output_format=miniaudio.SampleFormat.SIGNED16,
                        nchannels=decoded.nchannels,
                        sample_rate=decoded.sample_rate,
                    )
                    self._render_scaled()
                    self.enabled = True
                    self.backend = "miniaudio"
                    print(
                        f"[Info] Sound loaded successfully: {sound_file} (volume: {int(self.volume * 100)}%)"
                    )
                    return
                except Exception as e:
                    print(f"[Warning] Failed to initialize miniaudio: {e}")
            else:
                print("[Warning] miniaudio unavailable - falling back to beep")
        elif sound_file is None:
            print("[Warning] No sound file found (expected assets/noise.mp3|.wav|.ogg)")
        else:
            print(f"[Warning] Sound file not found: {sound_file}")

        if WINSOUND_AVAILABLE and sys.platform == "win32":
            self.enabled = True
            self.backend = "winsound"
            print("[Info] Using Windows beep fallback for alert sound")
        else:
            print("[Warning] No audio backend available - running in visual-only mode")

    def _render_scaled(self):
        """Render the source samples at the current volume into self._scaled.

        Pre-scaling once per volume change keeps the realtime audio callback cheap
        (it only slices the buffer). The callback reads self._scaled by reference,
        so swapping in a new same-length buffer takes effect on its next pull.
        """
        if self.volume >= 0.999:
            scaled = self._source_samples
        else:
            scaled = np.clip(
                self._source_samples.astype(np.float32) * self.volume, -32768, 32767
            ).astype(np.int16)
        buf = array.array("h")
        buf.frombytes(scaled.tobytes())
        self._scaled = buf

    def _audio_stream(self):
        """Generator feeding looped, volume-scaled PCM frames to miniaudio."""
        required_frames = yield b""  # priming yield (miniaudio protocol)
        while not self._stop_requested:
            buf = self._scaled
            total = len(buf)
            needed = required_frames * self._nchannels
            out = array.array("h")
            while len(out) < needed:
                end = min(self._play_pos + (needed - len(out)), total)
                out.extend(buf[self._play_pos : end])
                self._play_pos = end
                if self._play_pos >= total:
                    self._play_pos = 0  # seamless loop back to start
            required_frames = yield out

    def _beep_loop(self):
        while not self._beep_stop_event.is_set():
            try:
                winsound.Beep(1400, 180)
            except Exception:
                break
            if self._beep_stop_event.wait(0.12):
                break

    def start_sound(self):
        """Start the alert sound if not already playing"""
        if not self.enabled or self.sound_playing:
            return

        if self.backend == "miniaudio":
            self._stop_requested = False
            self._play_pos = 0
            stream = self._audio_stream()
            next(stream)  # prime the generator to its first yield
            self._device.start(stream)
            self.sound_playing = True
            print("Alert sound started")
            return

        if self.backend == "winsound":
            self._beep_stop_event.clear()
            self._beep_thread = threading.Thread(target=self._beep_loop, daemon=True)
            self._beep_thread.start()
            self.sound_playing = True
            print("Alert sound started")

    def stop_sound(self):
        """Stop the alert sound"""
        if not self.enabled or not self.sound_playing:
            return

        if self.backend == "miniaudio":
            self._stop_requested = True
            self._device.stop()
        elif self.backend == "winsound":
            self._beep_stop_event.set()
            if self._beep_thread is not None:
                self._beep_thread.join(timeout=0.5)
                self._beep_thread = None

        self.sound_playing = False
        print("Alert sound stopped")

    def set_volume(self, volume):
        """Set the alert sound volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        if self.backend == "miniaudio" and self._source_samples is not None:
            self._render_scaled()  # callback picks up the new buffer on its next pull
            print(f"[Sound] Volume set to {int(self.volume * 100)}%")
        elif self.backend == "winsound":
            print("[Sound] Volume changes are not supported for Windows beep fallback")

    def cleanup(self):
        """Release audio resources"""
        self.stop_sound()
        if self.backend == "miniaudio" and self._device is not None:
            self._device.close()

