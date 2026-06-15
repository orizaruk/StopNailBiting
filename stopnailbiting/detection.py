"""Real-time nail-biting detection loop and alert state machine."""

import threading
import time

import cv2
import mediapipe as mp

from .constants import (
    COOLDOWN_PERIOD,
    DRINKING_CLASS_LABELS,
    DRINKING_DETECTION_INTERVAL,
    DRINKING_PERSISTENCE_FRAMES,
    FRAMES_REQUIRED,
    HAND_INDICES,
    LIP_INDICES,
    SENSITIVITY,
    TARGET_FPS,
    Z_DEPTH_THRESHOLD,
    FaceLandmarker,
    HandLandmarker,
    ObjectDetector,
    create_detector_options,
)
from .geometry import polygon_contains_buffered


class DetectionEngine:
    """Owns the webcam loop, MediaPipe inference, and the alert state machine.

    All detection state lives on the instance (previously module-level globals),
    and collaborators are injected so the engine can be constructed and reasoned
    about independently of the rest of the app.
    """

    def __init__(
        self,
        config,
        sound_manager,
        red_flash,
        media_controller,
        camera_manager,
        app_controller,
    ):
        self.config = config
        self.sound_manager = sound_manager
        self.red_flash = red_flash
        self.media_controller = media_controller
        self.camera_manager = camera_manager
        self.app_controller = app_controller

        self.cap = None
        self.alert_active = False
        self.last_biting_time = 0  # When biting was last detected
        self.consecutive_detections = 0  # Consecutive frames with detection

        # Drinking detection state
        self.drinking_frame_counter = 0
        self.drinking_detected = False
        self.drinking_persistence_counter = 0

    def _stop_active_alert(self):
        """Reset active alert state before a camera soft-reset or downtime."""
        if self.alert_active:
            self.sound_manager.stop_sound()
            self.red_flash.hide()
            self.media_controller.resume_all()
            self.alert_active = False

        self.consecutive_detections = 0
        self.last_biting_time = 0
        self.drinking_frame_counter = 0
        self.drinking_detected = False
        self.drinking_persistence_counter = 0

    @staticmethod
    def _warmup_camera(capture, frames=2):
        """Read a couple of frames after opening to avoid transient failures."""
        for _ in range(frames):
            if capture is None:
                return
            capture.read()
            time.sleep(0.02)

    def _open_camera_and_update_state(self, preferred_selection, reason, fallback=True):
        """Open a camera and synchronize tray + config state."""
        opened_cap, active_selection = self.camera_manager.open_camera(
            preferred_selection,
            fallback=fallback,
        )

        if active_selection is not None and active_selection != self.config.get(
            "camera_name"
        ):
            self.config.set("camera_name", active_selection)

        self.app_controller.set_active_camera(active_selection)
        self.app_controller.set_camera_unavailable(opened_cap is None)

        if opened_cap is None:
            print(f"[Camera] No available camera ({reason})")
        else:
            print(
                f"[Camera] Active camera: "
                f"{self.camera_manager.selection_to_log_label(active_selection)} "
                f"[{active_selection}] ({reason})"
            )
            self._warmup_camera(opened_cap)
        return opened_cap

    def cleanup(self):
        """Release all resources on application exit.

        Releases webcam, stops audio, destroys alert windows, resumes any paused
        media, and stops the tray icon. Called from the finally block to ensure
        cleanup on normal exit or exception.
        """
        print("\nCleaning up...")
        if self.cap is not None:
            self.cap.release()
        self.media_controller.resume_all(wait=True)
        self.sound_manager.cleanup()
        self.red_flash.cleanup()
        if self.app_controller.icon:
            self.app_controller.icon.stop()
        print("Shutdown complete.")

    def run(self):
        """Open the camera, start the tray, and run the detection loop to exit."""
        try:
            self.cap = self._open_camera_and_update_state(
                self.config.get("camera_name"),
                reason="startup",
            )

            # Start the system tray icon in a background thread
            tray_thread = threading.Thread(
                target=self.app_controller.run_tray, daemon=True
            )
            tray_thread.start()
            print("[Info] System tray icon started")

            hand_options, face_options, object_options = create_detector_options()
            with (
                HandLandmarker.create_from_options(hand_options) as hand_landmarker,
                FaceLandmarker.create_from_options(face_options) as face_landmarker,
                ObjectDetector.create_from_options(object_options) as object_detector,
            ):
                self._loop(hand_landmarker, face_landmarker, object_detector)
        except KeyboardInterrupt:
            pass  # Normal exit via Ctrl+C
        finally:
            self.cleanup()

    def _loop(self, hand_landmarker, face_landmarker, object_detector):
        """The main per-frame detection loop."""
        biting_detected = False
        was_paused_last_tick = self.app_controller.paused
        next_reacquire_attempt_at = 0.0
        last_reacquire_failure_log_time = 0.0

        while self.app_controller.running:
            # UPDATE THE RED FLASH (must be called even when paused for tkinter)
            self.red_flash.update()

            if self.app_controller.paused:
                self._stop_active_alert()
                if not was_paused_last_tick:
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                        print("[Camera] Released camera due to pause")
                was_paused_last_tick = True
                time.sleep(0.1)
                continue

            if was_paused_last_tick:
                was_paused_last_tick = False
                next_reacquire_attempt_at = 0.0
                last_reacquire_failure_log_time = 0.0
                print("[Camera] Resume requested - attempting to reacquire camera")

            has_switch_request, requested_selection = (
                self.app_controller.consume_camera_switch_request()
            )
            if has_switch_request:
                self._stop_active_alert()
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                self.cap = self._open_camera_and_update_state(
                    requested_selection,
                    reason="tray switch",
                )
                next_reacquire_attempt_at = 0.0
                last_reacquire_failure_log_time = 0.0

            if self.cap is None:
                now = time.time()
                if now >= next_reacquire_attempt_at:
                    preferred_selection = self.app_controller.active_camera_selection
                    if preferred_selection is None:
                        preferred_selection = self.config.get("camera_name")

                    strict_selected_camera = preferred_selection is not None
                    self.cap = self._open_camera_and_update_state(
                        preferred_selection,
                        reason="resume reacquire"
                        if strict_selected_camera
                        else "resume reacquire (auto)",
                        fallback=not strict_selected_camera,
                    )

                    if self.cap is None:
                        if (
                            last_reacquire_failure_log_time == 0.0
                            or now - last_reacquire_failure_log_time >= 5.0
                        ):
                            if strict_selected_camera:
                                print(
                                    "[Camera] Selected camera unavailable - retrying in 1.0s"
                                )
                            else:
                                print("[Camera] No available camera - retrying in 1.0s")
                            last_reacquire_failure_log_time = now
                        next_reacquire_attempt_at = now + 1.0
                    else:
                        print("[Camera] Camera reacquired")
                        next_reacquire_attempt_at = 0.0
                        last_reacquire_failure_log_time = 0.0
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                print("[Camera] Frame read failed - attempting reopen")
                self.cap.release()
                self.cap = self._open_camera_and_update_state(
                    self.app_controller.active_camera_selection,
                    reason="read failure recovery",
                )
                self._stop_active_alert()
                time.sleep(0.2)
                continue

            # Get current timestamp in milliseconds (must be monotonically increasing)
            timestamp_ms = int(time.time() * 1000)

            # Prepare the mediapipe image object
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Reset detection state at the start of each frame
            biting_detected = False

            # Run hand detection
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            # Only check face if hands detected
            if hand_result.hand_landmarks:
                face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

                if face_result.face_landmarks:
                    # Extract lip landmarks
                    face_landmarks_array = face_result.face_landmarks[0]
                    lip_points_coords_list = []
                    for index in LIP_INDICES:
                        landmark_info = face_landmarks_array[index]
                        lip_points_coords_list.append(
                            (landmark_info.x, landmark_info.y)
                        )

                    # Calculate average lip z-depth for comparison
                    lip_avg_z = sum(
                        face_landmarks_array[i].z for i in LIP_INDICES
                    ) / len(LIP_INDICES)

                    # Drinking detection (run every N frames to save resources)
                    if self.config.get("drinking_detection_enabled"):
                        self.drinking_frame_counter += 1

                        if self.drinking_frame_counter >= DRINKING_DETECTION_INTERVAL:
                            self.drinking_frame_counter = 0

                            # Run object detection
                            object_result = object_detector.detect_for_video(
                                mp_image, timestamp_ms
                            )

                            # Check if any drinking object is detected
                            drinking_this_frame = False
                            for detection in object_result.detections:
                                category_name = detection.categories[0].category_name
                                if category_name in DRINKING_CLASS_LABELS:
                                    drinking_this_frame = True
                                    print(
                                        f"[Drinking] {category_name} detected - suppressing alerts"
                                    )
                                    break

                            if drinking_this_frame:
                                self.drinking_detected = True
                                self.drinking_persistence_counter = (
                                    DRINKING_PERSISTENCE_FRAMES
                                )
                            elif self.drinking_persistence_counter > 0:
                                self.drinking_persistence_counter -= 1
                                if self.drinking_persistence_counter == 0:
                                    self.drinking_detected = False
                        else:
                            # Between detection frames, decrement persistence counter
                            if self.drinking_persistence_counter > 0:
                                self.drinking_persistence_counter -= 1
                                if self.drinking_persistence_counter == 0:
                                    self.drinking_detected = False

                    # Skip fingertip check if drinking object detected
                    if self.drinking_detected and self.config.get(
                        "drinking_detection_enabled"
                    ):
                        biting_detected = False
                    else:
                        # Check if hand landmarks intersect with mouth polygon
                        for hand in hand_result.hand_landmarks:
                            for hand_landmark_index in HAND_INDICES:
                                landmark_info = hand[hand_landmark_index]
                                if polygon_contains_buffered(
                                    landmark_info.x,
                                    landmark_info.y,
                                    lip_points_coords_list,
                                    SENSITIVITY,
                                ):
                                    # Z-depth check: finger must be at similar depth to
                                    # lips; filters out fingers passing in front of face.
                                    if abs(landmark_info.z - lip_avg_z) < Z_DEPTH_THRESHOLD:
                                        biting_detected = True
                                        break
                            if biting_detected:
                                break

            current_time = time.time()

            # Temporal smoothing: require multiple consecutive positive frames
            # to reduce false positives from brief hand movements near face
            if biting_detected:
                self.consecutive_detections += 1
            else:
                self.consecutive_detections = 0

            # Alert state machine: activate after FRAMES_REQUIRED consecutive detections,
            # then maintain alert until COOLDOWN_PERIOD after detection stops
            if self.consecutive_detections >= FRAMES_REQUIRED:
                # Sustained detection confirmed - update timestamp and activate alert
                self.last_biting_time = current_time

                if not self.alert_active:
                    # Check config before triggering each alert type
                    if self.config.get("pause_media_on_alert"):
                        self.media_controller.pause_all()
                    if self.config.get("sound_enabled"):
                        self.sound_manager.start_sound()
                    if self.config.get("flash_enabled"):
                        self.red_flash.flash()
                    self.alert_active = True
            else:
                # Not enough consecutive detections - check if cooldown has expired
                time_since_last_bite = current_time - self.last_biting_time
                if self.alert_active and time_since_last_bite > COOLDOWN_PERIOD:
                    self.sound_manager.stop_sound()
                    self.red_flash.hide()
                    if self.config.get("pause_media_on_alert"):
                        self.media_controller.resume_all()
                    self.alert_active = False

            # Limit frame rate to reduce CPU usage
            time.sleep(1 / TARGET_FPS)
