import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from time import monotonic

import cv2
import face_recognition

from access_control import AccessController
from config import (
    CAMERA_INDEX,
    CSV_PATH,
    DATABASE_PATH,
    FRAME_SCALE,
    RECOGNITION_INTERVAL_SECONDS,
    UNKNOWN_DIR,
)
from database import EventStore
from interface import AccessApp, draw_matches
from recognition import FaceMatch, FaceRecognizer


class AccessMonitor:
    def __init__(self) -> None:
        print("[INFO] Cargando registros de estudiantes y profesores...")
        self.recognizer = FaceRecognizer()
        self.event_store = EventStore(DATABASE_PATH, CSV_PATH)
        self.access_controller = AccessController(self.event_store)
        self._capture: cv2.VideoCapture | None = None
        self._capture_thread: threading.Thread | None = None
        self._recognition_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._recognition_frames: queue.Queue[tuple[object, int]] = queue.Queue(maxsize=1)
        self._display_frames: queue.Queue[tuple[object, int]] = queue.Queue(maxsize=1)
        self._events: queue.Queue[tuple[str, str, str]] = queue.Queue()
        self._detections_lock = threading.Lock()
        self._cached_locations: list[tuple[int, int, int, int]] = []
        self._cached_matches: list[FaceMatch] = []
        self._cached_face_count = 0
        self._cached_scale_x = 1.0
        self._cached_scale_y = 1.0
        self._error_lock = threading.Lock()
        self._error: str | None = None

    def start(self) -> tuple[bool, str]:
        capture = cv2.VideoCapture(CAMERA_INDEX)
        if not capture.isOpened():
            capture.release()
            return False, "No se pudo abrir la cámara. Comprueba la conexión y vuelve a intentarlo."

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._clear_queue(self._recognition_frames)
        self._clear_queue(self._display_frames)
        self._clear_queue(self._events)
        with self._detections_lock:
            self._cached_locations = []
            self._cached_matches = []
            self._cached_face_count = 0
            self._cached_scale_x = 1.0
            self._cached_scale_y = 1.0
        with self._error_lock:
            self._error = None
        self._capture = capture
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._camera_loop, daemon=True
        )
        self._recognition_thread = threading.Thread(
            target=self._recognition_loop, daemon=True
        )
        self._capture_thread.start()
        self._recognition_thread.start()
        return True, "Cámara activa. Monitoreando rostros."

    def stop(self) -> None:
        self._stop_event.set()
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
        for worker in (self._capture_thread, self._recognition_thread):
            if worker is not None and worker.is_alive():
                worker.join(timeout=2)
        self._capture_thread = None
        self._recognition_thread = None

    def read_frame(
        self,
    ) -> tuple[object | None, int, list[tuple[str, str, str]], str | None]:
        latest_frame = None
        while True:
            try:
                latest_frame = self._display_frames.get_nowait()
            except queue.Empty:
                break

        events: list[tuple[str, str, str]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break

        with self._error_lock:
            error = self._error
        if latest_frame is None:
            with self._detections_lock:
                face_count = self._cached_face_count
            return None, face_count, events, error

        frame, _frame_number = latest_frame
        with self._detections_lock:
            locations = list(self._cached_locations)
            matches = list(self._cached_matches)
            face_count = self._cached_face_count
            scale_x = self._cached_scale_x
            scale_y = self._cached_scale_y
        display_locations = [
            (
                int(top * scale_y),
                int(right * scale_x),
                int(bottom * scale_y),
                int(left * scale_x),
            )
            for location in locations
        ]
        frame = draw_matches(frame, display_locations, matches)
        return frame, face_count, events, error

    @staticmethod
    def _clear_queue(target_queue: queue.Queue) -> None:
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                return

    def _publish_camera_frame(self, frame, frame_number: int) -> None:
        self._publish_latest(
            self._recognition_frames, (frame, frame_number)
        )
        self._publish_latest(
            self._display_frames, (frame.copy(), frame_number)
        )

    @staticmethod
    def _publish_latest(target_queue: queue.Queue, item: tuple) -> None:
        if target_queue.full():
            try:
                target_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            pass

    def _capture_unknown_face(
        self,
        frame,
        location: tuple[int, int, int, int],
        timestamp: datetime,
        scale_x: float,
        scale_y: float,
    ) -> Path | None:
        top, right, bottom, left = (
            int(location[0] * scale_y),
            int(location[1] * scale_x),
            int(location[2] * scale_y),
            int(location[3] * scale_x),
        )
        height, width = frame.shape[:2]
        top, bottom = max(0, top), min(height, bottom)
        left, right = max(0, left), min(width, right)
        face_image = frame[top:bottom, left:right]
        if face_image.size == 0:
            return None

        UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
        image_path = UNKNOWN_DIR / f"desconocido_{timestamp:%Y%m%d_%H%M%S_%f}.jpg"
        return image_path if cv2.imwrite(str(image_path), face_image) else None

    def _camera_loop(self) -> None:
        capture = self._capture
        if capture is None:
            return

        frame_number = 0
        try:
            while not self._stop_event.is_set():
                success, frame = capture.read()
                if not success:
                    if not self._stop_event.is_set():
                        self._set_error("La cámara dejó de enviar imágenes.")
                    return
                frame = cv2.flip(frame, 1)
                self._publish_camera_frame(frame, frame_number)
                frame_number += 1
        except Exception as error:
            self._set_error(f"Error en la cámara: {error}")

    def _recognition_loop(self) -> None:
        last_processed_at = 0.0
        try:
            while not self._stop_event.is_set():
                try:
                    frame, _frame_number = self._recognition_frames.get(timeout=0.1)
                except queue.Empty:
                    continue

                current_time = monotonic()
                if current_time - last_processed_at < RECOGNITION_INTERVAL_SECONDS:
                    continue
                last_processed_at = current_time

                small_width = max(1, int(frame.shape[1] * FRAME_SCALE))
                small_height = max(1, int(frame.shape[0] * FRAME_SCALE))
                small_frame = cv2.resize(
                    frame,
                    (small_width, small_height),
                    interpolation=cv2.INTER_AREA,
                )
                scale_x = frame.shape[1] / small_width
                scale_y = frame.shape[0] / small_height
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(
                    rgb_small_frame,
                    number_of_times_to_upsample=0,
                    model="hog",
                )
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, locations
                )
                matches = [
                    self.recognizer.identify(encoding)
                    for encoding in face_encodings
                ]
                with self._detections_lock:
                    self._cached_locations = locations
                    self._cached_matches = matches
                    self._cached_face_count = len(locations)
                    self._cached_scale_x = scale_x
                    self._cached_scale_y = scale_y

                for location, match in zip(locations, matches):
                    timestamp = datetime.now()
                    if match.known:
                        action = self.access_controller.register_known(
                            match.name, match.role, timestamp
                        )
                        if action:
                            print(
                                f"[{action}] {match.role}: {match.name} a las "
                                f"{timestamp:%H:%M:%S} ({timestamp:%d/%m/%Y})"
                            )
                            self._events.put(
                                (
                                    timestamp.strftime("%H:%M:%S"),
                                    action,
                                    f"{match.role} - {match.name}",
                                )
                            )
                    elif self.access_controller.allow_unknown_attempt(timestamp):
                        image_path = self._capture_unknown_face(
                            frame, location, timestamp, scale_x, scale_y
                        )
                        self.access_controller.register_unknown(timestamp, image_path)
                        self._events.put(
                            (
                                timestamp.strftime("%H:%M:%S"),
                                "TENTATIVA",
                                    "Desconocido",
                            )
                        )
        except Exception as error:
            if not self._stop_event.is_set():
                self._set_error(f"Error en el reconocimiento: {error}")

    def _set_error(self, message: str) -> None:
        with self._error_lock:
            self._error = message


def main() -> None:
    monitor = AccessMonitor()
    root = tk.Tk()
    AccessApp(root, monitor.start, monitor.stop, monitor.read_frame)
    root.mainloop()
    monitor.stop()


if __name__ == "__main__":
    main()