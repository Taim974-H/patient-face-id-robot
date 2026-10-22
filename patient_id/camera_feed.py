"""A local camera acting as the BOT camera: feeds frames to the continuous tracker at a modest frame rate."""
import threading
import time

import cv2

# Camera 1 (USB) is left in manual exposure by some tools, which gives a black picture; auto exposure fixes it.
CAMERA_PROFILES = {1: {cv2.CAP_PROP_AUTO_EXPOSURE: 0.75, cv2.CAP_PROP_BRIGHTNESS: 165}}


class CameraFeed(threading.Thread):
    def __init__(self, index: int, tracker, fps: float = 5.0):
        super().__init__(daemon=True)
        self.index, self.tracker, self.period = index, tracker, 1.0 / fps
        self.frame = None
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()

    def latest_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stop_flag.set()

    def run(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[camera] cannot open camera {self.index}")
            return
        for prop, value in CAMERA_PROFILES.get(self.index, {}).items():
            cap.set(prop, value)
        next_run = 0.0
        while not self.stop_flag.is_set():
            ok, frame = cap.read()  # keep draining so frames are never stale
            if not ok:
                time.sleep(0.2)
                continue
            with self.lock:
                self.frame = frame
            now = time.monotonic()
            if now >= next_run:
                next_run = now + self.period
                try:
                    self.tracker.update(frame)
                except Exception as exc:
                    print(f"[camera] tracker error: {exc}")
        cap.release()
