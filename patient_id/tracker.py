"""Continuous patient presence tracking (slide 5): is the right patient still in front of / following the robot?

Feed it camera frames; it reports, per watched patient, present / lost plus where they are in the frame
(bearing and proximity) so the robot can correct its guidance. Emits events when a patient is lost / re-found.
"""
import threading
import time
from collections import OrderedDict, deque

PRESENT, LOST, NOT_SEEN_YET = "present", "lost", "not_seen_yet"


class ContinuousTracker:
    def __init__(self, engine, store, threshold, margin=0.0, lost_after_s=2.0, max_tracked=2, clock=time.monotonic):
        self.engine, self.store = engine, store
        self.threshold, self.margin = threshold, margin
        self.lost_after_s, self.max_tracked, self.clock = lost_after_s, max_tracked, clock
        self.watched: OrderedDict[str, dict] = OrderedDict()
        self.events = deque(maxlen=200)
        self._seq = 0
        self.lock = threading.Lock()
        self.last_frame_at = None
        self.last_frame_faces = 0
        self.listeners = []  # callables(event_dict)

    # -- configuration ---------------------------------------------------------
    def watch(self, patient_ids: list[str]) -> list[str]:
        """Watch these patients; only the latest `max_tracked` are kept. Returns the ids now watched."""
        with self.lock:
            for pid in patient_ids:
                if pid in self.watched:
                    self.watched.move_to_end(pid)
                else:
                    self.watched[pid] = {"state": NOT_SEEN_YET, "last_seen": None, "first_seen": None,
                                         "box": None, "score": None, "bearing": None, "proximity": None}
            while len(self.watched) > self.max_tracked:
                self.watched.popitem(last=False)
            return list(self.watched)

    def clear(self):
        with self.lock:
            self.watched.clear()

    # -- per frame -------------------------------------------------------------
    def update(self, frame) -> dict:
        h, w = frame.shape[:2]
        faces = self.engine.analyze(frame)
        now = self.clock()
        seen: dict[str, tuple] = {}
        unknown = 0
        for f in faces:
            pid, score, _ = self.store.match(f.embedding, self.threshold, self.margin)
            if pid is None:
                unknown += 1
            elif pid not in seen or score > seen[pid][0]:
                seen[pid] = (score, f.box)
        with self.lock:
            self.last_frame_at, self.last_frame_faces = now, len(faces)
            for pid, st in self.watched.items():
                if pid in seen:
                    score, (x, y, bw, bh) = seen[pid]
                    was = st["state"]
                    st.update(last_seen=now, box=[round(v) for v in (x, y, bw, bh)], score=round(score, 3),
                              bearing=round(((x + bw / 2) - w / 2) / (w / 2), 3),   # -1 left ... +1 right
                              proximity=round(bw / w, 3))                           # bigger = closer
                    if st["first_seen"] is None:
                        st["first_seen"] = now
                    st["state"] = PRESENT
                    if was == LOST:
                        self._emit("patient_reacquired", pid)
                    elif was == NOT_SEEN_YET:
                        self._emit("patient_detected", pid)
                elif st["state"] == PRESENT and now - st["last_seen"] > self.lost_after_s:
                    st["state"] = LOST
                    self._emit("patient_lost", pid)
            return self._status(now, unknown)

    def status(self) -> dict:
        """Current status without a new frame. Also flags 'lost' if frames stopped arriving."""
        now = self.clock()
        with self.lock:
            for pid, st in self.watched.items():
                if st["state"] == PRESENT and now - st["last_seen"] > self.lost_after_s:
                    st["state"] = LOST
                    self._emit("patient_lost", pid)
            return self._status(now, None)

    def _status(self, now, unknown):
        out = []
        for pid, st in self.watched.items():
            p = self.store.get(pid)
            out.append({"patient_id": pid, "name": p.name if p else None, "state": st["state"],
                        "seconds_since_seen": None if st["last_seen"] is None else round(now - st["last_seen"], 2),
                        "box": st["box"], "score": st["score"], "bearing": st["bearing"],
                        "proximity": st["proximity"]})
        return {"patients": out, "unknown_faces": unknown, "all_present": bool(out) and all(
            p["state"] == PRESENT for p in out)}

    def _emit(self, kind: str, patient_id: str):
        self._seq += 1
        ev = {"seq": self._seq, "event": kind, "patient_id": patient_id, "time": time.time()}
        self.events.append(ev)
        for fn in self.listeners:
            try:
                fn(ev)
            except Exception as exc:  # a broken listener must never stop tracking
                print(f"[tracker] listener error: {exc}")

    def events_since(self, seq: int = 0) -> list[dict]:
        return [e for e in self.events if e["seq"] > seq]
