"""Registered patients (embeddings only, no photos). Bounded: only the latest `capacity` patients are kept."""
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAX_EMBEDDINGS_PER_PATIENT = 10


@dataclass
class Patient:
    patient_id: str
    name: str
    embeddings: np.ndarray            # (n, 128), L2-normalised
    qr_code: str | None = None
    enrolled_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {"patient_id": self.patient_id, "name": self.name, "embeddings": len(self.embeddings),
                "has_qr": self.qr_code is not None, "enrolled_at": self.enrolled_at, "last_used": self.last_used}


class PatientStore:
    def __init__(self, path: Path, capacity: int = 5):
        self.path, self.capacity = Path(path), capacity
        self.lock = threading.RLock()
        self.patients: dict[str, Patient] = {}
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self):
        if not self.path.exists():
            return
        for d in json.loads(self.path.read_text()).get("patients", []):
            self.patients[d["patient_id"]] = Patient(
                d["patient_id"], d["name"], np.array(d["embeddings"], np.float32), d.get("qr_code"),
                d.get("enrolled_at", 0.0), d.get("last_used", 0.0))

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"patients": [{"patient_id": p.patient_id, "name": p.name, "embeddings": p.embeddings.tolist(),
                              "qr_code": p.qr_code, "enrolled_at": p.enrolled_at, "last_used": p.last_used}
                             for p in self.patients.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written file

    # -- registry ------------------------------------------------------------
    def enrol(self, patient_id: str, name: str, embeddings: np.ndarray, qr_code: str | None = None) -> list[str]:
        """Add or extend a patient. Returns ids evicted to stay within capacity (least recently used first)."""
        with self.lock:
            existing = self.patients.get(patient_id)
            if existing is not None:
                embeddings = np.vstack([existing.embeddings, embeddings])[-MAX_EMBEDDINGS_PER_PATIENT:]
                qr_code = qr_code or existing.qr_code
            self.patients[patient_id] = Patient(patient_id, name or patient_id, embeddings.astype(np.float32), qr_code)
            evicted = []
            while len(self.patients) > self.capacity:
                victim = min((p for p in self.patients.values() if p.patient_id != patient_id),
                             key=lambda p: p.last_used)
                del self.patients[victim.patient_id]
                evicted.append(victim.patient_id)
            self._save()
            return evicted

    def get(self, patient_id: str) -> Patient | None:
        return self.patients.get(patient_id)

    def remove(self, patient_id: str) -> bool:
        with self.lock:
            gone = self.patients.pop(patient_id, None) is not None
            if gone:
                self._save()
            return gone

    def touch(self, patient_id: str):
        with self.lock:
            if patient_id in self.patients:
                self.patients[patient_id].last_used = time.time()
                self._save()

    def find_by_qr(self, code: str) -> Patient | None:
        return next((p for p in self.patients.values() if p.qr_code and p.qr_code == code), None)

    # -- matching ------------------------------------------------------------
    def match(self, embedding: np.ndarray, threshold: float, margin: float = 0.0):
        """Returns (patient_id or None, best_score, ranking[(id, score)...])."""
        with self.lock:
            ranking = sorted(((p.patient_id, float((p.embeddings @ embedding).max()))
                              for p in self.patients.values()), key=lambda t: t[1], reverse=True)
        if not ranking:
            return None, -1.0, ranking
        best_id, best = ranking[0]
        runner_up = ranking[1][1] if len(ranking) > 1 else -1.0
        if best < threshold or (best - runner_up) < margin:
            return None, best, ranking
        return best_id, best, ranking
