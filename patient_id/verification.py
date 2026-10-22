"""Verification gate: the scan workflow stays blocked until the registered patient is identified AND confirmed.

Slide 7: Capture -> Identify -> Confirm (user accepts/rejects) -> Proceed.
Failure: request re-identification, max 3 attempts, keep the workflow blocked.
"""
import threading
import time
from collections import OrderedDict

READY, AWAITING, CONFIRMED, BLOCKED = "ready", "awaiting_confirmation", "confirmed", "blocked"


class Gate:
    def __init__(self, session_id: str, expected_patient_id: str | None, max_attempts: int):
        self.session_id = session_id
        self.expected = expected_patient_id
        self.max_attempts = max_attempts
        self.created = time.time()
        self.reset()

    def reset(self):
        """Operator override: clears attempts and re-opens identification."""
        self.state = READY
        self.attempts = 0
        self.candidate = None   # {"patient_id", "name", "score", "method"}
        self.reason = "waiting for identification"

    @property
    def released(self) -> bool:
        return self.state == CONFIRMED

    def _fail(self, reason: str):
        self.candidate = None
        self.reason = reason
        self.state = BLOCKED if self.attempts >= self.max_attempts else READY
        if self.state == BLOCKED:
            self.reason += f" - {self.max_attempts} attempts used, workflow stays blocked until an operator resets it"

    def submit(self, patient_id: str | None, name: str | None, score: float, method: str, note: str = ""):
        """Record one identification attempt (face or QR)."""
        if self.state != READY:
            raise GateError(f"gate is {self.state}; identification not accepted")
        self.attempts += 1
        if patient_id is None:
            return self._fail(note or "no registered patient recognised")
        if self.expected is not None and patient_id != self.expected:
            return self._fail("wrong patient: person in front of the robot is not the registered patient")
        self.state = AWAITING
        self.candidate = {"patient_id": patient_id, "name": name, "score": round(score, 3), "method": method}
        self.reason = "identified, waiting for user confirmation"

    def confirm(self, accept: bool):
        if self.state != AWAITING:
            raise GateError(f"gate is {self.state}; nothing to confirm")
        if accept:
            self.state = CONFIRMED
            self.reason = "confirmed, workflow released"
        else:
            self._fail("identification rejected by user")

    def view(self) -> dict:
        who = self.candidate["name"] if self.candidate else None
        return {
            "session_id": self.session_id, "state": self.state, "released": self.released,
            "expected_patient_id": self.expected, "attempts": self.attempts,
            "attempts_left": max(self.max_attempts - self.attempts, 0), "candidate": self.candidate,
            "reason": self.reason,
            # text the robot screen / TTS can show directly (see INTEGRATION.md)
            "prompt": (f"Are you {who}?" if self.state == AWAITING else
                       "Please look at the camera" if self.state == READY else
                       "Thank you, you are confirmed" if self.state == CONFIRMED else
                       "I could not identify you. Please wait for staff"),
        }


class GateError(Exception):
    pass


class GateRegistry:
    """Keeps only the most recent sessions so memory stays bounded."""

    def __init__(self, max_attempts: int, max_sessions: int = 20):
        self.max_attempts, self.max_sessions = max_attempts, max_sessions
        self.sessions: OrderedDict[str, Gate] = OrderedDict()
        self.lock = threading.Lock()

    def start(self, session_id: str, expected_patient_id: str | None) -> Gate:
        with self.lock:
            gate = Gate(session_id, expected_patient_id, self.max_attempts)
            self.sessions[session_id] = gate
            self.sessions.move_to_end(session_id)
            while len(self.sessions) > self.max_sessions:
                self.sessions.popitem(last=False)
            return gate

    def get(self, session_id: str) -> Gate | None:
        return self.sessions.get(session_id)
