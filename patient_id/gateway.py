"""Client for the ACU API Gateway (routes verified against robot_specs/api-guide.pdf).

Safety rules from the guide are built in: dry run when no URL is configured, never spoofs telemetry, only
touches the one device id you configure, and never retries writes automatically.
"""
import requests

TIMEOUT = 3  # seconds; the workflow needs feedback within 3 s


class GatewayClient:
    def __init__(self, base_url: str = "", device_id: str = "", dry_run: bool | None = None):
        self.base = base_url.rstrip("/")
        self.device_id = device_id
        self.dry_run = (not self.base) if dry_run is None else dry_run
        self.sent: list[tuple] = []  # what was (or would be) sent; handy for tests

    def _call(self, method: str, path: str, body=None):
        self.sent.append((method, path, body))
        if self.dry_run:
            print(f"[gateway dry-run] {method} {path} {body if body is not None else ''}")
            return None
        r = requests.request(method, self.base + path, json=body, timeout=TIMEOUT)
        if not r.ok:
            raise RuntimeError(f"{method} {path}: HTTP {r.status_code} {r.text[:200]}")
        return r.json() if r.text else None

    # -- reads -------------------------------------------------------------------
    def health(self):
        return self._call("GET", "/api/health")

    def devices(self):
        return self._call("GET", "/api/acu/devices")

    # -- notify the robot / ACU -----------------------------------------------------
    def send_instruction(self, command: str):
        """ACU -> robot: stores the latest command string (no catalogue defined by the gateway)."""
        return self._call("POST", f"/api/acu/instructions/{self.device_id}", {"command": command})

    def publish_confirmation(self, workflow_id: str, title: str, tts: str | None = None):
        """Create and publish a one-step Yes/No Confirmation screen on the robot (accept / reject the identity)."""
        wf = {
            "WorkflowID": workflow_id, "WorkflowName": title, "Description": "Patient identity confirmation",
            "RobotId": self.device_id,
            "Steps": [{"StepId": "Step-1", "ScreenId": "Confirmation", "Title": title, "TTS": tts or title,
                       "Buttons": {"Yes": {"Label": "Yes, that's me", "Action": "Next"},
                                   "No": {"Label": "No", "Action": "Previous"}}}],
        }
        self._call("POST", "/api/workflows/robot", wf)
        return self._call("POST", f"/api/acu/robot-workflows/{workflow_id}/assignments/{self.device_id}")
