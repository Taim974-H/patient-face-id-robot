"""Settings. Defaults follow the hackathon edge constraints; override with patient_id_config.json or PID_<NAME> env vars."""
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    # --- paths
    models_dir: str = str(ROOT / "models")
    data_dir: str = str(ROOT / "data")
    # --- recognition
    det_score: float = 0.80          # face detector confidence
    match_threshold: float = 0.363   # SFace cosine threshold (recommended by OpenCV)
    match_margin: float = 0.0        # best must beat runner-up by this much (raise to be stricter)
    max_image_side: int = 1280       # larger frames are downscaled before detection (latency)
    auto_brighten: bool = True       # lift dark frames (hospital corridors, backlit cameras)
    # --- hackathon constraints (problem statement slides 5 and 7)
    max_patients: int = 5            # identification: "latest 5 patients only"
    max_tracked: int = 2             # continuous guidance: "latest 2 patients stored"
    max_attempts: int = 3            # identification: "max. 3 attempts"
    lost_after_s: float = 2.0        # patient counts as lost if unseen this long (feedback must stay <= 3 s)
    # --- ACU API Gateway (empty url = dry run, nothing is sent)
    gateway_url: str = ""            # e.g. http://192.168.1.100:3000   (no trailing /api)
    robot_device_id: str = ""        # the approved device your team may control
    gateway_notify: bool = False     # push patient_lost / released events to the gateway
    publish_confirmation: bool = False  # publish a Yes/No Confirmation screen to the robot
    command_patient_lost: str = "patient_lost"       # agree the real command strings with the organizers
    command_patient_found: str = "patient_found"
    # --- local camera acting as the BOT camera (index) ; None = frames arrive over HTTP only
    camera: int | None = None
    track_fps: float = 5.0
    host: str = "0.0.0.0"
    port: int = 8080


def load(path: Path | None = None) -> Settings:
    s = Settings()
    path = path or ROOT / "patient_id_config.json"
    if path.exists():
        data = json.loads(path.read_text())
        for f in fields(s):
            if f.name in data:
                setattr(s, f.name, data[f.name])
    for f in fields(s):
        raw = os.environ.get("PID_" + f.name.upper())
        if raw is None:
            continue
        cur = getattr(s, f.name)
        if isinstance(cur, bool):
            val = raw.lower() in ("1", "true", "yes")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        elif f.name == "camera":
            val = int(raw) if raw.strip() else None
        else:
            val = raw
        setattr(s, f.name, val)
    return s
