"""End-to-end self test of the patient identification service (no robot, no gateway, no camera needed).

    python selftest.py

Uses the photos in reference_images/ and a temporary data folder, so it never touches real registrations.
"""
import base64
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from patient_id import config
from patient_id.gateway import GatewayClient
from patient_id.service import create_app

HERE = Path(__file__).resolve().parent
REFS = {"abdullah": HERE / "reference_images" / "sitting_base1_abdullah.jpg",
        "taimoor": HERE / "reference_images" / "sitting_base2_taimoor.jpg"}
results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def jpeg(img):
    return cv2.imencode(".jpg", img)[1].tobytes()


def post_image(client, url, img):
    return client.post(url, content=jpeg(img), headers={"content-type": "image/jpeg"})


def main():
    s = config.Settings(data_dir=tempfile.mkdtemp(), lost_after_s=0.4)
    app = create_app(s)
    c = TestClient(app)
    imgs = {k: cv2.imread(str(v)) for k, v in REFS.items()}

    # --- enrolment
    for pid, path in REFS.items():
        r = c.post("/patients", data={"patient_id": pid, "name": pid.title()},
                   files=[("files", (path.name, path.read_bytes(), "image/jpeg"))])
        check(f"enrol {pid}", r.status_code == 200, r.text[:80])
    check("enrol rejects image without a face",
          c.post("/patients", data={"patient_id": "x"}, files=[("files", ("b.jpg", jpeg(np.zeros((300, 300, 3), np.uint8)), "image/jpeg"))]).status_code == 422)

    # --- identify (also with a darkened frame: robot cameras can be dim)
    for pid in REFS:
        r = post_image(c, "/identify", imgs[pid]).json()
        check(f"identify {pid}", r["faces"] and r["faces"][0]["patient_id"] == pid, str(r["faces"][0]["score"]))
    dark = (imgs["abdullah"] * 0.15).astype(np.uint8)
    r = post_image(c, "/identify", dark).json()
    check("identify in a dark frame", bool(r["faces"]) and r["faces"][0]["patient_id"] == "abdullah",
          str([f["score"] for f in r["faces"]]))

    # --- verification gate: happy path
    c.post("/verify/start", json={"session_id": "s1", "expected_patient_id": "abdullah"})
    t0 = time.perf_counter()
    r = post_image(c, "/verify/s1/face", imgs["abdullah"]).json()
    latency = (time.perf_counter() - t0) * 1000
    check("face verify -> awaiting confirmation", r["state"] == "awaiting_confirmation" and not r["released"], r["prompt"])
    check("workflow NOT released before confirmation", r["released"] is False)
    r = c.post("/verify/s1/confirm", json={"accept": True}).json()
    check("confirm -> released", r["released"] and r["state"] == "confirmed")
    check(f"latency within 3 s ({latency:.0f} ms)", latency < 3000)

    # --- wrong patient x3 -> blocked, stays blocked, operator reset
    c.post("/verify/start", json={"session_id": "s2", "expected_patient_id": "taimoor"})
    for i in range(3):
        r = post_image(c, "/verify/s2/face", imgs["abdullah"]).json()
    check("3 wrong attempts -> blocked", r["state"] == "blocked" and r["attempts"] == 3 and not r["released"], r["reason"])
    check("blocked gate rejects further attempts", post_image(c, "/verify/s2/face", imgs["taimoor"]).status_code == 409)
    r = c.post("/verify/s2/reset").json()
    check("operator reset re-opens gate", r["state"] == "ready" and r["attempts"] == 0)

    # --- bad capture does not burn an attempt
    r = post_image(c, "/verify/s2/face", np.zeros((300, 300, 3), np.uint8)).json()
    check("no face in frame -> attempt not consumed", r["captured"] is False and r["attempts"] == 0)

    # --- rejection by the user counts as a failed attempt
    c.post("/verify/start", json={"session_id": "s3", "expected_patient_id": "abdullah"})
    post_image(c, "/verify/s3/face", imgs["abdullah"])
    r = c.post("/verify/s3/confirm", json={"accept": False}).json()
    check("user rejection -> re-identify", r["state"] == "ready" and r["attempts"] == 1 and not r["released"])

    # --- QR identification
    app.state.store.patients["abdullah"].qr_code = "PATIENT-ABDULLAH-001"
    qr = cv2.QRCodeEncoder.create().encode("PATIENT-ABDULLAH-001")
    qr = cv2.resize(qr, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    qr = cv2.copyMakeBorder(qr, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    c.post("/verify/start", json={"session_id": "s4", "expected_patient_id": "abdullah"})
    r = post_image(c, "/verify/s4/qr", cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR) if qr.ndim == 2 else qr).json()
    check("QR code identifies patient", r.get("state") == "awaiting_confirmation", str(r.get("reason")))
    c.post("/verify/start", json={"session_id": "s5", "expected_patient_id": "abdullah"})
    r = c.post("/verify/s5/qr", json={"code": "WRONG"}).json()
    check("unknown QR code fails attempt", r["attempts"] == 1 and r["state"] == "ready")

    # --- continuous tracking: present -> lost -> reacquired
    c.post("/track/watch", json={"patient_ids": ["abdullah"]})
    r = post_image(c, "/track/frame", imgs["abdullah"]).json()
    check("tracker: patient present", r["patients"][0]["state"] == "present",
          f"bearing={r['patients'][0]['bearing']} proximity={r['patients'][0]['proximity']}")
    time.sleep(0.6)
    r = post_image(c, "/track/frame", np.zeros((480, 640, 3), np.uint8)).json()
    check("tracker: patient lost after timeout", r["patients"][0]["state"] == "lost")
    r = post_image(c, "/track/frame", imgs["abdullah"]).json()
    check("tracker: patient re-acquired", r["patients"][0]["state"] == "present")
    ev = [e["event"] for e in c.get("/track/events").json()["events"]]
    check("tracker events", ev == ["patient_detected", "patient_lost", "patient_reacquired"], str(ev))
    c.post("/track/watch", json={"patient_ids": ["abdullah", "taimoor"]})
    check("tracker keeps at most 2 patients", len(app.state.tracker.watched) <= 2)

    # --- bounded storage: latest 5 patients
    for i in range(7):
        app.state.store.enrol(f"dummy{i}", f"Dummy {i}", np.random.randn(1, 128).astype(np.float32))
    check("store keeps only latest 5 patients", len(app.state.store.patients) == 5, str(list(app.state.store.patients)))

    # --- gateway client payloads (dry run)
    gw = GatewayClient("", "robot-001")
    gw.publish_confirmation("team-blue-confirm", "Are you Abdullah?")
    gw.send_instruction("patient_lost")
    wf = gw.sent[0][2]
    check("gateway workflow matches api-guide format",
          wf["WorkflowID"] and wf["RobotId"] == "robot-001" and wf["Steps"][0]["ScreenId"] == "Confirmation"
          and set(wf["Steps"][0]["Buttons"]) == {"Yes", "No"})
    check("gateway publish route", gw.sent[1][:2] == ("POST", "/api/acu/robot-workflows/team-blue-confirm/assignments/robot-001"))

    # --- memory
    h = c.get("/health").json()
    check(f"memory under 2 GB ({h['memory_mb']} MB)", h["memory_mb"] < 2048)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
