"""HTTP service (the ACU "Patient Validator" domain service). Sits behind the ACU API Gateway.

Run:   python -m patient_id.service            (see INTEGRATION.md)
Docs:  http://<host>:8080/docs   (Swagger UI)
"""
import base64
import os
import time

import cv2
import numpy as np
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from . import config
from .engine import FaceEngine, decode_image
from .gateway import GatewayClient
from .store import PatientStore
from .tracker import ContinuousTracker
from .verification import GateError, GateRegistry

LATENCY_BUDGET_MS = 3000  # slide 5: feedback within 3 s


class StartBody(BaseModel):
    session_id: str
    expected_patient_id: str | None = None


class ConfirmBody(BaseModel):
    accept: bool


class WatchBody(BaseModel):
    patient_ids: list[str]


class QrBody(BaseModel):
    code: str | None = None          # a code the robot already decoded
    image_base64: str | None = None  # or a picture containing the QR code


async def read_image(request: Request) -> np.ndarray:
    """Accepts multipart (field 'file'), raw image bytes (Content-Type image/*), or JSON {'image_base64': ...}."""
    ctype = request.headers.get("content-type", "")
    try:
        if ctype.startswith("multipart/"):
            form = await request.form()
            f = form.get("file")
            if f is None:
                raise ValueError("multipart field 'file' is missing")
            data = await f.read()
        elif ctype.startswith("application/json"):
            body = await request.json()
            data = base64.b64decode(body["image_base64"])
        else:
            data = await request.body()
        return decode_image(data)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(400, f"bad image: {exc}")


def create_app(settings: config.Settings | None = None, engine: FaceEngine | None = None) -> FastAPI:
    s = settings or config.load()
    engine = engine or FaceEngine(s.models_dir, s.det_score, s.max_image_side, s.auto_brighten)
    store = PatientStore(os.path.join(s.data_dir, "patients.json"), s.max_patients)
    gates = GateRegistry(s.max_attempts)
    tracker = ContinuousTracker(engine, store, s.match_threshold, s.match_margin, s.lost_after_s, s.max_tracked)
    gateway = GatewayClient(s.gateway_url, s.robot_device_id)

    if s.gateway_notify:
        def notify(ev):
            cmd = {"patient_lost": s.command_patient_lost, "patient_reacquired": s.command_patient_found}.get(ev["event"])
            if cmd:
                gateway.send_instruction(cmd)
        tracker.listeners.append(notify)

    app = FastAPI(title="Patient identification service", version="1.0")
    app.state.s, app.state.engine, app.state.store = s, engine, store
    app.state.gates, app.state.tracker, app.state.gateway, app.state.camera = gates, tracker, gateway, None

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Process-Time-ms"] = f"{ms:.0f}"
        if ms > LATENCY_BUDGET_MS:
            print(f"[warn] {request.method} {request.url.path} took {ms:.0f} ms (> {LATENCY_BUDGET_MS})")
        return response

    # ---------------------------------------------------------------- health
    @app.get("/health")
    def health():
        rss = psutil.Process().memory_info().rss / 2**20
        return {"status": "ok", "memory_mb": round(rss), "patients": len(store.patients),
                "capacity": s.max_patients, "camera": s.camera, "gateway": "dry-run" if gateway.dry_run else s.gateway_url}

    # -------------------------------------------------------------- patients
    @app.get("/patients")
    def list_patients():
        return [p.public() for p in store.patients.values()]

    @app.post("/patients")
    async def enrol(request: Request):
        """multipart: patient_id, name (optional), qr_code (optional), one or more image files in 'files'."""
        form = await request.form()
        pid = (form.get("patient_id") or "").strip()
        if not pid:
            raise HTTPException(400, "patient_id is required")
        files = form.getlist("files")
        if not files:
            raise HTTPException(400, "attach at least one reference image as 'files'")
        images = [decode_image(await f.read()) for f in files]

        def work():
            return np.vstack([engine.enrol_embeddings(img) for img in images])

        try:
            embs = await run_in_threadpool(work)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        evicted = store.enrol(pid, form.get("name") or pid, embs, form.get("qr_code") or None)
        return {"patient_id": pid, "embeddings": len(store.get(pid).embeddings), "evicted": evicted}

    @app.delete("/patients/{patient_id}")
    def delete_patient(patient_id: str):
        if not store.remove(patient_id):
            raise HTTPException(404, "unknown patient")
        return {"deleted": patient_id}

    # -------------------------------------------------------------- identify
    @app.post("/identify")
    async def identify(request: Request):
        """Who is in this picture? Stateless, no gate. Returns every face with its best registered match."""
        img = await read_image(request)
        faces = await run_in_threadpool(engine.analyze, img)
        out = []
        for f in faces:
            pid, score, ranking = store.match(f.embedding, s.match_threshold, s.match_margin)
            p = store.get(pid) if pid else None
            out.append({"box": [round(v) for v in f.box], "detector_score": round(f.score, 3), "patient_id": pid,
                        "name": p.name if p else None, "score": round(score, 3)})
        return {"faces": out}

    # ----------------------------------------------------- verification gate
    def gate_or_404(sid: str):
        g = gates.get(sid)
        if g is None:
            raise HTTPException(404, "unknown session; call POST /verify/start first")
        return g

    @app.post("/verify/start")
    def verify_start(body: StartBody):
        if body.expected_patient_id and store.get(body.expected_patient_id) is None:
            raise HTTPException(404, "expected patient is not registered")
        return gates.start(body.session_id, body.expected_patient_id).view()

    @app.get("/verify/{sid}")
    def verify_state(sid: str):
        return gate_or_404(sid).view()

    @app.post("/verify/{sid}/face")
    async def verify_face(sid: str, request: Request):
        gate = gate_or_404(sid)
        img = await read_image(request)
        faces = await run_in_threadpool(engine.analyze, img)
        if not faces:  # a bad capture is not the patient's fault: no attempt is consumed
            return {**gate.view(), "captured": False, "note": "no face in the image, try again"}
        pid, score, _ = store.match(faces[0].embedding, s.match_threshold, s.match_margin)
        p = store.get(pid) if pid else None
        try:
            gate.submit(pid, p.name if p else None, score, "face")
        except GateError as exc:
            raise HTTPException(409, str(exc))
        if p:
            store.touch(pid)
        if gate.state == "awaiting_confirmation" and s.publish_confirmation:
            _try(lambda: gateway.publish_confirmation(f"confirm-{sid}"[:60], gate.view()["prompt"]))
        return {**gate.view(), "captured": True, "faces_in_frame": len(faces)}

    @app.post("/verify/{sid}/qr")
    async def verify_qr(sid: str, request: Request):
        gate = gate_or_404(sid)
        ctype = request.headers.get("content-type", "")
        code = None
        if ctype.startswith("application/json"):
            body = QrBody(**await request.json())
            code = body.code
            if code is None and body.image_base64:
                code = await run_in_threadpool(_decode_qr, decode_image(base64.b64decode(body.image_base64)))
        else:
            code = await run_in_threadpool(_decode_qr, await read_image(request))
        if not code:
            return {**gate.view(), "captured": False, "note": "no QR code found, try again"}
        p = store.find_by_qr(code)
        try:
            gate.submit(p.patient_id if p else None, p.name if p else None, 1.0, "qr", "QR code not registered")
        except GateError as exc:
            raise HTTPException(409, str(exc))
        if p:
            store.touch(p.patient_id)
        return {**gate.view(), "captured": True}

    @app.post("/verify/{sid}/confirm")
    def verify_confirm(sid: str, body: ConfirmBody):
        gate = gate_or_404(sid)
        try:
            gate.confirm(body.accept)
        except GateError as exc:
            raise HTTPException(409, str(exc))
        return gate.view()

    @app.post("/verify/{sid}/reset")
    def verify_reset(sid: str):
        gate = gate_or_404(sid)
        gate.reset()
        return gate.view()

    # ------------------------------------------------------------- tracking
    @app.post("/track/watch")
    def track_watch(body: WatchBody):
        unknown = [p for p in body.patient_ids if store.get(p) is None]
        if unknown:
            raise HTTPException(404, f"not registered: {unknown}")
        return {"watching": tracker.watch(body.patient_ids), "max_tracked": s.max_tracked}

    @app.delete("/track/watch")
    def track_clear():
        tracker.clear()
        return {"watching": []}

    @app.post("/track/frame")
    async def track_frame(request: Request):
        """Robot posts a camera frame; response says whether each watched patient is present or lost."""
        img = await read_image(request)
        return await run_in_threadpool(tracker.update, img)

    @app.get("/track/status")
    def track_status():
        return tracker.status()

    @app.get("/track/events")
    def track_events(since: int = 0):
        return {"events": tracker.events_since(since)}

    @app.get("/camera/snapshot.jpg")
    def snapshot():
        from fastapi.responses import Response
        cam = app.state.camera
        frame = cam.latest_frame() if cam else None
        if frame is None:
            raise HTTPException(404, "no local camera running")
        ok, buf = cv2.imencode(".jpg", frame)
        return Response(buf.tobytes(), media_type="image/jpeg")

    def _try(fn):
        try:
            fn()
        except Exception as exc:  # gateway trouble must never break identification
            print(f"[gateway] {exc}")

    return app


def _decode_qr(img: np.ndarray) -> str | None:
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    return text or None


def main():
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="Patient identification service")
    ap.add_argument("--camera", type=int, default=None, help="use a local camera as the BOT camera (index)")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    s = config.load()
    if args.camera is not None:
        s.camera = args.camera
    if args.port:
        s.port = args.port
    app = create_app(s)
    if s.camera is not None:
        from .camera_feed import CameraFeed
        app.state.camera = CameraFeed(s.camera, app.state.tracker, s.track_fps)
        app.state.camera.start()
    uvicorn.run(app, host=s.host, port=s.port, log_level="warning")


if __name__ == "__main__":
    main()
