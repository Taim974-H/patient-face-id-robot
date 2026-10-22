# Patient identification: robot / ACU integration

Built from `robot_specs/` (problem statements, API guide, KettyBot Pro specs). This is the **Patient Validator**
domain service from the Component View (slide 4): it runs locally on the ACU server behind the API Gateway,
receives camera images from the BOT, and answers "is this the right patient, and are they still here?".

```
KettyBot Pro --(image)--> ACU API Gateway --> patient_id service (this) --> result / events
      ^                                                                      |
      +--------- confirmation screen + guidance command <--------------------+
```

## Which problem statements it covers

| Slide | Requirement | Where |
|---|---|---|
| 7 Patient identification | Face match, QR code, **confirm** before release, **max 3 attempts**, stay **blocked**, **latest 5 patients**, **< 2 GB**, local | `verification.py`, `store.py`, `/verify/*` |
| 5 Continuous guidance | Register, verify, **notify if patient is lost**, **latest 2 patients**, **<= 3 s** feedback, local | `tracker.py`, `/track/*` |
| 2 Movement sensor | "Someone is in front of me", approaching | `bearing` and `proximity` in `/track/status` |
| 7 Voice verification | not built (needs a speech model) | out of scope |

Measured on this PC (CPU only): about 200 MB RAM, 160 ms per identification (limit 3 s, 2 GB).

## Run it

```powershell
pip install -r requirements.txt
python -m patient_id.enroll --from-reference-images        # registers Abdullah and Taimoor
python -m patient_id.service                                # frames arrive over HTTP from the BOT
python -m patient_id.service --camera 1                     # or use the USB camera as the BOT camera
python selftest.py                                          # 26 end-to-end checks, no hardware needed
```

Swagger UI: `http://<acu-host>:8080/docs`. Settings: `patient_id_config.json` or `PID_<NAME>` env vars
(see `patient_id/config.py`).

## API (all JSON; images as multipart field `file`, raw `image/jpeg` body, or `{"image_base64": "..."}`)

| Call | Purpose |
|---|---|
| `POST /patients` multipart `patient_id`, `name`, `qr_code`, `files` (1+ photos) | Register. Keeps only the latest 5 (least recently used is evicted). Stores embeddings, not photos. |
| `POST /verify/start` `{session_id, expected_patient_id}` | Open the gate for one scan workflow |
| `POST /verify/{sid}/face` image | One identification attempt. Wrong person, unknown person: attempt used. No face in frame: **not** counted |
| `POST /verify/{sid}/qr` image or `{"code": "..."}` | Same, by registered QR code |
| `POST /verify/{sid}/confirm` `{accept: true/false}` | User accepts or rejects. Only accept sets `released: true` |
| `POST /verify/{sid}/reset` | Operator override after `blocked` |
| `POST /track/watch` `{patient_ids: [...]}` | Watch up to 2 patients |
| `POST /track/frame` image | Per-frame status: `present` or `lost`, `bearing` (-1 left, +1 right), `proximity` (bigger = closer) |
| `GET /track/status`, `GET /track/events?since=N` | Poll status / events (`patient_detected`, `patient_lost`, `patient_reacquired`) |
| `POST /identify` image | Stateless "who is this" |
| `GET /health` | Memory, patient count |

Gate states: `ready` -> `awaiting_confirmation` -> `confirmed` (released) or `blocked` (3 failures). The
response `prompt` field is text the robot can show or speak ("Are you Abdullah?").

## Talking to the ACU API Gateway (`patient_id/gateway.py`)

Routes come from `robot_specs/api-guide.pdf`. It runs as a **dry run** (prints, sends nothing) until you set:

```json
{ "gateway_url": "http://<gateway-host>:3000", "robot_device_id": "<your approved device>",
  "publish_confirmation": true, "gateway_notify": true }
```

- `publish_confirmation`: on a face match, creates and publishes a Yes/No **Confirmation** workflow screen on
  the robot (`POST /api/workflows/robot`, then `POST /api/acu/robot-workflows/{id}/assignments/{deviceId}`).
- `gateway_notify`: on `patient_lost` / `patient_reacquired`, sends the command string via
  `POST /api/acu/instructions/{deviceId}` so the robot can trigger corrective guidance.

## Things to confirm with the organizers (the guide does not say)

1. **How BOT images reach this service.** The guide has no image-upload route. Ask which gateway route
   forwards BOT frames, or have the robot client POST to `/track/frame` and `/verify/*` directly.
2. **Command strings.** The gateway has no command catalogue. Agree what the robot does on
   `patient_lost` / `patient_found` and change `command_patient_lost` / `command_patient_found`.
3. **How the robot reports Yes/No.** Workflow buttons only do Next/Previous. The robot client should call
   `/verify/{sid}/confirm`, or you read the result from `recentEvents` in `GET /api/acu/devices`.
4. **Use your assigned device only.** Writes change shared state; do not post telemetry as someone else's robot.

## Known limits

- Matching margin is thin on these reference photos (Taimoor scored 0.37 live against a 0.363 threshold). Register
  2-3 front-facing, well-lit photos per patient (`POST /patients` accepts several files) before the demo.
- Face match alone is not enough for patient safety. Keep the confirmation step and the QR fallback.
- Poor lighting hurts: dark frames are auto-brightened, but a backlit patient is still harder.
- Registrations are in `data/patients.json` (embeddings only). Delete it to wipe them.
