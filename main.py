"""Real-time face detection + recognition against a few reference images,
with a single-window live editor (video on the left, settings panel on the right).

Reference images live in ./reference_images and are named like
    <anything>_base<N>_<name>.jpg      e.g. sitting_base1_abdullah.jpg

Detector:   YuNet  (OpenCV FaceDetectorYN)
Recognizer: SFace  (OpenCV FaceRecognizerSF, 128-d embeddings, cosine match)

Usage:
    python main.py                 # camera 1 (default)
    python main.py --camera 0

Panel: click presets, drag sliders, click toggles, mouse wheel scrolls the panel.
Keys:  s = save settings now   r = reset to this camera's default preset   q / ESC = quit
Settings are auto-saved on exit to settings_ui_camera<N>.json and reloaded next time.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from air_play import COLORS, AirPlay

HERE = Path(__file__).resolve().parent
REF_DIR = HERE / "reference_images"
DETECTOR_MODEL = HERE / "models" / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = HERE / "models" / "face_recognition_sface_2021dec.onnx"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
WINDOW = "Face Recognition"

# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------
# Panel layout: ("hwdefaults",) | ("header", text) | ("presets",) | ("toggle", key, label) | ("slider", key, label, lo, hi)
LAYOUT = [
    ("header", "Presets"),
    ("presets",),
    ("header", "Image"),
    ("toggle", "enhance", "Enhancement on"),
    ("slider", "brightness", "Brightness", -100, 100),
    ("slider", "contrast", "Contrast %", 50, 200),
    ("slider", "saturation", "Saturation %", 0, 300),
    ("slider", "ambient", "Ambient (shadow fill)", 0, 100),
    ("header", "Advanced image"),
    ("toggle", "auto_levels", "Auto levels (de-haze)"),
    ("slider", "levels_clip", "Auto levels clip x0.1%", 0, 50),
    ("slider", "black", "Black point", 0, 120),
    ("slider", "white", "White point", 130, 255),
    ("toggle", "auto_gamma", "Auto gamma (from scene)"),
    ("slider", "target", "Auto gamma target", 20, 200),
    ("slider", "gamma", "Gamma x100", 20, 300),
    ("slider", "clahe", "Local contrast %", 0, 100),
    ("slider", "clahe_clip", "Local contrast clip x0.1", 1, 60),
    ("slider", "clahe_tile", "Local contrast tiles", 1, 16),
    ("header", "Recognition"),
    ("slider", "det_score", "Detection confidence %", 30, 99),
    ("slider", "match", "Match threshold x0.01", 10, 80),
    ("toggle", "show_raw", "Show raw beside result"),
    ("header", "Camera hardware (sent to camera on change)"),
    ("hwdefaults",),
    ("toggle", "hw_auto_exp", "Auto exposure"),
    ("slider", "hw_exposure", "Exposure (log2 s)", -13, 0),
    ("slider", "hw_gain", "Gain", 0, 255),
    ("slider", "hw_bright", "Brightness", 0, 255),
    ("slider", "hw_contrast", "Contrast", 0, 255),
    ("slider", "hw_sat", "Saturation", 0, 255),
    ("toggle", "hw_auto_wb", "Auto white balance"),
]

# Everything an image preset controls, with its neutral ("camera as-is") value.
IMAGE_NEUTRAL = {
    "enhance": 1, "brightness": 0, "contrast": 100, "saturation": 100, "ambient": 0,
    "auto_levels": 0, "levels_clip": 10, "black": 0, "white": 255,
    "auto_gamma": 0, "target": 60, "gamma": 100,
    "clahe": 0, "clahe_clip": 20, "clahe_tile": 6,
}
RECOG_DEFAULT = {"det_score": 85, "match": 36, "show_raw": 0}

# Hardware keys -> (cv2 property, value sent when slider value v is used)
HARDWARE = {
    "hw_auto_exp": cv2.CAP_PROP_AUTO_EXPOSURE,   # toggle: 0.75 = auto, 0.25 = manual (DirectShow)
    "hw_exposure": cv2.CAP_PROP_EXPOSURE,
    "hw_gain": cv2.CAP_PROP_GAIN,
    "hw_bright": cv2.CAP_PROP_BRIGHTNESS,
    "hw_contrast": cv2.CAP_PROP_CONTRAST,
    "hw_sat": cv2.CAP_PROP_SATURATION,
    "hw_auto_wb": cv2.CAP_PROP_AUTO_WB,
}

# Camera 1's own driver values as first found (before any tuning); "Restore camera defaults" sets these.
# The driver remembers settings between runs, so this is the way back after experimenting.
HARDWARE_DEFAULTS = {"hw_auto_exp": 1, "hw_exposure": -6, "hw_gain": 128, "hw_bright": 128,
                     "hw_contrast": 64, "hw_auto_wb": 1}
# Startup profile: camera defaults with auto exposure on (the driver can be left in manual mode by
# earlier runs, which gives a black picture) and the camera's brightness nudged up from 128.
HARDWARE_STARTUP = {**HARDWARE_DEFAULTS, "hw_bright": 165}

# Presets only touch the image keys; the camera hardware is left at its own defaults.
PRESETS = {
    "Camera default": {"enhance": 0},
    # a little brighter, a little more saturated, a little less contrast, a little ambient fill
    "Subtle": {"brightness": 20, "contrast": 92, "saturation": 112, "ambient": 35},
    "Low light": {"auto_levels": 1, "auto_gamma": 1, "target": 60, "clahe": 30,
                  "saturation": 115, "ambient": 30},
    "Backlit": {"auto_gamma": 1, "target": 75, "ambient": 60, "clahe": 40,
                "saturation": 115, "contrast": 95},
    "Vivid": {"brightness": 10, "contrast": 110, "saturation": 140},
    "My preset": {},  # filled from the saved file, or Subtle until you save your own
}
PRESET_ORDER = list(PRESETS)


def preset_values(name, presets):
    return {**IMAGE_NEUTRAL, **presets[name]}


def default_preset_name(camera):
    return "Camera default" if camera == 0 else "Subtle"  # camera 0 looks fine untouched


def settings_path(camera):
    return HERE / f"settings_ui_camera{camera}.json"


# ---------------------------------------------------------------------------
# Image enhancement
# ---------------------------------------------------------------------------
class Enhancer:
    def __init__(self):
        self.gamma_smooth = None  # smoothed auto values, avoid flicker
        self.levels = None

    def reset(self):
        self.gamma_smooth = self.levels = None

    def __call__(self, frame, s):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0]

        if s["auto_levels"]:
            # stretch from the scene's own darkest/brightest pixels: removes grey haze
            lo, hi = np.percentile(l[::4, ::4], [s["levels_clip"] / 10.0, 100 - s["levels_clip"] / 10.0])
            hi = max(hi, lo + 80)  # don't blow up noise in very dark scenes
            new = (lo, hi)
            self.levels = new if self.levels is None else tuple(0.9 * a + 0.1 * b for a, b in zip(self.levels, new))
            bp, wp = self.levels[0] / 255.0, self.levels[1] / 255.0
        else:
            bp, wp = s["black"] / 255.0, max(s["white"], s["black"] + 10) / 255.0
        lv = np.clip((np.linspace(0, 1, 256) - bp) / (wp - bp), 0, 1)

        if s["auto_gamma"]:
            med = min(max(float(np.median(cv2.LUT(l, (lv * 255).astype(np.uint8)))), 1.0), 254.0)
            g = np.clip(np.log(s["target"] / 255) / np.log(med / 255), 0.4, 1.5)
            self.gamma_smooth = g if self.gamma_smooth is None else 0.9 * self.gamma_smooth + 0.1 * g
            gamma = self.gamma_smooth
        else:
            gamma = s["gamma"] / 100.0

        c = lv ** gamma
        c = c + (s["ambient"] / 100.0) * 2.5 * c * (1 - c) ** 2  # lifts shadows/low mids, keeps black & white fixed
        c = (c - 0.5) * (s["contrast"] / 100.0) + 0.5 + s["brightness"] / 255.0
        l = cv2.LUT(l, (np.clip(c, 0, 1) * 255).astype(np.uint8))

        if s["clahe"]:
            clahe = cv2.createCLAHE(s["clahe_clip"] / 10.0, (s["clahe_tile"],) * 2)
            a = s["clahe"] / 100.0
            l = cv2.addWeighted(l, 1 - a, clahe.apply(l), a, 0)

        lab[:, :, 0] = l
        if s["saturation"] != 100:
            ab = lab[:, :, 1:].astype(np.float32)
            lab[:, :, 1:] = np.clip((ab - 128) * (s["saturation"] / 100.0) + 128, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Face detection / recognition
# ---------------------------------------------------------------------------
def person_name(path: Path) -> str:
    m = re.search(r"base\d+_(.+)$", path.stem, re.IGNORECASE)
    return (m.group(1) if m else path.stem).replace("_", " ").title()


def make_detector(size=(320, 320), score_threshold=0.85):
    return cv2.FaceDetectorYN.create(str(DETECTOR_MODEL), "", size, score_threshold, 0.3, 5000)


def detect(detector, frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    return faces if faces is not None else np.empty((0, 15), dtype=np.float32)


def embed(recognizer, frame, face):
    return recognizer.feature(recognizer.alignCrop(frame, face))


def load_references(detector, recognizer):
    """Return {name: [embedding, ...]} built from the reference images."""
    detector.setScoreThreshold(0.7)  # reference photos are clean; be lenient
    refs = {}
    for path in sorted(REF_DIR.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"[warn] cannot read {path.name}")
            continue
        faces = detect(detector, img)
        if len(faces) == 0:
            print(f"[warn] no face found in {path.name}, skipping")
            continue
        name = person_name(path)
        embs = [embed(recognizer, img, max(faces, key=lambda f: f[2] * f[3]))]
        flipped = cv2.flip(img, 1)  # mirrored copy: more robust with so few references
        ff = detect(detector, flipped)
        if len(ff):
            embs.append(embed(recognizer, flipped, max(ff, key=lambda f: f[2] * f[3])))
        refs.setdefault(name, []).extend(embs)
        print(f"[ref] {name}: {path.name} ({len(embs)} embeddings)")
    return refs


def identify(recognizer, emb, refs, threshold):
    best_name, best_score = "Unknown", -1.0
    for name, embs in refs.items():
        score = max(recognizer.match(emb, e, cv2.FaceRecognizerSF_FR_COSINE) for e in embs)
        if score > best_score:
            best_name, best_score = name, score
    return ("Unknown" if best_score < threshold else best_name), best_score


# ---------------------------------------------------------------------------
# In-window UI (panel drawn next to the video, handled with mouse events)
# ---------------------------------------------------------------------------
PANEL_W = 440
ROW_H = 26
TRACK_X0, TRACK_X1 = 225, PANEL_W - 75
BG, FG, DIM, ACCENT = (32, 32, 32), (225, 225, 225), (140, 140, 140), (255, 170, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, x, y, color=FG, scale=0.46, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


class Panel:
    def __init__(self, vals, active_preset):
        self.vals = vals
        self.active_preset = active_preset
        self.scroll = 0
        self.content_h = 0
        self.video_w = 0  # x offset of the panel in the window
        self.height = 0
        self.hits = []  # (kind, key, (x0, y0, x1, y1)) in panel coordinates
        self.drag = None  # (key, lo, hi)
        self.actions = []  # ("preset", name) | ("save_preset",) | ("reset",)

    # -- drawing -----------------------------------------------------------
    def render(self, height, layout):
        self.height = height
        img = np.full((height, PANEL_W, 3), BG, np.uint8)
        self.hits = []
        y = 10 - self.scroll
        for item in layout:
            kind = item[0]
            if kind == "note":
                text(img, item[1], 12, y + 14, DIM, 0.42)
                y += 20
            elif kind == "buttons":
                items = item[1]
                per_row, pad = 3, 12
                bw = (PANEL_W - 2 * pad - (per_row - 1) * 6) // per_row
                for i, (label, action, active) in enumerate(items):
                    r, c = divmod(i, per_row)
                    self._button(img, label, pad + c * (bw + 6), y + r * 32, bw, 26, active, action)
                y += ((len(items) + per_row - 1) // per_row) * 32
            elif kind == "swatches":
                for i, (name, bgr) in enumerate(item[1]):
                    x0 = 12 + i * 50
                    cv2.rectangle(img, (x0, y), (x0 + 42, y + 30), bgr, -1)
                    if i == item[2]:
                        cv2.rectangle(img, (x0 - 3, y - 3), (x0 + 45, y + 33), (255, 255, 255), 2)
                    self.hits.append(("button", ("color", i), (x0, y, x0 + 42, y + 30)))
                y += 40
            elif kind == "header":
                y += 8
                text(img, item[1].upper(), 12, y + 14, ACCENT, 0.45, 1)
                cv2.line(img, (12, y + 20), (PANEL_W - 12, y + 20), (70, 70, 70), 1)
                y += ROW_H
            elif kind == "presets":
                y = self._presets(img, y)
            elif kind == "hwdefaults":
                self._button(img, "Restore camera defaults", 12, y, PANEL_W - 24, 26, False, ("hw_defaults",))
                y += 32
            elif kind == "toggle":
                _, key, label = item
                on = bool(self.vals[key])
                text(img, label, 12, y + 17)
                x0 = TRACK_X0
                cv2.rectangle(img, (x0, y + 3), (x0 + 46, y + 21), (0, 150, 0) if on else (70, 70, 70), -1)
                cv2.circle(img, (x0 + 36 if on else x0 + 10, y + 12), 7, FG, -1)
                text(img, "ON" if on else "OFF", x0 + 58, y + 17, FG if on else DIM)
                self.hits.append(("toggle", key, (0, y, PANEL_W, y + ROW_H)))
                y += ROW_H
            else:
                _, key, label, lo, hi = item
                v = self.vals[key]
                text(img, label, 12, y + 17)
                cy = y + 13
                cv2.line(img, (TRACK_X0, cy), (TRACK_X1, cy), (80, 80, 80), 3)
                kx = int(TRACK_X0 + (v - lo) / (hi - lo) * (TRACK_X1 - TRACK_X0))
                cv2.line(img, (TRACK_X0, cy), (kx, cy), ACCENT, 3)
                cv2.circle(img, (kx, cy), 7, FG, -1)
                text(img, str(int(v)), TRACK_X1 + 16, y + 17)
                self.hits.append(("slider", (key, lo, hi), (0, y, PANEL_W, y + ROW_H)))
                y += ROW_H
        self.content_h = y + self.scroll + 10
        if self.content_h > height:  # scrollbar
            bar_h = max(30, int(height * height / self.content_h))
            bar_y = int(self.scroll / (self.content_h - height) * (height - bar_h))
            cv2.rectangle(img, (PANEL_W - 6, bar_y), (PANEL_W - 2, bar_y + bar_h), (110, 110, 110), -1)
        return img

    def _presets(self, img, y):
        pad, per_row = 12, 3
        w = (PANEL_W - 2 * pad - (per_row - 1) * 6) // per_row
        names = PRESET_ORDER
        for i, name in enumerate(names):
            r, c = divmod(i, per_row)
            x0 = pad + c * (w + 6)
            y0 = y + r * 32
            self._button(img, name, x0, y0, w, 26, name == self.active_preset, ("preset", name))
        y += ((len(names) + per_row - 1) // per_row) * 32
        w2 = (PANEL_W - 2 * pad - 6) // 2
        self._button(img, "Save as My preset", pad, y, w2, 26, False, ("save_preset",))
        self._button(img, "Reset image", pad + w2 + 6, y, w2, 26, False, ("reset",))
        return y + 32

    def _button(self, img, label, x, y, w, h, active, action):
        cv2.rectangle(img, (x, y), (x + w, y + h), ACCENT if active else (65, 65, 65), -1)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.44, 1)
        text(img, label, x + (w - tw) // 2, y + (h + th) // 2, (20, 20, 20) if active else FG, 0.44)
        self.hits.append(("button", action, (x, y, x + w, y + h)))

    # -- mouse -------------------------------------------------------------
    def _set_slider(self, px, key, lo, hi):
        f = min(max((px - TRACK_X0) / (TRACK_X1 - TRACK_X0), 0.0), 1.0)
        self.vals[key] = int(round(lo + f * (hi - lo)))

    def mouse(self, event, x, y, flags, _param):
        px = x - self.video_w
        if event == cv2.EVENT_MOUSEWHEEL:
            if px >= 0:
                self.scroll -= 60 if flags > 0 else -60
                self.scroll = min(max(self.scroll, 0), max(self.content_h - self.height, 0))
            return
        if event == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag and flags & cv2.EVENT_FLAG_LBUTTON:
            self._set_slider(px, *self.drag)
        elif event == cv2.EVENT_LBUTTONDOWN and px >= 0:
            for kind, key, (x0, y0, x1, y1) in self.hits:
                if not (x0 <= px < x1 and y0 <= y < y1):
                    continue
                if kind == "toggle":
                    self.vals[key] = 0 if self.vals[key] else 1
                elif kind == "slider":
                    self.drag = key
                    self._set_slider(px, *key)
                else:
                    self.actions.append(key)
                break


TABS = ["Recognition", "Air Play"]
TAB_H = 38
AIR_ACTIONS = {"mode", "color", "undo", "clear", "save", "ttt_new"}


class Tabs:
    """Tab bar drawn above the video + panel; clicking a tab sets `selected`."""

    def __init__(self):
        self.selected = None
        self.rects = []

    def render(self, width, active):
        bar = np.full((TAB_H, width, 3), (22, 22, 22), np.uint8)
        self.rects = []
        x = 8
        for i, name in enumerate(TABS):
            (tw, th), _ = cv2.getTextSize(name, FONT, 0.6, 1)
            w = tw + 36
            colour = BG if i == active else (50, 50, 50)
            cv2.rectangle(bar, (x, 6), (x + w, TAB_H), colour, -1)
            if i == active:
                cv2.line(bar, (x, 6), (x + w, 6), ACCENT, 3)
            text(bar, name, x + 18, 26, FG if i == active else DIM, 0.6)
            self.rects.append((x, x + w))
            x += w + 4
        return bar

    def click(self, event, x):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (x0, x1) in enumerate(self.rects):
                if x0 <= x < x1:
                    self.selected = i


def air_layout(air, vals):
    mode = air.mode
    items = [
        ("header", "Mode"),
        ("buttons", [("Draw", ("mode", "draw"), mode == "draw"),
                     ("Tic-Tac-Toe", ("mode", "ttt"), mode == "ttt")]),
    ]
    if mode == "draw":
        items += [
            ("header", "Draw"),
            ("swatches", COLORS, air.draw.color),
            ("buttons", [("Undo (z)", ("undo",), False), ("Clear (c)", ("clear",), False),
                         ("Save PNG", ("save",), False)]),
            ("slider", "brush", "Brush size", 2, 40),
            ("note", "1 finger: draw    2 fingers: move without drawing"),
            ("note", "Open palm: erase nearby strokes"),
            ("note", "Point at the colour bar (top-left) to pick a colour"),
        ]
    else:
        items += [
            ("header", "Tic-Tac-Toe"),
            ("buttons", [("New game (n)", ("ttt_new",), False)]),
            ("toggle", "vs_cpu", "Play vs computer (you are X)"),
            ("slider", "hold_ms", "Hold time (ms)", 300, 2500),
            ("note", "Point at a square with one finger and hold still."),
            ("note", "Two players take turns: X goes first."),
        ]
    items += [
        ("header", "Hand tracking"),
        ("toggle", "skeleton", "Show hand skeleton"),
        ("slider", "smooth", "Smoothing", 0, 90),
        ("slider", "bg_dim", "Dim camera %", 0, 90),
        ("note", f"Hands detected: {air.hand_count}"),
        ("note", "Image tuning (brightness etc.) is in the Recognition tab."),
    ]
    return items


def active_preset_name(vals, presets):
    for name in PRESET_ORDER:
        if all(vals[k] == v for k, v in preset_values(name, presets).items()):
            return name
    return None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_state(path, camera):
    """Returns (vals, my_preset). vals has image + recognition keys (hardware is filled in later)."""
    presets = dict(PRESETS)
    presets["My preset"] = dict(PRESETS[default_preset_name(camera)])
    vals = {**preset_values(default_preset_name(camera), presets), **RECOG_DEFAULT}
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if "my_preset" in data:
                presets["My preset"] = {k: v for k, v in data["my_preset"].items() if k in IMAGE_NEUTRAL}
            vals.update({k: v for k, v in data.get("vals", {}).items()
                         if k in IMAGE_NEUTRAL or k in RECOG_DEFAULT or k in HARDWARE})
            print(f"Loaded {path.name}")
        except (OSError, ValueError, AttributeError):
            print(f"[warn] could not read {path.name}, using defaults")
    return vals, presets


def save_state(path, vals, presets):
    keys = set(IMAGE_NEUTRAL) | set(RECOG_DEFAULT) | set(HARDWARE)
    path.write_text(json.dumps({
        "vals": {k: v for k, v in vals.items() if k in keys},
        "my_preset": {k: vals_ for k, vals_ in presets["My preset"].items()},
    }, indent=2))


def init_hardware(cap, vals):
    """Show the camera's current driver values on the hardware sliders (does not change the camera)."""
    ranges = {item[1]: (item[3], item[4]) for item in LAYOUT if item[0] == "slider"}
    for key, prop in HARDWARE.items():
        cur = cap.get(prop)
        if key == "hw_auto_exp":
            v = 1 if (cur == -1.0 or cur >= 0.5) else 0
        elif key == "hw_auto_wb":
            v = 1 if cur != 0 else 0
        else:
            lo, hi = ranges[key]
            v = int(round(min(max(cur if cur == cur else lo, lo), hi)))
        vals[key] = v
    return dict((k, vals[k]) for k in HARDWARE)  # what the camera currently has


def apply_hardware(cap, vals, applied):
    for key, prop in HARDWARE.items():
        if vals[key] == applied[key]:
            continue
        if key == "hw_exposure" and vals["hw_auto_exp"]:
            continue  # setting exposure would switch the camera to manual mode
        applied[key] = vals[key]
        if key == "hw_auto_exp":
            cap.set(prop, 0.75 if vals[key] else 0.25)
        else:
            cap.set(prop, vals[key])


def draw_label(frame, label, x, y, color):
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.7, 2)
    top = max(y - th - 10, 0)
    cv2.rectangle(frame, (x, top), (x + tw + 6, top + th + 10), color, -1)
    cv2.putText(frame, label, (x + 3, top + th + 3), FONT, 0.7, (255, 255, 255), 2)


def fit(img, max_w, max_h):
    h, w = img.shape[:2]
    k = min(1.0, max_w / w, max_h / h)
    return img if k == 1.0 else cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1, help="camera index (default 1)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    detector = make_detector()
    recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_MODEL), "")
    refs = load_references(detector, recognizer)
    if not refs:
        sys.exit("No usable reference faces found in " + str(REF_DIR))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # DirectShow: most reliable for USB cams on Windows
    if args.camera == 0:  # camera 1 only works at its native resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}.")

    path = settings_path(args.camera)
    vals, presets = load_state(path, args.camera)
    saved_hw = {k: vals[k] for k in HARDWARE if k in vals}
    init_hardware(cap, vals)  # start sliders from what the camera reports...
    if args.camera == 0:  # camera 0 looks fine as it is: leave its hardware alone
        applied = dict((k, vals[k]) for k in HARDWARE)
    else:
        vals.update(HARDWARE_STARTUP)  # startup profile, or your saved hardware settings
        vals.update(saved_hw)
        applied = {k: None for k in HARDWARE}  # None = send everything once at start
    last_exposure = vals["hw_exposure"]
    vals.update(AirPlay.DEFAULTS)
    panel = Panel(vals, active_preset_name(vals, presets))
    enhancer = Enhancer()
    air = AirPlay()
    tab = 0  # index into TABS
    tabs = Tabs()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        if y < TAB_H:
            tabs.click(event, x)
        else:
            panel.mouse(event, x, y - TAB_H, flags, param)

    cv2.setMouseCallback(WINDOW, on_mouse)

    toast, toast_until = "", 0.0
    tick, fps = cv2.getTickCount(), 0.0
    print("Running. Tab = switch tab, s = save, r = reset, q/ESC = quit.")

    while True:
        ok, raw = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        if tabs.selected is not None:
            if tabs.selected != tab:
                tab, panel.scroll = tabs.selected, 0
            tabs.selected = None

        for action in panel.actions:
            if action[0] in AIR_ACTIONS:
                air.handle(action)
                continue
            if action[0] == "preset":
                vals.update(preset_values(action[1], presets))
                enhancer.reset()
                toast = action[1]
            elif action[0] == "save_preset":
                presets["My preset"] = {k: vals[k] for k in IMAGE_NEUTRAL}
                save_state(path, vals, presets)
                toast = "Saved as My preset"
            elif action[0] == "hw_defaults":
                vals.update(HARDWARE_STARTUP)
                last_exposure = vals["hw_exposure"]
                toast = "Camera defaults restored"
            elif action[0] == "reset":
                vals.update(preset_values(default_preset_name(args.camera), presets))
                enhancer.reset()
                toast = "Reset"
            toast_until = time.time() + 2
        panel.actions.clear()

        if vals["hw_exposure"] != last_exposure:  # moving the exposure slider means manual exposure
            vals["hw_auto_exp"] = 0
            last_exposure = vals["hw_exposure"]
        apply_hardware(cap, vals, applied)
        detector.setScoreThreshold(vals["det_score"] / 100.0)
        frame = enhancer(raw, vals) if vals["enhance"] else raw.copy()

        if TABS[tab] == "Air Play":
            frame = air.process(cv2.flip(frame, 1), vals)  # mirrored, like a mirror
            layout = air_layout(air, vals)
        else:
            layout = LAYOUT
            for face in detect(detector, frame):
                x, y, w, h = face[:4].astype(int)
                name, score = identify(recognizer, embed(recognizer, frame, face), refs, vals["match"] / 100.0)
                color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                draw_label(frame, f"{name} {score:.2f}", x, y, color)

        now = cv2.getTickCount()
        fps = 0.9 * fps + 0.1 * (cv2.getTickFrequency() / (now - tick))
        tick = now
        cv2.putText(frame, f"{fps:.1f} FPS", (10, 25), FONT, 0.7, (255, 255, 0), 2)
        if time.time() < toast_until:
            cv2.putText(frame, toast, (10, 55), FONT, 0.7, (0, 255, 255), 2)

        show_raw = vals["show_raw"] and TABS[tab] == "Recognition"
        video = cv2.hconcat([raw, frame]) if show_raw else frame
        video = fit(video, 1100, 720)
        height = max(video.shape[0], 720)
        if video.shape[0] < height:
            video = cv2.copyMakeBorder(video, 0, height - video.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=BG)

        panel.video_w = video.shape[1]
        panel.active_preset = active_preset_name(vals, presets)
        panel.scroll = min(panel.scroll, max(panel.content_h - height, 0))
        body = cv2.hconcat([video, panel.render(height, layout)])
        cv2.imshow(WINDOW, cv2.vconcat([tabs.render(body.shape[1], tab), body]))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == 9:  # Tab
            tabs.selected = (tab + 1) % len(TABS)
        elif key == ord("z"):
            air.handle(("undo",))
        elif key == ord("c") and TABS[tab] == "Air Play":
            air.handle(("clear",))
        elif key == ord("n"):
            air.handle(("ttt_new",))
        if key == ord("s"):
            save_state(path, vals, presets)
            toast, toast_until = f"Saved {path.name}", time.time() + 2
        elif key == ord("r"):
            panel.actions.append(("reset",))
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    save_state(path, vals, presets)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
