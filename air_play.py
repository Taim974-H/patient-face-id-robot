"""Air Play: finger tracking for drawing in the air and two-player tic-tac-toe.

Hand tracking uses MediaPipe Hands (21 landmarks per hand, up to 2 hands).

Draw tab gestures (the video is mirrored, so it behaves like a mirror):
    index finger only up      -> draw
    index + middle up         -> pen lifted (just moves the cursor)
    open palm (4+ fingers)    -> eraser: deletes strokes near the palm
    point at the colour bar   -> pick a colour
Tic-tac-toe: point with your index finger at a square and hold still until the ring fills.
"""
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent

COLORS = [  # (name, BGR)
    ("Red", (60, 60, 240)),
    ("Orange", (0, 150, 255)),
    ("Yellow", (0, 230, 255)),
    ("Green", (80, 210, 80)),
    ("Cyan", (240, 220, 0)),
    ("Magenta", (220, 70, 220)),
    ("White", (255, 255, 255)),
]
X_COLOR, O_COLOR = (0, 150, 255), (240, 220, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

TIPS = (8, 12, 16, 20)   # index, middle, ring, pinky fingertips
PIPS = (6, 10, 14, 18)


class HandTracker:
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.5)
        self.connections = list(mp.solutions.hands.HAND_CONNECTIONS)

    def process(self, bgr):
        """Return a list of (21, 2) pixel-coordinate arrays, left-to-right on screen."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.hands.process(rgb)
        hands = []
        for hand in res.multi_hand_landmarks or []:
            hands.append(np.array([[p.x * w, p.y * h] for p in hand.landmark], np.float32))
        hands.sort(key=lambda p: p[0, 0])
        return hands

    def draw_skeleton(self, img, pts):
        for a, b in self.connections:
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (200, 200, 200), 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p.astype(int)), 3, (0, 0, 255), -1, cv2.LINE_AA)


def gesture(pts):
    """'point' | 'peace' | 'open' | 'other' from the 21 landmarks (works for rotated hands)."""
    wrist = pts[0]
    up = [np.linalg.norm(pts[t] - wrist) > 1.12 * np.linalg.norm(pts[p] - wrist) for t, p in zip(TIPS, PIPS)]
    idx, mid, ring, pinky = up
    if idx and not mid and not ring and not pinky:
        return "point"
    if idx and mid and not ring and not pinky:
        return "peace"
    if sum(up) >= 4:
        return "open"
    return "other"


class Smoother:
    """Exponential smoothing of one fingertip per hand slot, to calm jitter."""

    def __init__(self):
        self.pos = {}

    def __call__(self, slot, p, amount):
        a = 1.0 - amount / 100.0  # amount 0 = raw, 90 = very smooth
        old = self.pos.get(slot)
        self.pos[slot] = p if old is None else old + a * (p - old)
        return self.pos[slot]

    def keep_only(self, slots):
        self.pos = {k: v for k, v in self.pos.items() if k in slots}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
class DrawBoard:
    def __init__(self):
        self.strokes = []      # [color, size, [points]]
        self.current = {}      # hand slot -> stroke being drawn
        self.color = 0
        self.last_pick = 0.0

    def undo(self):
        if self.strokes:
            stroke = self.strokes.pop()
            for cur in [s for s, st in self.current.items() if st is stroke]:
                del self.current[cur]

    def clear(self):
        self.strokes.clear()
        self.current.clear()

    def palette_rects(self, w):
        size, gap = 44, 8
        return [(10 + i * (size + gap), 10, 10 + i * (size + gap) + size, 10 + size) for i in range(len(COLORS))]

    def in_palette(self, p, w):
        for i, (x0, y0, x1, y1) in enumerate(self.palette_rects(w)):
            if x0 - 6 <= p[0] <= x1 + 6 and y0 - 6 <= p[1] <= y1 + 6:
                return i
        return None

    def update(self, slot, pts, g, tip, size, w):
        """Apply one hand's gesture. Returns the cursor to draw: (position, kind, radius)."""
        if g == "point":
            pick = self.in_palette(tip, w)
            if pick is not None:
                self.current.pop(slot, None)
                self.color = pick
                return tip, "pick", 14
            stroke = self.current.get(slot)
            if stroke is None:
                stroke = [self.color, size, []]
                self.strokes.append(stroke)
                self.current[slot] = stroke
            pts_list = stroke[2]
            if not pts_list or np.linalg.norm(tip - pts_list[-1]) >= 2.0:
                pts_list.append(tip.copy())
            return tip, "draw", size
        self.current.pop(slot, None)
        if g == "peace":
            return tip, "hover", size
        if g == "open":
            centre = pts[9]
            radius = 55.0
            self.strokes[:] = [s for s in self.strokes
                               if not any(np.linalg.norm(p - centre) < radius for p in s[2])]
            self.current = {k: v for k, v in self.current.items() if any(v is s for s in self.strokes)}
            return centre, "erase", int(radius)
        return None, None, 0

    def lost(self, active_slots):
        for slot in [s for s in self.current if s not in active_slots]:
            del self.current[slot]

    def render(self, img):
        layer = np.zeros_like(img)
        for color, size, pts in self.strokes:
            color = COLORS[color][1] if isinstance(color, int) else color
            if len(pts) == 1:
                cv2.circle(layer, tuple(pts[0].astype(int)), size // 2, color, -1, cv2.LINE_AA)
            elif len(pts) > 1:
                cv2.polylines(layer, [np.array(pts, np.int32)], False, color, size, cv2.LINE_AA)
        mask = layer.any(axis=2)
        img[mask] = layer[mask]

    def draw_palette(self, img):
        for i, (x0, y0, x1, y1) in enumerate(self.palette_rects(img.shape[1])):
            cv2.rectangle(img, (x0, y0), (x1, y1), COLORS[i][1], -1)
            if i == self.color:
                cv2.rectangle(img, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (255, 255, 255), 3)

    def save(self, img):
        out = HERE / "drawings"
        out.mkdir(exist_ok=True)
        path = out / time.strftime("drawing_%Y%m%d_%H%M%S.png")
        cv2.imwrite(str(path), img)
        return path


# ---------------------------------------------------------------------------
# Tic-tac-toe
# ---------------------------------------------------------------------------
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner_of(board):
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a], (a, b, c)
    return None, None


def minimax(board, player, me):
    w, _ = winner_of(board)
    if w:
        return (1 if w == me else -1), None
    free = [i for i, v in enumerate(board) if not v]
    if not free:
        return 0, None
    best_score, best_move = (-2, None) if player == me else (2, None)
    for i in free:
        board[i] = player
        score, _ = minimax(board, "O" if player == "X" else "X", me)
        board[i] = ""
        if (player == me and score > best_score) or (player != me and score < best_score):
            best_score, best_move = score, i
    return best_score, best_move


class TicTacToe:
    def __init__(self):
        self.new_game()
        self.dwell = {}  # slot -> (target, start_time)

    def new_game(self):
        self.board = [""] * 9
        self.turn = "X"
        self.winner = None
        self.line = None
        self.cpu_at = None
        self.dwell = {}

    @property
    def over(self):
        return self.winner is not None or all(self.board)

    def geometry(self, w, h):
        side = int(min(w, h) * 0.72)
        x0, y0 = (w - side) // 2, (h - side) // 2 + 12
        return x0, y0, side

    def cell_at(self, p, w, h):
        x0, y0, side = self.geometry(w, h)
        if not (x0 <= p[0] < x0 + side and y0 <= p[1] < y0 + side):
            return None
        c = side / 3
        return int((p[1] - y0) // c) * 3 + int((p[0] - x0) // c)

    def play(self, i):
        if self.over or self.board[i]:
            return
        self.board[i] = self.turn
        self.winner, self.line = winner_of(self.board)
        self.turn = "O" if self.turn == "X" else "X"

    def update(self, tips, hold_s, vs_cpu, w, h, now):
        """tips: {slot: fingertip} for hands that are pointing. Returns {slot: progress 0..1}."""
        progress = {}
        if vs_cpu and not self.over and self.turn == "O":
            if self.cpu_at is None:
                self.cpu_at = now + 0.7
            elif now >= self.cpu_at:
                _, move = minimax(list(self.board), "O", "O")
                if move is not None:
                    self.play(move)
                self.cpu_at = None
            tips = {}  # the computer is thinking
        for slot in [s for s in self.dwell if s not in tips]:
            del self.dwell[slot]
        for slot, tip in tips.items():
            if self.over:
                target = "restart" if self._in_restart(tip, w, h) else None
            else:
                cell = self.cell_at(tip, w, h)
                target = cell if cell is not None and not self.board[cell] else None
            old = self.dwell.get(slot)
            if target is None:
                self.dwell.pop(slot, None)
                continue
            if old is None or old[0] != target:
                self.dwell[slot] = (target, now)
                continue
            frac = (now - old[1]) / hold_s
            progress[slot] = min(frac, 1.0)
            if frac >= 1.0:
                if target == "restart":
                    self.new_game()
                else:
                    self.play(target)
                self.dwell.pop(slot, None)
                break
        return progress

    def _restart_rect(self, w, h):
        bw, bh = 340, 44  # below the board so the final position stays visible
        return (w - bw) // 2, h - bh - 8, (w + bw) // 2, h - 8

    def _in_restart(self, p, w, h):
        x0, y0, x1, y1 = self._restart_rect(w, h)
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def status(self, vs_cpu):
        if self.winner:
            return f"{self.winner} wins!"
        if all(self.board):
            return "It's a draw"
        if vs_cpu and self.turn == "O":
            return "Computer is thinking..."
        return f"{self.turn}'s turn"

    def render(self, img, vs_cpu):
        h, w = img.shape[:2]
        x0, y0, side = self.geometry(w, h)
        c = side // 3
        shade = img.copy()
        cv2.rectangle(shade, (x0, y0), (x0 + side, y0 + side), (0, 0, 0), -1)
        cv2.addWeighted(shade, 0.35, img, 0.65, 0, img)
        for i in (1, 2):
            cv2.line(img, (x0 + i * c, y0), (x0 + i * c, y0 + side), (255, 255, 255), 4, cv2.LINE_AA)
            cv2.line(img, (x0, y0 + i * c), (x0 + side, y0 + i * c), (255, 255, 255), 4, cv2.LINE_AA)
        for i, v in enumerate(self.board):
            cx, cy = x0 + (i % 3) * c + c // 2, y0 + (i // 3) * c + c // 2
            r = int(c * 0.3)
            if v == "X":
                cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), X_COLOR, 8, cv2.LINE_AA)
                cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), X_COLOR, 8, cv2.LINE_AA)
            elif v == "O":
                cv2.circle(img, (cx, cy), r, O_COLOR, 8, cv2.LINE_AA)
        if self.line:
            a, _, b = self.line
            pa = (x0 + (a % 3) * c + c // 2, y0 + (a // 3) * c + c // 2)
            pb = (x0 + (b % 3) * c + c // 2, y0 + (b // 3) * c + c // 2)
            cv2.line(img, pa, pb, (0, 255, 0), 10, cv2.LINE_AA)
        msg = self.status(vs_cpu)
        colour = X_COLOR if self.turn == "X" and not self.over else O_COLOR if not self.over else (0, 255, 0)
        (tw, _), _ = cv2.getTextSize(msg, FONT, 1.1, 3)
        cv2.putText(img, msg, ((w - tw) // 2, max(y0 - 12, 34)), FONT, 1.1, colour, 3, cv2.LINE_AA)
        if self.over:
            rx0, ry0, rx1, ry1 = self._restart_rect(w, h)
            cv2.rectangle(img, (rx0, ry0), (rx1, ry1), (40, 40, 40), -1)
            cv2.rectangle(img, (rx0, ry0), (rx1, ry1), (255, 255, 255), 2)
            label = "Hold finger here: play again"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
            cv2.putText(img, label, ((rx0 + rx1 - tw) // 2, (ry0 + ry1 + th) // 2), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Everything the Air Play tab needs
# ---------------------------------------------------------------------------
class AirPlay:
    DEFAULTS = {"brush": 8, "smooth": 55, "bg_dim": 20, "hold_ms": 900, "skeleton": 1, "vs_cpu": 0}

    def __init__(self):
        self.mode = "draw"  # "draw" | "ttt"
        self.tracker = None
        self.smoother = Smoother()
        self.draw = DrawBoard()
        self.ttt = TicTacToe()
        self.toast, self.toast_until = "", 0.0
        self.last_view = None
        self.hand_count = 0

    def say(self, text):
        self.toast, self.toast_until = text, time.time() + 2

    def handle(self, action):
        kind = action[0]
        if kind == "mode":
            self.mode = action[1]
            self.smoother.pos.clear()
        elif kind == "color":
            self.draw.color = action[1]
        elif kind == "undo":
            self.draw.undo()
        elif kind == "clear":
            self.draw.clear()
            self.say("Cleared")
        elif kind == "save" and self.last_view is not None:
            self.say(f"Saved {self.draw.save(self.last_view).name}")
        elif kind == "ttt_new":
            self.ttt.new_game()

    def process(self, frame, vals):
        """frame: mirrored BGR image. Returns the image with hands, drawing / game on top."""
        if self.tracker is None:
            self.tracker = HandTracker()
        h, w = frame.shape[:2]
        hands = self.tracker.process(frame)
        self.hand_count = len(hands)
        out = frame
        if vals["bg_dim"]:
            out = cv2.convertScaleAbs(frame, alpha=1 - vals["bg_dim"] / 100.0)
        now = time.time()
        slots = set(range(len(hands)))
        self.smoother.keep_only(slots)

        cursors, tips_pointing = [], {}
        for slot, pts in enumerate(hands):
            g = gesture(pts)
            tip = self.smoother(slot, pts[8], vals["smooth"])
            if self.mode == "draw":
                cursors.append(self.draw.update(slot, pts, g, tip, vals["brush"], w))
            elif g == "point":
                tips_pointing[slot] = tip
        if self.mode == "draw":
            self.draw.lost(slots)
            self.draw.render(out)
            self.draw.draw_palette(out)
            self.last_view = out.copy()
            for pos, kind, radius in cursors:
                if pos is None:
                    continue
                p = tuple(np.asarray(pos).astype(int))
                col = COLORS[self.draw.color][1]
                if kind == "draw":
                    cv2.circle(out, p, max(radius // 2, 3) + 3, (255, 255, 255), 1, cv2.LINE_AA)
                elif kind == "hover":
                    cv2.circle(out, p, max(radius // 2, 3) + 3, col, 2, cv2.LINE_AA)
                elif kind == "erase":
                    cv2.circle(out, p, radius, (180, 180, 255), 2, cv2.LINE_AA)
                    cv2.putText(out, "erase", (p[0] - 22, p[1] + 5), FONT, 0.5, (180, 180, 255), 1, cv2.LINE_AA)
        else:
            progress = self.ttt.update(tips_pointing, vals["hold_ms"] / 1000.0, bool(vals["vs_cpu"]), w, h, now)
            self.ttt.render(out, bool(vals["vs_cpu"]))
            for slot, tip in tips_pointing.items():
                p = tuple(tip.astype(int))
                cv2.circle(out, p, 8, (255, 255, 255), -1, cv2.LINE_AA)
                if slot in progress:
                    cv2.ellipse(out, p, (22, 22), -90, 0, int(360 * progress[slot]), (0, 255, 0), 4, cv2.LINE_AA)

        if vals["skeleton"]:
            for pts in hands:
                self.tracker.draw_skeleton(out, pts)
        hint = ("1 finger: draw | 2 fingers: move | open palm: erase" if self.mode == "draw"
                else "Point at a square and hold still")
        if not (self.mode == "ttt" and self.ttt.over):
            cv2.putText(out, hint, (10, h - 12), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if not hands:
            cv2.putText(out, "Show your hand to the camera", (10, h - 36), FONT, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        if now < self.toast_until:
            cv2.putText(out, self.toast, (10, h - 60), FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return out
