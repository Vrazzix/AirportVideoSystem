"""
Declarative event engine.

Events are described as data (a list of rule dicts) and evaluated per frame by
``EventEngine``. This decouples *what counts as an event* from the detection /
drawing code, so new scenarios (e.g. "door open → boarding") can be added by
appending a config entry instead of writing new branches inside process_frame.

Two rule kinds are supported:

    * ``proximity`` — object of ``class_a`` is within ``margin`` px of an object
      of ``class_b`` (e.g. chock placed at wheel). The "anchor" class drives
      observability: if the anchor is not visible we cannot judge removal.

    * ``presence`` — object of ``class_id`` is present in the frame (e.g. an open
      aircraft door). An optional ``off_class`` provides an explicit "negative"
      state (e.g. door-closed); when it is set, the scene is only observable
      while either class is visible, so the door leaving the frame is not
      mistaken for closing.

Each rule has independent temporal debouncing (hysteresis): a transition is only
confirmed after the raw condition holds for ``on_frames`` / ``off_frames``
consecutive *observable* frames.

The engine is pure logic (no OpenCV / model deps) so it is trivially testable.
"""

import uuid

from .utils import is_box_near_box


def _fmt_dur(sec: float) -> str:
    """Format a duration in seconds as M:SS or H:MM:SS."""
    sec = int(round(max(0.0, sec)))
    m, s = divmod(sec, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ──────────────────────────────────────────────────────────────────────────────
# Hysteresis (shared with the standalone ChockServiceState in detection.py)
# ──────────────────────────────────────────────────────────────────────────────

class _Hysteresis:
    """Confirms a boolean condition after it holds for N consecutive observable frames."""

    def __init__(self, on_frames: int = 5, off_frames: int = 15):
        self.on_frames  = max(1, int(on_frames))
        self.off_frames = max(1, int(off_frames))
        self.active     = False
        self._on_count  = 0
        self._off_count = 0

    def update(self, condition: bool, observable: bool = True) -> bool:
        if not observable:
            self._on_count  = 0
            self._off_count = 0
            return self.active
        if condition:
            self._on_count += 1
            self._off_count = 0
            if not self.active and self._on_count >= self.on_frames:
                self.active = True
        else:
            self._off_count += 1
            self._on_count = 0
            if self.active and self._off_count >= self.off_frames:
                self.active = False
        return self.active


# ──────────────────────────────────────────────────────────────────────────────
# Rule
# ──────────────────────────────────────────────────────────────────────────────

class EventRule:
    """
    A single declarative event rule.

    Config keys (dict):
        id          unique rule id (str)
        kind        'proximity' | 'presence'
        source      detection source key (e.g. 'combo', 'door', 'person')
        on_frames   frames to confirm the ON transition  (default 5)
        off_frames  frames to confirm the OFF transition (default 15)

        # presentation
        on_type / off_type      event 'type' string emitted on transition
        on_label / off_label    human label
        on_icon / off_icon      emoji / icon
        status_on / status_off  status-bar text
        color_on / color_off    BGR tuple/list for the status bar

        # kind == 'proximity'
        class_a, class_b   class ids of the two participants
        margin             max gap in px to count as "near"
        anchor             'a' | 'b' | 'both' — which participant must be visible
                           for the scene to be observable (default 'b')

        # kind == 'presence'
        class_id           class id whose presence is the ON condition
        off_class          optional class id for the explicit OFF state (-1 = none)

        show_timer         if True, append a running ⏱ timer to the status text
                           while active (time since the ON transition)
    """

    def __init__(self, cfg: dict):
        self.id         = cfg["id"]
        self.kind       = cfg.get("kind", "presence")
        self.source     = cfg.get("source", "combo")
        self.on_frames  = int(cfg.get("on_frames", 5))
        self.off_frames = int(cfg.get("off_frames", 15))

        self.on_type    = cfg.get("on_type",  f"{self.id}_on")
        self.off_type   = cfg.get("off_type", f"{self.id}_off")
        self.on_label   = cfg.get("on_label",  self.id)
        self.off_label  = cfg.get("off_label", self.id)
        self.on_icon    = cfg.get("on_icon",  "✅")
        self.off_icon   = cfg.get("off_icon", "⚠️")
        self.status_on  = cfg.get("status_on",  self.on_label)
        self.status_off = cfg.get("status_off", self.off_label)
        self.color_on   = tuple(cfg.get("color_on",  (0, 255, 0)))
        self.color_off  = tuple(cfg.get("color_off", (0, 0, 255)))

        # proximity
        self.class_a = cfg.get("class_a")
        self.class_b = cfg.get("class_b")
        self.margin  = float(cfg.get("margin", 80))
        self.anchor  = cfg.get("anchor", "b")

        # presence
        self.class_id  = cfg.get("class_id")
        self.off_class = int(cfg.get("off_class", -1))

        self.show_timer = bool(cfg.get("show_timer", False))

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, dets: dict) -> tuple:
        """
        Returns (condition_met, observable) for the current frame.

        ``dets`` is {source: {class_id: [ [x1,y1,x2,y2], ... ]}}.
        """
        by_cls = dets.get(self.source, {})

        if self.kind == "proximity":
            a_boxes = by_cls.get(self.class_a, [])
            b_boxes = by_cls.get(self.class_b, [])
            condition = any(
                is_box_near_box(a, b, margin=self.margin)
                for a in a_boxes for b in b_boxes
            )
            if self.anchor == "a":
                observable = len(a_boxes) > 0
            elif self.anchor == "both":
                observable = len(a_boxes) > 0 and len(b_boxes) > 0
            else:  # 'b'
                observable = len(b_boxes) > 0
            return condition, observable

        # presence
        on_boxes  = by_cls.get(self.class_id, [])
        condition = len(on_boxes) > 0
        if self.off_class >= 0:
            off_boxes  = by_cls.get(self.off_class, [])
            observable = condition or len(off_boxes) > 0
        else:
            observable = True
        return condition, observable

    # ── presentation helpers ───────────────────────────────────────────────────

    def type_for(self, phase: str) -> str:
        return self.on_type if phase == "on" else self.off_type

    def label_for(self, phase: str) -> str:
        return self.on_label if phase == "on" else self.off_label

    def icon_for(self, phase: str) -> str:
        return self.on_icon if phase == "on" else self.off_icon

    def status_for(self, active: bool) -> str:
        return self.status_on if active else self.status_off

    def color_for(self, active: bool) -> tuple:
        return self.color_on if active else self.color_off


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

class EventEngine:
    """
    Evaluates a list of EventRule objects each frame and records transitions.

    Typical use (per video/job):
        engine = EventEngine(rules_cfg)
        ...
        engine.update(dets, frame_idx, time_sec)   # call per processed frame
        for ev in engine.drain_events():           # newly fired transitions
            ... attach a thumbnail, store ...
        lines = engine.status_lines()              # for the on-frame status bar
    """

    def __init__(self, rules_cfg: list):
        self.rules     = [EventRule(c) for c in (rules_cfg or []) if c.get("enabled", True)]
        self._state    = {r.id: _Hysteresis(r.on_frames, r.off_frames) for r in self.rules}
        self._active   = {r.id: False for r in self.rules}
        self._on_since = {r.id: None for r in self.rules}   # time_sec of ON transition
        self._last_t   = 0.0
        self._pending  = []

    def update(self, dets: dict, frame_idx: int = 0, time_sec: float = 0.0) -> None:
        self._last_t = time_sec
        for r in self.rules:
            cond, obs = r.evaluate(dets)
            now = self._state[r.id].update(cond, obs)
            if now != self._active[r.id]:
                self._active[r.id] = now
                phase = "on" if now else "off"
                ev = {
                    "id":       uuid.uuid4().hex[:8],
                    "rule":     r.id,
                    "type":     r.type_for(phase),
                    "frame":    frame_idx,
                    "time_sec": round(time_sec, 1),
                    "label":    r.label_for(phase),
                    "icon":     r.icon_for(phase),
                    "thumb":    True,
                }
                if now:
                    self._on_since[r.id] = time_sec
                else:
                    started = self._on_since.get(r.id)
                    if started is not None:
                        ev["duration_sec"] = round(time_sec - started, 1)
                    self._on_since[r.id] = None
                self._pending.append(ev)

    def drain_events(self) -> list:
        """Return and clear the list of transitions fired since the last call."""
        ev, self._pending = self._pending, []
        return ev

    def status_lines(self) -> list:
        """Return [(text, color_bgr), ...] for every rule's current state."""
        lines = []
        for r in self.rules:
            active = self._active[r.id]
            text   = r.status_for(active)
            if active and r.show_timer and self._on_since.get(r.id) is not None:
                text += f"  [{_fmt_dur(self._last_t - self._on_since[r.id])}]"
            lines.append((text, r.color_for(active)))
        return lines
