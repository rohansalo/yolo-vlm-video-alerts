"""Turns per-frame tracks into higher-level events.

Event types
  zone_enter      object's ground point enters a zone
  zone_exit       object leaves a zone
  moving_in_zone  object is moving inside a zone (fires once per object per zone)
  loiter          object stays (nearly) still inside a zone for >= loiter_s seconds

Motion is measured relative to the object's own box height, so a small far-away car and a
large nearby car moving at the same real speed are both detected (perspective-robust).
"""
import time
from collections import deque


class TrackState:
    def __init__(self, tid, cls, t):
        self.id = tid
        self.cls = cls
        self.first_seen = t
        self.last_seen = t
        self.frames = 0
        self.history = deque(maxlen=90)  # (t, cx, cy_bottom, box_h)
        self.zones = {}  # zone name -> time entered
        self.still_since = {}  # zone name -> time it became stationary
        self.fired = set()  # (event_type, zone) already emitted
        self.speed = 0.0  # box-heights per second
        self.moving = False
        self.conf = 0.0
        self.box = None


class EventEngine:
    def __init__(self, zones, move_thresh=0.4, loiter_s=20.0, min_frames=3, expire_s=2.0):
        self.zones = zones
        self.move_thresh = move_thresh
        self.loiter_s = loiter_s
        self.min_frames = min_frames
        self.expire_s = expire_s
        self.tracks = {}
        self.unique_ids = {}  # cls -> set of ids seen this session

    def _speed(self, st, t, window=1.0):
        pts = [p for p in st.history if t - p[0] <= window]
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < 0.25:
            return 0.0
        (t0, x0, y0, _), (t1, x1, y1, h1) = pts[0], pts[-1]
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return dist / max(h1, 1.0) / (t1 - t0)

    def update(self, detections, w, h, t=None):
        t = t or time.time()
        events = []

        for d in detections:
            x1, y1, x2, y2 = d["box"]
            cx, cy, bh = (x1 + x2) / 2, y2, y2 - y1  # bottom-centre = ground contact point
            st = self.tracks.get(d["id"])
            if st is None:
                st = self.tracks[d["id"]] = TrackState(d["id"], d["cls"], t)
            st.last_seen, st.conf, st.box = t, d["conf"], d["box"]
            st.frames += 1
            st.history.append((t, cx, cy, bh))
            st.speed = self._speed(st, t)
            st.moving = st.speed >= self.move_thresh
            if st.frames < self.min_frames:  # ignore flickering one-frame detections
                continue
            self.unique_ids.setdefault(st.cls, set()).add(st.id)

            for z in self.zones:
                inside = z.contains(cx, cy, w, h)
                was_inside = z.name in st.zones
                if inside and not was_inside:
                    st.zones[z.name] = t
                    events.append(self._event("zone_enter", st, z.name, t))
                elif not inside and was_inside:
                    del st.zones[z.name]
                    st.still_since.pop(z.name, None)
                    st.fired.discard(("moving_in_zone", z.name))
                    st.fired.discard(("loiter", z.name))
                    events.append(self._event("zone_exit", st, z.name, t))

                if not inside:
                    continue
                if st.moving:
                    st.still_since.pop(z.name, None)
                    if ("moving_in_zone", z.name) not in st.fired:
                        st.fired.add(("moving_in_zone", z.name))
                        events.append(self._event("moving_in_zone", st, z.name, t))
                else:
                    since = st.still_since.setdefault(z.name, t)
                    if t - since >= self.loiter_s and ("loiter", z.name) not in st.fired:
                        st.fired.add(("loiter", z.name))
                        ev = self._event("loiter", st, z.name, t)
                        ev["dwell_s"] = round(t - since, 1)
                        events.append(ev)

        for tid in [k for k, s in self.tracks.items() if t - s.last_seen > self.expire_s]:
            del self.tracks[tid]
        return events

    @staticmethod
    def _event(kind, st, zone, t):
        return {
            "type": kind,
            "zone": zone,
            "track_id": st.id,
            "object_class": st.cls,
            "confidence": round(st.conf, 3),
            "speed": round(st.speed, 2),
            "box": [round(v, 1) for v in st.box],
            "t": t,
        }

    def occupancy(self):
        """Current number of objects per zone and class."""
        occ = {z.name: {} for z in self.zones}
        for st in self.tracks.values():
            for zname in st.zones:
                occ[zname][st.cls] = occ[zname].get(st.cls, 0) + 1
        return occ

    def active(self):
        return [s for s in self.tracks.values() if s.frames >= self.min_frames]
