"""Regions of interest (polygons) stored in normalised [0-1] coordinates so they work at any resolution.

zones.json format:
  [{"name": "back_road", "points": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.4], [0.1, 0.4]]}]
"""
import json
import os

import cv2
import numpy as np


class Zone:
    def __init__(self, name, points):
        self.name = name
        self.points = np.array(points, dtype=np.float32)

    def polygon(self, w, h):
        return (self.points * np.array([w, h], dtype=np.float32)).astype(np.float32)

    def contains(self, x, y, w, h):
        return cv2.pointPolygonTest(self.polygon(w, h).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0

    def to_dict(self):
        return {"name": self.name, "points": self.points.round(4).tolist()}


def full_frame_zone():
    return Zone("full_frame", [[0, 0], [1, 0], [1, 1], [0, 1]])


def load_zones(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [Zone(z["name"], z["points"]) for z in json.load(f)]


def save_zones(zones, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([z.to_dict() for z in zones], f, indent=2)


def draw_zones_interactive(frame, path):
    """Click to draw polygons on a frame.

    Left-click: add point | n: finish current zone (name it in the terminal)
    u: undo last point    | c: clear all          | s: save & continue | q: cancel
    """
    h, w = frame.shape[:2]
    zones, current = [], []
    win = "Draw zones  [click=point  n=finish zone  u=undo  c=clear  s=save  q=cancel]"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([x / w, y / h])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1280), min(h, 720))
    cv2.setMouseCallback(win, on_mouse)
    print("\n[zones] Click points around the area to watch. Press 'n' to finish a zone, 's' to save.")

    while True:
        canvas = frame.copy()
        for z in zones:
            poly = z.polygon(w, h).astype(np.int32)
            cv2.polylines(canvas, [poly], True, (0, 255, 0), 2)
            cv2.putText(canvas, z.name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if current:
            pts = (np.array(current) * [w, h]).astype(np.int32)
            cv2.polylines(canvas, [pts], False, (0, 200, 255), 2)
            for p in pts:
                cv2.circle(canvas, tuple(p), 4, (0, 200, 255), -1)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("n") and len(current) >= 3:
            name = input(f"[zones] Name for zone {len(zones) + 1} (e.g. back_road): ").strip() or f"zone{len(zones) + 1}"
            zones.append(Zone(name.replace(" ", "_"), current.copy()))
            current.clear()
        elif key == ord("u") and current:
            current.pop()
        elif key == ord("c"):
            zones.clear()
            current.clear()
        elif key == ord("s"):
            if len(current) >= 3:
                zones.append(Zone(f"zone{len(zones) + 1}", current.copy()))
            if zones:
                save_zones(zones, path)
                print(f"[zones] Saved {len(zones)} zone(s) to {path}")
            break
        elif key == ord("q"):
            break
    cv2.destroyWindow(win)
    return zones
