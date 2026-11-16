"""Alert fan-out: console, audio beep, WebSocket broadcast and an optional webhook (e.g. n8n).

A per-(event type, zone) cooldown prevents notification floods on busy scenes.
"""
import base64
import threading
import time

import cv2
import requests

try:
    import winsound
except ImportError:  # non-Windows
    winsound = None


class Notifier:
    def __init__(self, ws_server=None, webhook_url=None, cooldown_s=30.0, beep=True, send_image=True):
        self.ws = ws_server
        self.webhook_url = webhook_url
        self.cooldown_s = cooldown_s
        self.beep = beep and winsound is not None
        self.send_image = send_image
        self._last = {}

    def broadcast(self, message):
        """Push any JSON-serialisable message to connected WebSocket clients (dashboards)."""
        if not self.ws:
            return
        try:
            self.ws(message)
        except Exception as e:
            print(f"[ws] broadcast failed: {e}")

    def should_alert(self, ev):
        key = (ev["type"], ev["zone"])
        t = time.time()
        if t - self._last.get(key, 0) < self.cooldown_s:
            return False
        self._last[key] = t
        return True

    def alert(self, payload, snapshot=None):
        who = f"{payload['object_class']} #{payload['track_id']}" if payload.get("track_id") is not None else ""
        print(
            f"\033[91m[ALERT] {payload['ts']}  {payload['type']}  zone={payload['zone']}  "
            f"{who}  -> {payload.get('description') or ''}\033[0m"
        )
        if self.beep:
            threading.Thread(target=lambda: winsound.Beep(1000, 500), daemon=True).start()
        self.broadcast({"kind": "alert", **payload})
        if self.webhook_url:
            body = dict(payload)
            if self.send_image and snapshot is not None:
                ok, buf = cv2.imencode(".jpg", snapshot, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    body["image_base64"] = base64.b64encode(buf.tobytes()).decode()
            threading.Thread(target=self._post, args=(body,), daemon=True).start()

    def _post(self, body):
        try:
            r = requests.post(self.webhook_url, json=body, timeout=10)
            if r.status_code >= 300:
                print(f"[webhook] {r.status_code}: {r.text[:150]}")
        except Exception as e:
            print(f"[webhook] failed: {e}")
