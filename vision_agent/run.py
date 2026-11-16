"""Entry point: live video -> YOLO tracking -> zone/motion events -> VLM descriptions -> SQLite + alerts.

Run from the vlm_alerts folder:
    python -m vision_agent.run --source 0 --draw-zones                 # webcam, draw zones first
    python -m vision_agent.run --source "https://www.youtube.com/watch?v=..." --summary-every 60
    python -m vision_agent.run --source rtsp://user:pass@cam/stream --no-display
    python -m vision_agent.run --source my_video.mp4 --loop --ask "Is anyone running?"

API key: --api-key nvapi-... or set NVIDIA_API_KEY in the environment.
"""
import argparse
import os
import sys
import time

import cv2

from .capture import VideoSource
from .detector import Detector
from .events import EventEngine
from .notifier import Notifier
from .storage import Store, now
from .vlm_client import DEFAULT_MODEL, DEFAULT_URL, VLMClient, VLMWorker
from .websocket_server import WebSocketServer
from .zones import draw_zones_interactive, full_frame_zone, load_zones

EVENT_TYPES = ("zone_enter", "zone_exit", "moving_in_zone", "loiter")


def parse_args():
    p = argparse.ArgumentParser(description="Real-time vision agent (YOLO + zones + NVIDIA VLM)")
    p.add_argument("--source", required=True,
                   help="Video file, webcam index (0), rtsp/http stream URL, or YouTube/Twitch link")
    p.add_argument("--loop", action="store_true", help="Loop a video file")
    p.add_argument("--max-height", type=int, default=720,
                   help="Max stream resolution for YouTube/Twitch links (lower = less bandwidth, faster)")
    p.add_argument("--api-key", default=os.environ.get("NVIDIA_API_KEY"), help="NVIDIA API key (or env NVIDIA_API_KEY)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="VLM model name")
    p.add_argument("--url", default=DEFAULT_URL, help="OpenAI-compatible chat completions URL")
    p.add_argument("--weights", default="yolo11n.pt", help="YOLO weights (auto-downloaded)")
    p.add_argument("--classes", default="person,car,truck,bus,motorcycle,bicycle",
                   help="Comma-separated YOLO classes to track ('' = all)")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument("--zones", default="zones.json", help="Zones file (normalised polygons)")
    p.add_argument("--draw-zones", action="store_true", help="Draw zones on the first frame before starting")
    p.add_argument("--alert-on", default="moving_in_zone,loiter", help=f"Event types that raise alerts: {EVENT_TYPES}")
    p.add_argument("--move-thresh", type=float, default=0.4, help="Speed (box-heights/sec) counted as moving")
    p.add_argument("--loiter-s", type=float, default=20.0, help="Seconds still inside a zone before 'loiter'")
    p.add_argument("--cooldown", type=float, default=30.0, help="Seconds between alerts of same type+zone")
    p.add_argument("--summary-every", type=float, default=0.0, help="Ask the VLM for a scene summary every N sec (0=off)")
    p.add_argument("--ask", default=None,
                   help='Yes/no question checked continuously; alerts when the answer turns to yes, '
                        'e.g. --ask "Is anyone holding a phone?"')
    p.add_argument("--ask-every", type=float, default=10.0, help="Seconds between --ask checks")
    p.add_argument("--ask-when-objects", action="store_true",
                   help="Only run --ask checks while YOLO is tracking something (saves API credits)")
    p.add_argument("--webhook", default=None, help="POST alerts here (e.g. n8n / Zapier webhook)")
    p.add_argument("--ws-port", type=int, default=5433, help="WebSocket port for live events (0=off)")
    p.add_argument("--db", default="vision_agent.db", help="SQLite database path")
    p.add_argument("--snapshots", default="snapshots", help="Folder for alert snapshots")
    p.add_argument("--no-display", action="store_true", help="Run headless (no video window)")
    p.add_argument("--no-beep", action="store_true", help="Disable the alert beep")
    return p.parse_args()


SITUATION = {
    "zone_enter": "It has just entered the area being watched.",
    "zone_exit": "It has just left the area being watched.",
    "moving_in_zone": "It is moving.",
    "loiter": "It has stayed in roughly the same place for about {dwell} seconds.",
}


def mark_object(frame, box):
    """Copy of the frame with a red outline around the object (visual prompt for the VLM)."""
    marked = frame.copy()
    h, w = marked.shape[:2]
    pad = max(4, w // 200)
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(marked, (max(0, x1 - pad), max(0, y1 - pad)), (min(w - 1, x2 + pad), min(h - 1, y2 + pad)),
                  (0, 0, 255), max(2, w // 400))
    return marked


def event_prompt(ev, w, h):
    """Grounded prompt: plain facts from the CV model (class, position, behaviour), no internal jargon."""
    x1, y1, x2, y2 = ev["box"]
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    horiz = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"
    vert = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    where = "centre" if (horiz, vert) == ("centre", "middle") else f"{vert} {horiz}".replace("middle ", "")
    situation = SITUATION[ev["type"]].format(dwell=int(ev.get("dwell_s", 20)))
    obj = ev["object_class"]
    return (
        f"The {obj} outlined in red, in the {where} of the image, is being monitored. {situation} "
        f"In one short sentence, describe what this {obj} is doing and anything notable about it, such as its "
        "appearance or what it is carrying or next to. Describe only what you can see. "
        "Do not mention the red outline."
    )


def draw(frame, zones, tracks, alert_zones, status, banner):
    h, w = frame.shape[:2]
    for z in zones:
        color = (0, 0, 255) if z.name in alert_zones else (0, 200, 0)
        poly = z.polygon(w, h).astype("int32")
        cv2.polylines(frame, [poly], True, color, 2)
        cv2.putText(frame, z.name, tuple(int(v) for v in poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for st in tracks:
        x1, y1, x2, y2 = (int(v) for v in st.box)
        color = (0, 165, 255) if st.moving else (255, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{st.cls} #{st.id}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.rectangle(frame, (0, 0), (w, 28), (30, 30, 30), -1)
    cv2.putText(frame, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if banner:
        cv2.rectangle(frame, (0, h - 34), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, banner[: int(w / 9)], (8, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def main():
    args = parse_args()
    if not args.api_key:
        sys.exit("No API key. Pass --api-key nvapi-... or set NVIDIA_API_KEY.")
    alert_on = {s.strip() for s in args.alert_on.split(",") if s.strip()}
    bad = alert_on - set(EVENT_TYPES)
    if bad:
        sys.exit(f"Unknown --alert-on types {sorted(bad)}. Valid: {EVENT_TYPES}")

    try:
        source = VideoSource(args.source, loop=args.loop, max_height=args.max_height)
    except RuntimeError as e:
        sys.exit(f"[run] {e}")
    print(f"[run] source={args.source} live={source.is_live} fps={source.fps:.1f}")
    source_label = f"{args.source}  ({source.title})" if source.title else args.source

    # zones
    zones = load_zones(args.zones)
    if args.draw_zones:
        frame = None
        while frame is None and not source.ended:
            _, frame = source.read()
            time.sleep(0.05)
        if frame is not None:
            zones = draw_zones_interactive(frame, args.zones) or zones
    if not zones:
        zones = [full_frame_zone()]
        print("[run] no zones defined -> watching the full frame (use --draw-zones to define areas)")
    print(f"[run] zones: {[z.name for z in zones]}")

    classes = [c for c in args.classes.split(",") if c.strip()] or None
    detector = Detector(args.weights, classes=classes, conf=args.conf)
    engine = EventEngine(zones, move_thresh=args.move_thresh, loiter_s=args.loiter_s)
    store = Store(args.db)
    sid = store.start_session(source_label, args.model, zones)
    os.makedirs(args.snapshots, exist_ok=True)

    ws = None
    if args.ws_port:
        ws = WebSocketServer(port=args.ws_port)
        ws.run()
        print(f"[run] websocket events on ws://localhost:{args.ws_port}")
    notifier = Notifier(ws, webhook_url=args.webhook, cooldown_s=args.cooldown, beep=not args.no_beep)

    ui = {"banner": "", "banner_until": 0.0, "alert_zones": {}, "watch_yes": False, "watch_answer": ""}
    if args.ask:
        print(f'[run] watching for: "{args.ask}" every {args.ask_every:g}s')

    def snapshot(frame, name):
        path = os.path.join(args.snapshots, f"{sid}_{name}.jpg")
        cv2.imwrite(path, frame)
        return path

    def on_vlm_result(job, text, err, latency):
        desc = text if text else f"(VLM error: {err})"
        lat = round(latency, 2)
        if job["kind"] == "event":
            store.set_event_description(job["event_id"], desc, lat)
            payload = dict(job["payload"], description=desc, vlm_latency_s=lat)
            notifier.alert(payload, job["frame"])
            ui["banner"] = f"ALERT {payload['type']} @ {payload['zone']}: {desc}"
            ui["banner_until"] = time.time() + 8
        elif job["kind"] == "ask":  # one-off question from the dashboard
            store.answer_question(job["qid"], desc, lat, job["snap"], ok=text is not None)
            print(f"[ask] {job['question']}\n      -> {desc}")
            notifier.broadcast({"kind": "answer", "ts": now(), "question": job["question"], "answer": desc})
        elif job["kind"] == "watch":  # continuous --ask yes/no check
            store.answer_question(job["qid"], desc, lat, ok=text is not None)
            if text is None:
                print(f"[watch] {desc}")
                return
            is_yes = text.strip().lower().lstrip("\"'*` ").startswith("yes")
            ui["watch_answer"] = "YES" if is_yes else "no"
            ev = {"type": "question_yes", "zone": "full_frame", "track_id": None, "object_class": "scene",
                  "confidence": None, "speed": None}
            if is_yes and not ui["watch_yes"] and notifier.should_alert(ev):  # fire on no -> yes
                snap = snapshot(job["frame"], f"ask_{job['qid']}")
                event_id = store.add_event(sid, ev, True, snap)
                description = f'"{args.ask}" -> {text.strip()}'
                store.set_event_description(event_id, description, lat)
                payload = dict(ev, ts=now(), event_id=event_id, session_id=sid, snapshot=snap,
                               question=args.ask, description=description, vlm_latency_s=lat)
                notifier.alert(payload, job["frame"])
                ui["banner"] = f"ALERT: {args.ask} -> YES"
                ui["banner_until"] = time.time() + 8
            ui["watch_yes"] = is_yes
        else:  # summary
            store.add_summary(sid, job["counts"], desc, round(latency, 2))
            print(f"[summary] {job['counts']} -> {desc}")
            notifier.broadcast({"kind": "summary", "ts": now(), "counts": job["counts"], "description": desc})

    worker = VLMWorker(VLMClient(args.api_key, model=args.model, url=args.url), on_vlm_result)

    last_id, frames, t_start, last_summary = -1, 0, time.time(), time.time()
    last_poll, last_watch, last_progress = 0.0, 0.0, time.time()
    watch_prompt = None
    if args.ask:
        watch_prompt = args.ask if "yes or no" in args.ask.lower() else f"{args.ask} Answer only yes or no."
    fps_t, fps_n, fps = time.time(), 0, 0.0
    print("[run] running - press 'q' in the video window (or Ctrl+C) to stop")
    try:
        while True:
            fid, frame = source.read()
            if frame is None or fid == last_id:
                if source.ended:
                    print("[run] video ended")
                    break
                time.sleep(0.005)
                continue
            last_id = fid
            frames += 1
            h, w = frame.shape[:2]
            t = time.time()

            events = engine.update(detector.track(frame), w, h, t)
            for ev in events:
                is_alert = ev["type"] in alert_on and notifier.should_alert(ev)
                snap = marked = None
                if is_alert:
                    # visual prompt: outline the object so the VLM (and the snapshot viewer) know which one
                    marked = mark_object(frame, ev["box"])
                    snap = os.path.join(args.snapshots, f"{sid}_{ev['type']}_{ev['zone']}_{ev['track_id']}_{fid}.jpg")
                    cv2.imwrite(snap, marked)
                event_id = store.add_event(sid, ev, is_alert, snap)
                payload = {k: v for k, v in ev.items() if k != "t"}
                payload.update(ts=now(), event_id=event_id, session_id=sid, snapshot=snap)
                notifier.broadcast({"kind": "event", **payload})
                if is_alert:
                    ui["alert_zones"][ev["zone"]] = t + 5
                    job = dict(kind="event", event_id=event_id, payload=payload)
                    if not worker.submit(marked, event_prompt(ev, w, h), **job):
                        notifier.alert(dict(payload, description="(VLM busy - no description)"), marked)

            # questions typed on the dashboard (answered about the current frame)
            if t - last_poll >= 0.5:
                last_poll = t
                for qid, question in store.claim_pending_questions(sid):
                    snap = snapshot(frame, f"q{qid}")
                    prompt = f"{question}\nAnswer based only on what is visible in this image. Be concise."
                    if not worker.submit(frame.copy(), prompt, kind="ask", qid=qid, question=question, snap=snap):
                        store.release_question(qid)  # VLM queue full - try again shortly

            # continuous yes/no question (--ask)
            if (watch_prompt and t - last_watch >= args.ask_every and worker.pending() == 0
                    and (not args.ask_when_objects or engine.active())):
                last_watch = t
                qid = store.add_question(sid, args.ask, kind="watch", status="answering")
                worker.submit(frame.copy(), watch_prompt, kind="watch", qid=qid)

            if args.summary_every and t - last_summary >= args.summary_every and worker.pending() == 0:
                last_summary = t
                counts = {c: len(ids) for c, ids in engine.unique_ids.items()}
                worker.submit(frame.copy(), "Describe what is happening in this scene in one or two sentences.",
                              kind="summary", counts=counts)

            fps_n += 1
            if t - fps_t >= 1.0:
                fps, fps_t, fps_n = fps_n / (t - fps_t), t, 0
            if t - last_progress >= 5.0:  # live frames/FPS for the dashboard
                last_progress = t
                store.update_progress(sid, frames, round(frames / max(t - t_start, 1e-6), 2))

            if not args.no_display:
                ui["alert_zones"] = {z: u for z, u in ui["alert_zones"].items() if u > t}
                occ = engine.occupancy()
                occ_txt = "  ".join(f"{z}:{sum(c.values())}" for z, c in occ.items())
                status = f"{fps:4.1f} fps | tracks {len(engine.active())} | {occ_txt} | VLM queue {worker.pending()}"
                if args.ask and ui["watch_answer"]:
                    status += f" | ask: {ui['watch_answer']}"
                banner = ui["banner"] if t < ui["banner_until"] else ""
                cv2.imshow("Vision Agent", draw(frame, zones, engine.active(), ui["alert_zones"], status, banner))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = max(time.time() - t_start, 1e-6)
        store.end_session(sid, frames, round(frames / elapsed, 2))
        source.release()
        cv2.destroyAllWindows()
        print(f"[run] session {sid}: {frames} frames, {frames / elapsed:.1f} fps avg -> {args.db}")


if __name__ == "__main__":
    main()
