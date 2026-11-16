# Realtime Vision Alerts — YOLO tracking + NVIDIA VLM descriptions

Watch a webcam, IP camera, video file or YouTube live stream. A local YOLO model detects and tracks
people and vehicles, zone/motion rules turn that into events, and an NVIDIA vision-language model
(NIM) describes each alert in plain English. Everything is stored in SQLite, shown on a live
dashboard, and can be pushed to n8n for email/WhatsApp notifications.

```
video source ─► capture ─► YOLO + ByteTrack ─► zone / motion events ─► SQLite
 (webcam, RTSP,   newest     (local, free)      zone_enter · zone_exit     │
  file, YouTube)  frame                         moving_in_zone · loiter    ▼
                                                        │ alert?       dashboard :8050
                                                        ▼                (+ Ask box)
                                         snapshot ─► NVIDIA VLM ─► console · beep · WebSocket · webhook (n8n)
                                                     "describe it"
```

YOLO decides **whether** something happened; the VLM only explains **what** happened. That keeps
it fast, cheap on API credits, and far less prone to hallucinated alerts than asking a VLM about
every frame.

## Setup

```powershell
git clone https://github.com/rohansalo/yolo-vlm-video-alerts.git
cd yolo-vlm-video-alerts
python -m venv venv
.\venv\Scripts\Activate.ps1          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+.

Get an API key at [build.nvidia.com](https://build.nvidia.com) (starts with `nvapi-`) and check it:

```powershell
$env:NVIDIA_API_KEY = "nvapi-..."    # macOS/Linux: export NVIDIA_API_KEY=nvapi-...
python test_api.py
```

The YOLO weights (`yolo11n.pt`, 5 MB) download automatically on first run.

## Run

Two terminals, both with the venv active, from this folder:

```powershell
# Terminal 1 — the agent
python -m vision_agent.run --source 0 --draw-zones

# Terminal 2 — the dashboard, then open http://localhost:8050
python -m vision_agent.dashboard
```

Press `q` in the video window (or Ctrl+C) to stop the agent.

### Sources

| Source | `--source` |
|---|---|
| Webcam | `0` |
| IP / CCTV camera | `rtsp://user:pass@192.168.1.50:554/stream1` |
| Phone camera app | `http://192.168.1.23:8080/video` |
| Video file | `C:\videos\street.mp4` (add `--loop`) |
| YouTube / Twitch (live or not) | `"https://www.youtube.com/watch?v=..."` — quote it |

Web video links are resolved with `yt-dlp`; expired stream URLs are refreshed automatically on reconnect.
Use `--max-height 480` to reduce bandwidth.

### Zones

`--draw-zones` opens the first frame: click points around an area, press `n` and type a name **in the
terminal**, then `s` to save. Zones are stored in `zones.json` and reused next run. An object is
"in" a zone when the bottom-centre of its box (its feet) is inside, so draw zones on the ground.
Without zones the whole frame is watched.

### Events and alerts

| Event | Fires when | Alert by default |
|---|---|---|
| `zone_enter` / `zone_exit` | an object enters / leaves a zone | no |
| `moving_in_zone` | an object moves inside a zone (once per object per zone) | yes |
| `loiter` | an object stays still in a zone ≥ `--loiter-s` seconds (20) | yes |
| `question_yes` | the `--ask` answer turns from no to yes | yes, with `--ask` |

`--alert-on zone_enter,loiter` picks which events alert; `--cooldown 120` allows at most one alert
per event type and zone every 2 minutes.

### Asking questions

- **Dashboard Ask box** — type any question; the agent answers it about the current live frame and
  saves the frame it looked at.
- **`--ask "Is anyone holding a phone?"`** — a yes/no question checked every `--ask-every` seconds
  (10); alerts when the answer turns to yes. `--ask-when-objects` only checks while YOLO sees
  something. Each check is one API call.

### Common options

| Option | Default | Purpose |
|---|---|---|
| `--classes` | `person,car,truck,bus,motorcycle,bicycle` | YOLO classes to track (`""` = all 80) |
| `--conf` | `0.35` | detection confidence threshold |
| `--summary-every` | off | VLM scene summary every N seconds |
| `--webhook` | off | POST every alert (JSON + base64 snapshot) to this URL |
| `--ws-port` | `5433` | WebSocket feed of events/alerts (`0` = off) |
| `--model` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | VLM model |
| `--url` | NVIDIA hosted endpoint | any OpenAI-compatible chat completions URL (e.g. a local NIM) |
| `--db` / `--snapshots` | `vision_agent.db` / `snapshots/` | where results are stored |
| `--no-display` | | run headless |

Run `python -m vision_agent.run --help` for the full list.

## Email / WhatsApp alerts with n8n

1. In n8n, create a **Webhook** node (POST, path `vision-alert`), then add a **Gmail**, **Send Email**
   or **Twilio** (WhatsApp) node after it.
2. Use fields from the request body in the message, e.g.
   `{{ $json.body.type }} – {{ $json.body.object_class }} in {{ $json.body.zone }}` and
   `{{ $json.body.description }}`.
3. Publish the workflow and pass its **Production URL** (`/webhook/`, not `/webhook-test/`):

```powershell
python -m vision_agent.run --source 0 --webhook "https://YOUR-N8N/webhook/vision-alert" --cooldown 120
```

Webhook payload fields: `ts`, `type`, `zone`, `object_class`, `track_id`, `confidence`, `speed`, `box`,
`description`, `vlm_latency_s`, `event_id`, `session_id`, `snapshot`, `image_base64`
(plus `question` for `question_yes`).

## Data

`vision_agent.db` (SQLite) holds:

| Table | Contents |
|---|---|
| `sessions` | one row per run: source, model, zones, start/end, frames, avg fps |
| `events` | every event: time, type, zone, object, confidence, speed, alert flag, snapshot path, VLM description |
| `questions` | dashboard questions and `--ask` checks with answers |
| `summaries` | periodic scene summaries |

Open it with the dashboard, [DB Browser for SQLite](https://sqlitebrowser.org), or any SQLite client.

## Project layout

```
vision_agent/
  run.py               entry point — wires everything together
  capture.py           threaded video reader (files, webcams, streams, yt-dlp links)
  detector.py          YOLO + ByteTrack tracking
  zones.py             zone polygons + interactive drawing
  events.py            tracks → zone / motion / loiter events
  vlm_client.py        NVIDIA VLM client + background worker
  notifier.py          console, beep, WebSocket and webhook alerts with cooldown
  storage.py           SQLite schema and writes
  websocket_server.py  live event feed
  dashboard.py/.html   web dashboard
test_api.py            checks your API key against the VLM models
```

## Credits & license

Started from the `vlm_alerts` example in NVIDIA's
[metropolis-nim-workflows](https://github.com/NVIDIA/metropolis-nim-workflows) (Apache-2.0) and rebuilt
around YOLO tracking, zones, a dashboard and webhook alerts. Released under the Apache License 2.0 —
see [LICENSE](LICENSE). Object detection by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
(AGPL-3.0); vision-language models served by [NVIDIA NIM](https://build.nvidia.com).
