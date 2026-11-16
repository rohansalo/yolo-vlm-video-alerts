# YOLO-VLM Video Alerts — real-time detection + NVIDIA vision-language descriptions

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
every frame. See [How it works](#how-it-works-yolo--vision-language-models) for the technical details.

## Screenshots

All screenshots are from real runs on a public YouTube live cam (Fiserv Forum, Milwaukee) with the
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` VLM.

**Live dashboard:** the Ask box, KPIs, event timeline, breakdowns by type/class/zone, the latest
VLM scene summary, and alerts with their snapshot and the VLM's description.

![Live dashboard session: Ask box with three answered questions, 11 events, 3 alerts, 1.7 s average VLM latency, an events-per-10s timeline, a twilight scene summary of the Fiserv Forum, and alert cards such as "A white car is driving down the street."](docs/images/dashboard.png)

**Ask about the live scene:** natural-language questions answered by the VLM about the current
frame in under 2 seconds, with the exact frame it looked at. Answers are not always right: here
it reads the "Fiserv Forum" sign as "Tisery forum" (see [Limitations](#limitations)).

![Ask box: "how many people are walking around" answered "2", "does anybody moving cardboard boxes" answered "No", and the shop-name question answered "Tisery forum"](docs/images/ask.png)

**Visual prompting:** for each alert the CV model's box is drawn on the frame in red before it is
sent to the VLM, so the description is about the right object. For this frame the VLM returned
*"The car is driving down the street, and there is a person standing on the sidewalk next to it."*

![Twilight street scene in front of the Fiserv Forum with one car outlined by a red box](docs/images/visual-prompt.jpg)

**What the CV model sees:** YOLO11n detections and a drawn zone, rendered with the agent's own
overlay on a frame from the same stream.

![Street scene with bounding boxes around cars and a bus and a green polygon zone labelled street](docs/images/detection.jpg)

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

Press `q` in the video window (or Ctrl+C) to stop the agent. The dashboard follows the newest
session; pick an older one from the Session menu or open `http://localhost:8050/?session=<id>`.

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

How frames are read depends on the source. Webcams and RTSP cameras push frames in real time, so the
reader always hands over the newest frame (lowest latency). YouTube/Twitch/HLS streams arrive as
multi-second segments in bursts, so the reader paces them out at the stream's own frame rate, like a
video player, a few seconds behind live. Without this, the agent would only ever see the last frame
of each burst (about 1 FPS); with it, a 720p YouTube live stream is processed at ~18 FPS on a laptop
CPU (headless).

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

## How it works: YOLO + vision-language models

### What is a vision-language model?

A vision-language model (VLM) is a large language model that can also *see*. Most VLMs share the
same three-part design:

```
image  ──► vision encoder (ViT) ──► projector ──► visual tokens ─┐
                                                                  ├─► language model ──► "A person in a red
prompt ──► tokenizer ─────────────────────────────► text tokens ─┘     (decoder)          jacket is waiting…"
```

1. **Vision encoder** — a Vision Transformer cuts the image into small patches and turns each patch
   into an embedding vector that captures what is in it.
2. **Projector** — a small network maps those image embeddings into the same vector space as the
   language model's word embeddings, so the picture becomes a sequence of "visual tokens".
3. **Language model** — reads the visual tokens and your text prompt together and generates the
   answer one token at a time, exactly like a chatbot.

Because it is trained on huge sets of image–text pairs and instructions, a VLM can describe scenes,
answer open-ended questions, read text in the image and reason about relationships ("a person
*holding* a phone", "the door is *open*") — things a fixed-class detector cannot express.

That flexibility has costs, which is why this project does not use a VLM alone:

| | YOLO (local detector) | VLM (NVIDIA NIM) |
|---|---|---|
| Output | boxes + 80 fixed classes, track IDs | free-form text |
| Speed | real-time on a laptop CPU (~18 FPS tracking a 720p live stream, ~50 ms per frame) | ~1–5 s per request over the network |
| Cost | free after download | one API request per image |
| Understands | *what* and *where* | *what is happening* and *why it matters* |
| Typical failure | missed or flickering boxes | confident but wrong descriptions (hallucination) |

So the two are combined: **YOLO runs on every frame and decides when something happened; the VLM is
called only at those moments to explain it.**

### The model

The default model is `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, called through NVIDIA NIM's
hosted, OpenAI-compatible endpoint. As its name indicates, it has 30B parameters in total, of which
about 3B are active per token (a mixture-of-experts design), which keeps responses fast for its
size. It can produce a "reasoning" trace before answering; this project switches that off (see below).
Any OpenAI-compatible VLM can be used with `--model` and `--url`, including a self-hosted NIM
container, in which case frames never leave your network.

### How the project uses the VLM

**1. When it is called (gating).** Never per frame — only when:

| Trigger | Prompt sent with the frame |
|---|---|
| An alert event passes its cooldown | `The car outlined in red, in the bottom left of the image, is being monitored. It has stayed in roughly the same place for about 23 seconds. In one short sentence, describe what this car is doing and anything notable about it, such as its appearance or what it is carrying or next to. Describe only what you can see. Do not mention the red outline.` |
| `--ask` timer (every `--ask-every` s) | `<your question> Answer only yes or no.` |
| Dashboard Ask box | `<your question>` + `Answer based only on what is visible in this image. Be concise.` |
| `--summary-every` timer | `Describe what is happening in this scene in one or two sentences.` |

The event prompt is **grounded** in the CV model's output in two ways:

- **Visual prompting.** The agent draws a red box around the detected object on the frame it sends,
  so in a busy scene the VLM knows exactly *which* car or person to describe.
- **Plain-language facts.** The text names the object class, where it is in the image (e.g. "bottom
  left") and what YOLO measured ("it has stayed in roughly the same place for about 23 seconds"),
  rather than internal names like `loiter` or zone IDs that a model would just repeat back. The VLM
  adds what only it can see: what the object is doing, what it looks like, what it is carrying or near.

**2. Preparing the image.** The exact frame that triggered the event is copied with the object
outlined, saved as the alert snapshot (so the description always matches the picture in the
dashboard and email), downscaled so its longest side is at most 768 px, JPEG-encoded at quality 80 and
sent inline as a base64 `data:` URL — typically 50–150 KB per request.

**3. The request.** A standard chat-completions call with a text part and an `image_url` part:

```json
{
  "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
  "messages": [{ "role": "user", "content": [
      { "type": "text", "text": "The car outlined in red, in the bottom left of the image, ..." },
      { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,/9j/4AAQ..." } }
  ]}],
  "chat_template_kwargs": { "enable_thinking": false },
  "max_tokens": 160,
  "temperature": 0.2
}
```

- `enable_thinking: false` turns off the model's reasoning trace, so it answers directly in about a
  second or two instead of spending its token budget "thinking" and getting cut off.
- `temperature: 0.2` keeps answers factual and consistent rather than creative.
- `max_tokens: 160` is plenty for one or two sentences and bounds latency and cost.

**4. Interpreting the answer.**

- *Event descriptions, summaries and Ask-box answers* are stored as text exactly as returned, together
  with the request latency (`vlm_latency_s`).
- *`--ask` answers* are parsed: quotes and markdown are stripped and the answer counts as "yes" only
  if it starts with "yes"; anything else is "no". Alerts are **edge-triggered** — one fires only when
  the answer flips from no to yes (and the cooldown allows it), so a scene that stays "yes" doesn't
  flood your inbox.

**5. Never blocking the video.** All VLM calls run on a background worker thread fed by a bounded
queue (8 jobs). Detection and tracking keep running at full speed while a request is in flight. If
the queue is full, an event alert is still sent immediately (marked "VLM busy - no description") and
a dashboard question is simply retried a moment later. Network errors and timeouts (40 s) are
recorded as the description text rather than crashing the agent.

### Limitations

- **One still frame per request.** The VLM sees a snapshot, not the video; motion and timing come only
  from YOLO's events (e.g. "loiter" means YOLO measured 20 s without movement).
- **It can be wrong.** Descriptions are helpful context, not ground truth. For zone events the alert
  decision is made by YOLO; for `--ask` the decision is the VLM's, so phrase questions about clearly
  visible things.
- **Privacy.** With the hosted endpoint, frames are sent to NVIDIA's cloud. Use `--url` with a local
  NIM to keep them on your own hardware.
- **Cost scales with use.** Each alert, question, summary and `--ask` check is one request
  (`--ask-every 10` ≈ 360 requests per hour). Cooldowns, zones and `--ask-when-objects` keep this down.

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
