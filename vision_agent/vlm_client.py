"""Vision-language model client (NVIDIA NIM, OpenAI-compatible) + a background worker queue.

The worker runs VLM calls off the video thread so detection never stalls waiting on the network.
"""
import base64
import queue
import threading
import time

import cv2
import requests

DEFAULT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


class VLMClient:
    def __init__(self, api_key, model=DEFAULT_MODEL, url=DEFAULT_URL, max_side=768, timeout=40):
        self.api_key = api_key
        self.model = model
        self.url = url
        self.max_side = max_side
        self.timeout = timeout

    def _encode(self, frame):
        h, w = frame.shape[:2]
        scale = self.max_side / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf.tobytes()).decode()

    def ask(self, frame, prompt, max_tokens=160):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode(frame)}"}},
                    ],
                }
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        r = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"VLM API {r.status_code}: {r.text[:200]}")
        content = r.json()["choices"][0]["message"].get("content")
        return (content or "").strip()


class VLMWorker:
    """Single background thread processing VLM jobs from a bounded queue.

    on_result(job, text, error, latency_s) is called from the worker thread.
    If the queue is full the job is rejected (submit returns False) so the caller can fall back.
    """

    def __init__(self, client, on_result, max_queue=8):
        self.client = client
        self.on_result = on_result
        self.q = queue.Queue(maxsize=max_queue)
        self.busy = False
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame, prompt, **meta):
        try:
            self.q.put_nowait({"frame": frame, "prompt": prompt, **meta})
            return True
        except queue.Full:
            return False

    def pending(self):
        return self.q.qsize() + (1 if self.busy else 0)

    def _run(self):
        while True:
            job = self.q.get()
            self.busy = True
            started = time.time()
            text, err = None, None
            try:
                text = self.client.ask(job["frame"], job["prompt"])
            except Exception as e:  # network/API errors must never kill the worker
                err = str(e)
            self.busy = False
            try:
                self.on_result(job, text, err, time.time() - started)
            except Exception as e:
                print(f"[vlm] result handler error: {e}")
