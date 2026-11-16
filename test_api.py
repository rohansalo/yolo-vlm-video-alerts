"""Quick check that your NVIDIA API key works with the vision models the agent uses.

Usage (from the vlm_alerts folder, venv active):
    python test_api.py                      # uses NVIDIA_API_KEY and a built-in sample image
    python test_api.py nvapi-YOURKEY
    python test_api.py nvapi-YOURKEY path/to/image.jpg
"""
import base64
import os
import sys
import time

import requests

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODELS = [
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",  # agent default
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
]

key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NVIDIA_API_KEY")
if not key:
    sys.exit("No API key. Usage: python test_api.py nvapi-YOURKEY  (or set NVIDIA_API_KEY)")

if len(sys.argv) > 2:
    image_path = sys.argv[2]
else:
    from ultralytics.utils import ASSETS  # sample photo (a bus with people) bundled with ultralytics

    image_path = str(ASSETS / "bus.jpg")
with open(image_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
print(f"Image: {image_path}\n")

headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
for model in MODELS:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in one sentence."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 200,
        "temperature": 0.2,
        "stream": False,
    }
    t = time.time()
    try:
        r = requests.post(URL, headers=headers, json=payload, timeout=90)
        secs = time.time() - t
        if r.status_code == 200:
            answer = r.json()["choices"][0]["message"].get("content")
            print(f"[OK  {secs:4.1f}s] {model}\n    {answer!r}\n")
        else:
            print(f"[{r.status_code}] {model}: {r.text[:200]}\n")
    except Exception as e:
        print(f"[ERR] {model}: {e}\n")
