"""Threaded video capture for files, webcams, RTSP/HTTP streams and YouTube/Twitch pages.

Live sources: a background thread keeps only the newest frame, so slow processing never
builds up lag (you always analyse "now", not 10 seconds ago). Dropped connections are retried.
Files: frames are paced at the file's native FPS so playback looks real-time.
Web video pages (YouTube, Twitch, ...): resolved to a direct stream URL with yt-dlp. The
resolved URL expires after a few hours, so reconnects resolve the page again.
"""
import threading
import time
from urllib.parse import urlparse

import cv2

# Pages that are not raw video and must be resolved by yt-dlp first
RESOLVE_HOSTS = ("youtube.com", "youtu.be", "twitch.tv", "vimeo.com", "dailymotion.com", "kick.com")


def needs_resolving(source):
    s = str(source)
    if not s.lower().startswith(("http://", "https://")):
        return False
    host = (urlparse(s).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in RESOLVE_HOSTS)


def resolve_stream(page_url, max_height=720):
    """Return (direct_url, is_live, title) for a YouTube/Twitch/... page using yt-dlp."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is needed for YouTube/Twitch links: pip install yt-dlp")
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # one stream with video, capped resolution. Audio isn't needed, so video-only streams are fine:
        # live -> combined HLS feed; regular videos -> video-only H.264 (most reliable for OpenCV)
        "format": f"best[height<={max_height}][vcodec!=none][protocol^=m3u8]/"
                  f"bestvideo[height<={max_height}][vcodec^=avc1]/bestvideo[height<={max_height}]/"
                  f"best[height<={max_height}][vcodec!=none]/bestvideo/best",
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(page_url, download=False)
    except Exception as e:
        raise RuntimeError(f"yt-dlp could not get a stream from {page_url}: {e}") from e
    if info.get("_type") == "playlist":
        info = (info.get("entries") or [None])[0] or {}
    url = info.get("url") or next((f["url"] for f in info.get("requested_formats") or [] if f.get("url")), None)
    if not url:
        raise RuntimeError(f"yt-dlp found no playable video stream for {page_url}")
    if info.get("live_status") == "is_upcoming":
        raise RuntimeError("That live stream hasn't started yet.")
    return url, bool(info.get("is_live")), info.get("title") or ""


class VideoSource:
    def __init__(self, source, loop=False, max_height=720):
        self.page_url = None
        self.max_height = max_height
        self.title = ""
        if needs_resolving(source):
            self.page_url = str(source)
            print(f"[capture] resolving {self.page_url} with yt-dlp ...")
            self.source, self.is_live, self.title = resolve_stream(self.page_url, max_height)
            print(f"[capture] {'LIVE' if self.is_live else 'video'}: {self.title}")
        else:
            self.source = int(source) if str(source).isdigit() else source
            self.is_live = isinstance(self.source, int) or str(self.source).lower().startswith(
                ("rtsp://", "rtmp://", "http://", "https://")
            )
        self.loop = loop and not self.is_live
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        if self.is_live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise internal buffering latency
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 < fps < 120 else 30.0
        # Pace frames at the video's own fps for files and segment-based streams (YouTube/Twitch/HLS).
        # HLS delivers a few seconds of video in one burst; unpaced, the reader would race through
        # each burst and the agent would only ever see its last frame (~1 fps). Webcams and RTSP
        # push frames in real time, so they stay unpaced (always the newest frame, minimal lag).
        self.paced = (not self.is_live) or bool(self.page_url) or ".m3u8" in str(self.source).lower()

        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self.running = True
        self.ended = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reconnect(self):
        print(f"[capture] stream lost, reconnecting to {self.page_url or self.source} ...")
        self.cap.release()
        time.sleep(1.0)
        if self.page_url:  # resolved URLs expire - get a fresh one
            try:
                self.source, _, _ = resolve_stream(self.page_url, self.max_height)
            except Exception as e:
                print(f"[capture] {e}")
                time.sleep(4.0)
        self.cap = cv2.VideoCapture(self.source)

    def _reader(self):
        frame_period = 1.0 / self.fps if self.paced else 0.0
        failures = 0
        next_due = time.time()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self.is_live:
                    failures += 1
                    if failures >= 50:
                        self._reconnect()
                        failures = 0
                    time.sleep(0.05)
                    continue
                self.ended = True
                break
            failures = 0
            with self._lock:
                self._frame = frame
                self._frame_id += 1
            if frame_period:
                next_due += frame_period
                delay = next_due - time.time()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -2.0:
                    # stalled >2 s (slow network/decoder): resume normal speed from here rather than
                    # sprinting through the backlog, which would skip frames again
                    next_due = time.time()
                # else: slightly behind (e.g. a segment download) - read without sleeping to catch up

    def read(self):
        """Return (frame_id, frame copy) of the newest frame, or (id, None) if none yet."""
        with self._lock:
            if self._frame is None:
                return self._frame_id, None
            return self._frame_id, self._frame.copy()

    def release(self):
        self.running = False
        self._thread.join(timeout=2)
        self.cap.release()
