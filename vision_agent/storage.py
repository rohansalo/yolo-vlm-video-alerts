"""SQLite persistence: sessions, events (with snapshots + VLM descriptions) and periodic scene summaries."""
import json
import sqlite3
import threading
from datetime import datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT,
    model TEXT,
    zones TEXT,
    started_at TEXT,
    ended_at TEXT,
    frames INTEGER DEFAULT 0,
    avg_fps REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    ts TEXT,
    type TEXT,
    zone TEXT,
    track_id INTEGER,
    object_class TEXT,
    confidence REAL,
    speed REAL,
    is_alert INTEGER,
    snapshot_path TEXT,
    description TEXT,
    vlm_latency_s REAL
);
CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    ts TEXT,
    object_counts TEXT,
    description TEXT,
    vlm_latency_s REAL
);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    kind TEXT DEFAULT 'ask',          -- 'ask' = one-off from the dashboard, 'watch' = --ask yes/no check
    question TEXT,
    status TEXT DEFAULT 'pending',    -- pending -> answering -> done | error
    asked_at TEXT,
    answered_at TEXT,
    answer TEXT,
    snapshot_path TEXT,
    vlm_latency_s REAL
);
CREATE INDEX IF NOT EXISTS idx_questions_session ON questions(session_id, status);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type, zone);
"""


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Store:
    def __init__(self, path="vision_agent.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _exec(self, sql, params=()):
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid

    def start_session(self, source, model, zones):
        sid = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._exec(
            "INSERT OR REPLACE INTO sessions (id, source, model, zones, started_at) VALUES (?,?,?,?,?)",
            (sid, str(source), model, json.dumps([z.to_dict() for z in zones]), now()),
        )
        return sid

    def update_progress(self, sid, frames, avg_fps):
        """Live heartbeat so the dashboard can show frames processed and FPS while running."""
        self._exec("UPDATE sessions SET frames=?, avg_fps=? WHERE id=?", (frames, avg_fps, sid))

    def end_session(self, sid, frames, avg_fps):
        self._exec("UPDATE sessions SET ended_at=?, frames=?, avg_fps=? WHERE id=?", (now(), frames, avg_fps, sid))

    def add_event(self, sid, ev, is_alert, snapshot_path=None):
        return self._exec(
            "INSERT INTO events (session_id, ts, type, zone, track_id, object_class, confidence, speed, is_alert, snapshot_path)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (sid, now(), ev["type"], ev["zone"], ev["track_id"], ev["object_class"], ev["confidence"], ev["speed"],
             int(is_alert), snapshot_path),
        )

    def set_event_description(self, event_id, description, latency):
        self._exec("UPDATE events SET description=?, vlm_latency_s=? WHERE id=?", (description, latency, event_id))

    def add_question(self, sid, question, kind="ask", status="pending"):
        return self._exec(
            "INSERT INTO questions (session_id, kind, question, status, asked_at) VALUES (?,?,?,?,?)",
            (sid, kind, question, status, now()),
        )

    def claim_pending_questions(self, sid, limit=3):
        """Atomically take up to `limit` pending dashboard questions for this session."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, question FROM questions WHERE session_id=? AND status='pending' ORDER BY id LIMIT ?",
                (sid, limit),
            ).fetchall()
            for qid, _ in rows:
                self.conn.execute("UPDATE questions SET status='answering' WHERE id=?", (qid,))
            self.conn.commit()
        return rows

    def release_question(self, qid):
        self._exec("UPDATE questions SET status='pending' WHERE id=?", (qid,))

    def answer_question(self, qid, answer, latency, snapshot_path=None, ok=True):
        self._exec(
            "UPDATE questions SET status=?, answer=?, answered_at=?, vlm_latency_s=?, snapshot_path=? WHERE id=?",
            ("done" if ok else "error", answer, now(), latency, snapshot_path, qid),
        )

    def add_summary(self, sid, counts, description, latency):
        return self._exec(
            "INSERT INTO summaries (session_id, ts, object_counts, description, vlm_latency_s) VALUES (?,?,?,?,?)",
            (sid, now(), json.dumps(counts), description, latency),
        )
