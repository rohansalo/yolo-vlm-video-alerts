"""Read-only web dashboard for the vision agent's SQLite database.

Run from the vlm_alerts folder (in a second terminal while the agent runs, or afterwards):
    python -m vision_agent.dashboard                  # http://localhost:8050
    python -m vision_agent.dashboard --db vision_agent.db --snapshots snapshots --port 8050

The page polls /api/... every 2 seconds, so a running session updates live.
"""
import argparse
import os
import sqlite3
from datetime import datetime, timedelta

from flask import Flask, abort, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))


def create_app(db_path, snapshot_dir):
    app = Flask(__name__)
    snapshot_dir = os.path.abspath(snapshot_dir)

    def q(sql, params=()):
        if not os.path.exists(db_path):
            return []
        # read-only: the dashboard can never modify or lock out the agent's writes
        conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        except sqlite3.OperationalError:  # tables not created yet
            return []
        finally:
            conn.close()

    def timeline(sid, start, end, max_buckets=60):
        """Event counts in time buckets sized to the session (10s / 1min / 10min), gaps filled with zeros."""
        duration = (end - start).total_seconds()
        size = 10 if duration <= 600 else 60 if duration <= 7200 else 600
        rows = q(
            """SELECT (CAST(strftime('%s', ts) AS INTEGER) / ?) * ? AS b, COUNT(*) AS n, SUM(is_alert) AS alerts
               FROM events WHERE session_id=? GROUP BY b""",
            (size, size, sid),
        )
        counts = {r["b"]: r for r in rows}
        # strftime('%s') treats the naive local timestamp as UTC, so build bucket keys the same way
        epoch = datetime(1970, 1, 1)
        first = int((start - epoch).total_seconds()) // size * size
        last = int((end - epoch).total_seconds()) // size * size
        if rows:
            last = max(last, max(counts))
        first = max(first, last - (max_buckets - 1) * size)
        fmt = "%H:%M:%S" if size < 60 else "%H:%M"
        return {
            "bucket_s": size,
            "rows": [
                {
                    "label": (epoch + timedelta(seconds=b)).strftime(fmt),
                    "n": counts.get(b, {}).get("n", 0),
                    "alerts": counts.get(b, {}).get("alerts", 0) or 0,
                }
                for b in range(first, last + 1, size)
            ],
        }

    @app.route("/")
    def index():
        return send_from_directory(HERE, "dashboard.html")

    @app.route("/api/sessions")
    def sessions():
        return jsonify(
            q(
                """SELECT s.id, s.source, s.model, s.started_at, s.ended_at, s.frames, s.avg_fps,
                          COUNT(e.id) AS events, COALESCE(SUM(e.is_alert), 0) AS alerts
                   FROM sessions s LEFT JOIN events e ON e.session_id = s.id
                   GROUP BY s.id ORDER BY s.started_at DESC LIMIT 100"""
            )
        )

    @app.route("/api/session/<sid>")
    def session(sid):
        rows = q("SELECT * FROM sessions WHERE id=?", (sid,))
        if not rows:
            abort(404)
        s = rows[0]
        start = datetime.strptime(s["started_at"], "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(s["ended_at"], "%Y-%m-%d %H:%M:%S") if s["ended_at"] else datetime.now()
        kpi = q(
            """SELECT COUNT(*) AS events, COALESCE(SUM(is_alert),0) AS alerts,
                      COUNT(DISTINCT object_class || ':' || track_id) AS objects,
                      AVG(vlm_latency_s) AS vlm_latency
               FROM events WHERE session_id=?""",
            (sid,),
        )[0]
        return jsonify(
            {
                "session": s,
                "live": s["ended_at"] is None,
                "duration_s": max(0, int((end - start).total_seconds())),
                "kpi": kpi,
                "by_type": q(
                    "SELECT type AS label, COUNT(*) AS n FROM events WHERE session_id=? GROUP BY type ORDER BY n DESC",
                    (sid,),
                ),
                "by_class": q(
                    """SELECT object_class AS label, COUNT(DISTINCT track_id) AS n FROM events
                       WHERE session_id=? AND track_id IS NOT NULL GROUP BY object_class ORDER BY n DESC""",
                    (sid,),
                ),
                "questions": q(
                    """SELECT id, question, status, asked_at, answered_at, answer, snapshot_path, vlm_latency_s
                       FROM questions WHERE session_id=? AND kind='ask' ORDER BY id DESC LIMIT 20""",
                    (sid,),
                ),
                "watch": (q(
                    """SELECT question, answer, answered_at FROM questions
                       WHERE session_id=? AND kind='watch' AND status='done' ORDER BY id DESC LIMIT 1""",
                    (sid,),
                ) or [None])[0],
                "by_zone": q(
                    """SELECT zone AS label, COUNT(*) AS n FROM events
                       WHERE session_id=? GROUP BY zone ORDER BY n DESC""",
                    (sid,),
                ),
                "timeline": timeline(sid, start, end),
                "events": q(
                    """SELECT id, ts, type, zone, track_id, object_class, confidence, speed, is_alert,
                              snapshot_path, description, vlm_latency_s
                       FROM events WHERE session_id=? ORDER BY id DESC LIMIT 200""",
                    (sid,),
                ),
                "summaries": q(
                    "SELECT ts, object_counts, description FROM summaries WHERE session_id=? ORDER BY id DESC LIMIT 5",
                    (sid,),
                ),
            }
        )

    @app.route("/api/session/<sid>/ask", methods=["POST"])
    def ask(sid):
        # JSON-only: browsers can't send this cross-site without a CORS preflight, which we never allow
        if not request.is_json:
            abort(415)
        question = str((request.get_json(silent=True) or {}).get("question", "")).strip()
        if not question:
            return jsonify(error="Type a question first."), 400
        if len(question) > 500:
            return jsonify(error="Keep the question under 500 characters."), 400
        rows = q("SELECT ended_at FROM sessions WHERE id=?", (sid,))
        if not rows:
            abort(404)
        if rows[0]["ended_at"]:
            return jsonify(error="This session has ended. Start the agent to ask about the live scene."), 409
        conn = sqlite3.connect(os.path.abspath(db_path), timeout=5)
        try:
            cur = conn.execute(
                "INSERT INTO questions (session_id, kind, question, status, asked_at) VALUES (?,?,?,?,?)",
                (sid, "ask", question, "pending", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
            return jsonify(id=cur.lastrowid)
        except sqlite3.OperationalError as e:
            return jsonify(error=f"Database not ready: {e}. Restart the agent once to create the questions table."), 503
        finally:
            conn.close()

    @app.route("/snapshot/<path:name>")
    def snapshot(name):
        # only serve files that live directly in the snapshots folder
        return send_from_directory(snapshot_dir, os.path.basename(name))

    return app


def main():
    p = argparse.ArgumentParser(description="Vision agent dashboard")
    p.add_argument("--db", default="vision_agent.db")
    p.add_argument("--snapshots", default="snapshots")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args()
    if not os.path.exists(args.db):
        print(f"[dashboard] {args.db} not found yet - it appears once the agent has run. Serving anyway.")
    print(f"[dashboard] open http://localhost:{args.port}")
    create_app(args.db, args.snapshots).run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
