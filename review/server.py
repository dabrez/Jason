"""Local review server: watch each candidate clip and label it.

Serves the review UI plus the cut clips, and persists labels to
clip_labels.json on every change (no explicit save step -- a review session
that gets interrupted keeps everything judged so far).

Labels are stored keyed by clip id (video_id:start-end) so they survive
re-cutting and can be replayed as a regression check against future
selection changes -- see review/score.py.

    python3 review/build.py --video .cache_video/<id>.mp4   # cut the set
    python3 review/server.py                                # judge it
"""
import argparse
import json
import os
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

REVIEW_DIR = "review_clips"
LABELS_PATH = "clip_labels.json"

_lock = threading.Lock()


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded so a long-running video stream can't block the label API --
    the browser holds video connections open while the reviewer watches.
    """
    daemon_threads = True


def load_labels(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(path: str, labels: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    os.replace(tmp, path)


class ReviewHandler(SimpleHTTPRequestHandler):
    review_dir = REVIEW_DIR
    labels_path = LABELS_PATH
    # HTTP/1.1 keep-alive: Chrome's <video> element stalls on HTTP/1.0
    # responses for media, waiting on a connection close that never cleanly
    # arrives. Every response here sends an accurate Content-Length, which
    # 1.1 requires for keep-alive to work.
    protocol_version = "HTTP/1.1"

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path

        if path == "/":
            return self._serve_file(os.path.join(os.path.dirname(__file__), "review.html"),
                                    "text/html; charset=utf-8")
        if path == "/api/manifest":
            manifest_path = os.path.join(self.review_dir, "manifest.json")
            if not os.path.exists(manifest_path):
                return self._send_json({"error": "No manifest. Run review/build.py first."}, 404)
            with open(manifest_path, "r", encoding="utf-8") as f:
                return self._send_json(json.load(f))
        if path == "/api/labels":
            with _lock:
                return self._send_json(load_labels(self.labels_path))
        if path.startswith("/clips/"):
            name = os.path.basename(path[len("/clips/"):])
            return self._serve_file(os.path.join(self.review_dir, name), "video/mp4")
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        self.send_error(404)

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/api/labels":
            return self.send_error(404)
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send_json({"error": "bad json"}, 400)

        clip_id = payload.get("id")
        if not clip_id:
            return self._send_json({"error": "missing id"}, 400)

        with _lock:
            labels = load_labels(self.labels_path)
            if payload.get("verdict") is None:
                labels.pop(clip_id, None)
            else:
                labels[clip_id] = {
                    "verdict": payload.get("verdict"),
                    "reasons": payload.get("reasons", []),
                    "note": payload.get("note", ""),
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "group": payload.get("group"),
                }
            save_labels(self.labels_path, labels)
            count = len(labels)
        return self._send_json({"ok": True, "count": count})

    def _serve_file(self, path, content_type):
        if not os.path.exists(path):
            return self.send_error(404)
        size = os.path.getsize(path)
        range_header = self.headers.get("Range")

        # Range support matters: without it, seeking inside a clip in the
        # browser's video element doesn't work.
        start, end = 0, size - 1
        status = 200
        if range_header and range_header.startswith("bytes="):
            spec = range_header[len("bytes="):].split(",")[0]
            lo, _, hi = spec.partition("-")
            if lo:
                start = int(lo)
                end = int(hi) if hi else size - 1
            elif hi:
                start = max(0, size - int(hi))
            start = max(0, min(start, size - 1))
            end = max(start, min(end, size - 1))
            status = 206

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return  # browser seeked away or closed the tab
                remaining -= len(chunk)

    def log_message(self, fmt, *args):
        # args[0] is the request line for normal logs but an int status code
        # when called from send_error, so coerce before matching.
        super().log_message(fmt, *args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--review-dir", default=REVIEW_DIR)
    parser.add_argument("--labels", default=LABELS_PATH)
    args = parser.parse_args()

    ReviewHandler.review_dir = args.review_dir
    ReviewHandler.labels_path = args.labels

    manifest = os.path.join(args.review_dir, "manifest.json")
    if not os.path.exists(manifest):
        print(f"No {manifest}. Run review/build.py first.", file=sys.stderr)
        return 1

    existing = len(load_labels(args.labels))
    print(f"Review server: http://localhost:{args.port}")
    print(f"Labels -> {args.labels} ({existing} already recorded)")
    print("Ctrl-C to stop.\n")
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler).serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. {len(load_labels(args.labels))} labels saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
