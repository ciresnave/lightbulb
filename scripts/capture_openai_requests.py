#!/usr/bin/env python3
"""Record what an OpenAI-compatible client (e.g. Cherry Studio) actually sends.

WHY: ROADMAP F4/F3 hinge on facts nobody here has observed: does the client
stream when it sends `tools`, which `tool_choice` does it use, does it call
`/v1/embeddings`. Its published docs are silent on all three. This is a stub
OpenAI endpoint that logs each request and answers just enough for the client
to carry on. It never reads the client's own configuration or database.

USAGE
    python scripts/capture_openai_requests.py            # listens on 127.0.0.1:8099
    python scripts/capture_openai_requests.py --port 8123

Writes one JSON object per request to `capture.jsonl` in the current directory:
method, path, headers (the `Authorization` value is replaced by "<redacted>"),
and the parsed JSON body. The bodies contain whatever you typed into the chat.

Binds 127.0.0.1 only. Stdlib only. Stop with Ctrl+C.
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "capture-test"
LOG_PATH = "capture.jsonl"


def log_request(method, path, headers, raw_body):
    try:
        body = json.loads(raw_body) if raw_body else None
    except ValueError:
        body = {"_unparsed_utf8": raw_body.decode("utf-8", "replace")}
    safe = {}
    for k, v in headers.items():
        safe[k] = "<redacted>" if k.lower() in ("authorization", "x-api-key", "api-key") else v
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "method": method, "path": path,
                            "headers": safe, "body": body}) + "\n")
    return body


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # one terse line per request  # pylint: disable=redefined-builtin
        sys.stderr.write("%s %s\n" % (self.command, self.path))

    def _send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        log_request("GET", self.path, dict(self.headers), b"")
        if self.path.rstrip("/").endswith("/models"):
            self._send_json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "capture"}]})
        else:
            self._send_json(404, {"error": {"message": "capture stub: not found"}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        body = log_request("POST", self.path, dict(self.headers), raw)
        path = self.path.split("?")[0].rstrip("/")
        if path.endswith("/chat/completions"):
            self._chat(body or {})
        elif path.endswith("/embeddings"):
            self._embed(body or {})
        else:
            self._send_json(404, {"error": {"message": "capture stub: not found"}})

    def _embed(self, body):
        inputs = body.get("input", [])
        count = len(inputs) if isinstance(inputs, list) else 1
        self._send_json(200, {"object": "list", "model": MODEL_ID,
                              "data": [{"object": "embedding", "index": i, "embedding": [0.0, 0.1, 0.2]}
                                       for i in range(count)],
                              "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    def _chat(self, body):
        text = "capture stub reply"
        usage = {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4}
        if not body.get("stream"):
            self._send_json(200, {"id": "cap-1", "object": "chat.completion", "created": 0,
                                  "model": MODEL_ID, "usage": usage,
                                  "choices": [{"index": 0, "finish_reason": "stop",
                                               "message": {"role": "assistant", "content": text}}]})
            return
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        payload = b"".join(
            b"data: " + json.dumps(dict(c, id="cap-1", object="chat.completion.chunk",
                                        created=0, model=MODEL_ID)).encode() + b"\n\n"
            for c in chunks) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8099)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("capture stub on http://127.0.0.1:%d/v1  (model id %r); logging to %s"
          % (args.port, MODEL_ID, LOG_PATH), file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
