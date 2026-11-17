#!/usr/bin/env python3
"""An OpenAI-style front for the ZONOS3 Mini server, for clients written against OpenAI's text to speech.

It listens on its own port, passes /v1/audio/speech to the model's server and smooths over what such clients send:
MP3, Opus, AAC and FLAC replies (transcoded from the server's WAV by ffmpeg), voice and model names the server does
not have (the default voice, the served model), `speed` and `instructions` (dropped; the reply's
X-Zonos-Ignored header names them), and requests that arrive together (spoken one after another, where the server
itself answers 503). Every other field goes through as it is. Standard library only; ffmpeg for the four formats.

    openai_front.py --upstream 127.0.0.1:30117 --port 30119 [--bind 127.0.0.1] [--voice default]
"""
import argparse
import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# response_format -> (ffmpeg output arguments, content type)
TRANSCODE = {
    "mp3": (["-f", "mp3"], "audio/mpeg"),
    "opus": (["-c:a", "libopus", "-f", "ogg"], "audio/ogg"),
    "aac": (["-f", "adts"], "audio/aac"),
    "flac": (["-f", "flac"], "audio/flac"),
}
DROPPED = ("speed", "instructions", "stream_format")
one_at_a_time = threading.Lock()


def upstream(args, method: str, path: str, body: bytes | None = None, timeout: float = 600):
    """(status, content type, bytes) from the model's server."""
    req = urllib.request.Request(f"http://{args.upstream}{path}", body, {"Content-Type": "application/json"} if body else {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type", "application/octet-stream"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", "application/json"), e.read()
    except OSError as e:
        return 502, "application/json", error(f"the ZONOS3 server at {args.upstream} did not answer: {e}")


def error(message: str) -> bytes:
    return json.dumps({"error": {"message": message, "type": "invalid_request_error"}}).encode()


def names(args) -> tuple[list[str], str]:
    """The server's voices and its model's name, asked once and kept."""
    if not args.known:
        _, _, v = upstream(args, "GET", "/v1/voices", timeout=10)
        _, _, m = upstream(args, "GET", "/v1/models", timeout=10)
        try:
            args.known = (json.loads(v)["voices"], json.loads(m)["data"][0]["id"])
        except (ValueError, KeyError, IndexError):
            return [], ""
    return args.known


def speak(args, raw: bytes):
    """(status, content type, bytes, ignored fields) for one speech request."""
    try:
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError
    except ValueError:
        return 400, "application/json", error("the request body must be a JSON object"), []
    voices, model = names(args)
    ignored = [k for k in DROPPED if body.pop(k, None) is not None]
    if model and body.get("model") != model:
        body["model"] = model
    voice = body.get("voice")
    if isinstance(voice, dict):  # OpenAI's {"id": ...} form
        voice = body["voice"] = voice.get("id")
    if voices and voice is not None and voice not in voices:
        ignored.append(f"voice={voice}")
        body["voice"] = args.voice
    wanted = body.get("response_format") or "mp3"  # OpenAI's default
    if wanted in TRANSCODE:
        if not shutil.which("ffmpeg"):
            return 501, "application/json", error(f"response_format {wanted} needs ffmpeg on this machine; wav and pcm do not"), ignored
        body["response_format"] = "wav"
        body.pop("stream", None)
    with one_at_a_time:
        for _ in range(600):  # the server may be busy with a request that did not come through here
            status, kind, data = upstream(args, "POST", "/v1/audio/speech", json.dumps(body).encode())
            if status != 503:
                break
            time.sleep(0.2)
    if status != 200 or wanted not in TRANSCODE:
        return status, kind, data, ignored
    out, kind = TRANSCODE[wanted]
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", *out, "pipe:1"], input=data, capture_output=True)
    if p.returncode != 0:
        return 500, "application/json", error(f"ffmpeg could not make {wanted}: {p.stderr.decode(errors='replace')[-300:]}"), ignored
    return 200, kind, p.stdout, ignored


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(self, status: int, kind: str, data: bytes, ignored: list[str] | None = None):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        if ignored:
            self.send_header("X-Zonos-Ignored", ", ".join(ignored))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/v1/models", "/v1/voices", "/health"):
            self.reply(*upstream(self.server.args, "GET", self.path, timeout=10))
        else:
            self.reply(404, "application/json", error("this front serves /v1/audio/speech, /v1/models, /v1/voices and /health"))

    def do_POST(self):
        if self.path != "/v1/audio/speech":
            return self.reply(404, "application/json", error("this front serves /v1/audio/speech"))
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self.reply(411, "application/json", error("Content-Length is required"))
        if n > 64 << 20:
            return self.reply(413, "application/json", error("the request is larger than 64 MiB"))
        self.reply(*speak(self.server.args, self.rfile.read(n)))

    def log_message(self, fmt, *a):
        print(f"front: {self.address_string()} {fmt % a}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream", default="127.0.0.1:30117", help="the ZONOS3 server's host:port")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30119)
    ap.add_argument("--voice", default="default", help="the voice for a name the server does not have (alloy, nova, ...)")
    args = ap.parse_args()
    args.known = None
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    srv.args = args
    print(f"front: OpenAI-style speech on http://{args.bind}:{args.port}/v1 -> {args.upstream}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
