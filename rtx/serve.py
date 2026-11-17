"""The server and its test page: started in the background (pid files, a log) or in the foreground (Ctrl-C, or the
systemd unit), stopped, watched; and one request to the running server."""

import base64
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

from .config import CLONE_GIB, LOAD_GIB, Settings, missing_codecs

EXT = {"wav": "wav", "mp3": "mp3", "ogg": "opus", "opus": "opus", "flac": "flac", "aac": "aac", "pcm": "pcm"}


def health(s: Settings, timeout: float = 3) -> dict | None:
    try:
        with urllib.request.urlopen(s.url.removesuffix("/v1") + "/health", timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def _pid(f) -> int | None:
    """A pid file's process if it runs; a file whose process is gone or a zombie (exited, unreaped) is removed."""
    try:
        p = int(f.read_text())
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(p)], capture_output=True, text=True).stdout.strip()
        if state and not state.startswith("Z"):
            return p
    except Exception:
        return None
    f.unlink(missing_ok=True)
    return None


def pid(s: Settings) -> int | None:
    return _pid(s.pid_file)


def free_gib() -> float | None:
    """The card's free memory, or None where nvidia-smi does not say."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
        return max(int(x) for x in out.split()) * 2**20 / 2**30
    except Exception:
        return None


def memory_check(s: Settings) -> None:
    """The load must fit; cloning from a clip needs more, which is said, not refused."""
    have = free_gib()
    if have is None:
        return
    if have < LOAD_GIB + 0.5:
        raise SystemExit(f"{have:.1f} GiB free on the card, the server needs {LOAD_GIB + 0.5:.1f}: stop other models first")
    if have < CLONE_GIB + 0.5:
        print(f"note: {have:.1f} GiB free on the card; a request with ref_audio needs {CLONE_GIB + 0.5:.1f} in all and will be refused")


def wait_ready(s: Settings, proc: subprocess.Popen, limit: float = 300) -> None:
    t0 = time.time()
    while time.time() - t0 < limit:
        if health(s):
            return
        if proc.poll() is not None:  # exited (a refused setting, say): poll() reaps it, where kill(pid, 0) sees a zombie
            s.pid_file.unlink(missing_ok=True)
            raise SystemExit(f"the server exited while loading (status {proc.returncode}); see {s.log_file}")
        time.sleep(1)
    raise SystemExit(f"not ready after {limit:.0f} s; see {s.log_file}")


def _announce(s: Settings) -> None:
    print(f"serving {s.served_name}: API {s.url}" + (f", test page {s.page_url}" if s.page_port else ""))
    for fmt, pkg in missing_codecs():
        print(f"note: {fmt} replies will be refused until its library is installed (Debian/Ubuntu: sudo apt install {pkg})")


def up(s: Settings, foreground: bool = False) -> None:
    if pid(s) or health(s):
        raise SystemExit(f"already serving at {s.url}")
    memory_check(s)
    s.cache.mkdir(parents=True, exist_ok=True)
    if foreground:
        return _foreground(s)
    log = open(s.log_file, "ab")
    p = subprocess.Popen(s.serve_args(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    s.pid_file.write_text(str(p.pid))
    wait_ready(s, p)
    if s.page_port:
        page = subprocess.Popen(s.page_args(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        s.page_pid_file.write_text(str(page.pid))
    _announce(s)


def _foreground(s: Settings) -> None:
    """The server and the page as this process's children, their output here; Ctrl-C or SIGTERM stops both."""
    procs = [subprocess.Popen(s.serve_args())]

    def stop(signum=None, frame=None):
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        raise SystemExit(0 if signum else 1)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        wait_ready(s, procs[0])
    except SystemExit:
        stop()
    if s.page_port:
        procs.append(subprocess.Popen(s.page_args()))
    _announce(s)
    print("Ctrl-C stops it", flush=True)
    while all(p.poll() is None for p in procs):
        time.sleep(1)
    stop()  # one of them ended by itself: the other goes too


def _stop(f, grace: float) -> bool:
    p = _pid(f)
    if not p:
        return False
    os.kill(p, signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < grace:
        try:
            os.kill(p, 0)
        except OSError:
            break
        time.sleep(0.5)
    else:
        os.kill(p, signal.SIGKILL)
    f.unlink(missing_ok=True)
    return True


def down(s: Settings, grace: float = 30) -> None:
    _stop(s.page_pid_file, 5)
    print("stopped" if _stop(s.pid_file, grace) else "not running")


def status(s: Settings) -> str:
    h = health(s)
    page = _pid(s.page_pid_file)
    return f"pid: {pid(s) or 'not running'}\nurl: {s.url}\nhealth: {json.dumps(h) if h else 'down'}\ntest page: {s.page_url if page else 'not running'}"


def say(s: Settings, text: str, out: str, voice: str = "", seed: int | None = None, ref: str = "") -> str:
    """`text` spoken by the running server into `out`, whose extension picks the format; `ref` is a WAV clip to clone."""
    fmt = EXT.get(out.rsplit(".", 1)[-1].lower())
    if fmt is None:
        raise SystemExit(f"{out}: the extension must be one of {', '.join(sorted(EXT))}")
    body = {"model": s.served_name, "input": text, "response_format": fmt}
    if voice:
        body["voice"] = voice
    if seed is not None:
        body["seed"] = seed
    if ref:
        body["ref_audio"] = base64.b64encode(open(ref, "rb").read()).decode()
    headers = {"content-type": "application/json", **({"authorization": f"Bearer {s.api_key}"} if s.api_key else {})}
    req = urllib.request.Request(s.url + "/audio/speech", data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{e.code}: {e.read().decode(errors='replace')[:400]}") from None
    open(out, "wb").write(data)
    return f"{out}: {len(data)} bytes in {time.time() - t0:.2f} s"


def voices(s: Settings) -> str:
    with urllib.request.urlopen(s.url + "/voices", timeout=10) as r:
        return " ".join(json.load(r)["voices"])
