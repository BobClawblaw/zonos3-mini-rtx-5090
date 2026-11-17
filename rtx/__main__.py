"""python3 -m rtx fetch|build|up|down|status|logs|command|say|voices|validate|systemd"""

import argparse
import shlex
import sys

from . import config, engine, serve, systemd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m rtx", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="the release at its pinned revision, into model_dir (needs the hf command)")
    b = sub.add_parser("build", help="the pinned engine, under tf_cache")
    b.add_argument("--rebuild", action="store_true")
    up = sub.add_parser("up", help="start the server and its test page (in the background; --foreground stays, Ctrl-C stops both)")
    up.add_argument("--foreground", action="store_true")
    sub.add_parser("down", help="stop what `up` started")
    sub.add_parser("status", help="pid, url, the server's health and the test page")
    logs = sub.add_parser("logs", help="the log's last lines")
    logs.add_argument("-n", type=int, default=40)
    sub.add_parser("command", help="print the serve command")
    say = sub.add_parser("say", help="speak text with the running server into a file (.wav .mp3 .ogg .flac .aac .pcm)")
    say.add_argument("text")
    say.add_argument("-o", "--out", default="reply.wav")
    say.add_argument("--voice", default="")
    say.add_argument("--seed", type=int)
    say.add_argument("--ref", default="", help="a WAV clip whose voice to clone")
    sub.add_parser("voices", help="the running server's voice names")
    sub.add_parser("validate", help="check the settings, the release and the build without starting anything")
    sub.add_parser("systemd", help="write the systemd user unit (not enabled) that keeps `up --foreground` running")
    a = p.parse_args(argv)
    s = config.load()
    if a.cmd == "fetch":
        print(f"release {engine.fetch(s)}")
    elif a.cmd == "build":
        engine.build(s, rebuild=a.rebuild)
        print(f"engine {s.run_binary}")
    elif a.cmd == "up":
        problems = config.validate(s)
        if problems:
            raise SystemExit("\n".join(problems))
        serve.up(s, foreground=a.foreground)
    elif a.cmd == "down":
        serve.down(s)
    elif a.cmd == "status":
        print(serve.status(s))
    elif a.cmd == "logs":
        print("\n".join(s.log_file.read_text(errors="replace").splitlines()[-a.n:]) if s.log_file.exists() else "no log")
    elif a.cmd == "command":
        print(shlex.join(s.serve_args()))
    elif a.cmd == "say":
        print(serve.say(s, a.text, a.out, a.voice, a.seed, a.ref))
    elif a.cmd == "voices":
        print(serve.voices(s))
    elif a.cmd == "validate":
        problems = config.validate(s)
        print("\n".join(problems) if problems else f"ok: {s.served_name} from {s.model} with {s.run_binary} at {s.url}")
        for fmt, pkg in config.missing_codecs():
            print(f"note: no library for {fmt} replies (Debian/Ubuntu: sudo apt install {pkg})")
        return 1 if problems else 0
    elif a.cmd == "systemd":
        print(systemd.write(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
