"""python3 -m rtx fetch|build|up|down|status|logs|command|say|voices|controls|validate|systemd"""

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
    say.add_argument("--language", default="", help="the text normalizer's grammar: en_us (the default), fr_fr, de, ja, ...")
    say.add_argument("--as-typed", action="store_true", help="no text normalization: speak the text as typed")
    say.add_argument("--rate", type=float, help="speaking rate, in UTF-8 bytes of text a second (about 6 to 30; slower is lower)")
    say.add_argument("--loudness", type=float, help="loudness in LUFS (about -30 to -16)")
    say.add_argument("--guidance", type=float, help="speaker guidance scale: 1 is off, above 1 follows the voice harder")
    say.add_argument("--control", action="append", default=[], metavar="NAME=DOSE", help="a steering control, as `controls` lists them (mood.valence=2); repeat for more")
    say.add_argument("--by-embedding", action="store_true", help="speak the voice by its speaker embedding: what the controls that move the voice need")
    say.add_argument("--controls-guidance", type=float, help="guidance scale on the controls (with --by-embedding): 1 is off")
    say.add_argument("--laci", action="store_true", help="watch the reading of the text and regenerate a skip, a stall or an early end")
    say.add_argument("--start", action="store_true", help="speech from the first frame, with no silence before it")
    say.add_argument("--no-short-text", action="store_true", help="speak a text of a word or two as it is, without the filler sentence it is otherwise generated with")
    ed = sub.add_parser("edit", help="re-say part of a recorded clip (a WAV of at most 60 s) with the running server")
    ed.add_argument("clip")
    ed.add_argument("target", help="what the clip should say after the edit")
    ed.add_argument("-o", "--out", default="edited.wav")
    ed.add_argument("--text", default="", help="what the clip says now: the words that differ from the target are re-said (clips up to 30 s)")
    ed.add_argument("--span", action="append", nargs=2, type=float, default=[], metavar=("START", "END"), help="a region to re-say, in seconds; repeat for up to three")
    ed.add_argument("--seed", type=int)
    ed.add_argument("--takes", type=int, default=1, help="1 to 4 versions of the edit")
    ed.add_argument("--control", action="append", default=[], metavar="NAME=DOSE", help="a steering control on the new audio; repeat for more")
    ed.add_argument("--language", default="")
    ed.add_argument("--plain-row", action="store_true", help="leave the clip's speaker embedding out of the prompt")
    al = sub.add_parser("align", help="where each word of a transcript is spoken in a clip (a WAV of at most 30 s)")
    al.add_argument("clip")
    al.add_argument("text")
    al.add_argument("--language", default="")
    sub.add_parser("voices", help="the running server's voice names")
    sub.add_parser("controls", help="the running server's steering controls, by kind")
    sub.add_parser("validate", help="check the settings, the release and the build without starting anything")
    sub.add_parser("systemd", help="write the systemd user unit (not enabled) that keeps `up --foreground` running")
    a = p.parse_args(argv)
    s = config.load()
    if a.cmd == "fetch":
        print(f"release {engine.fetch(s)}")
    elif a.cmd == "build":
        engine.build(s, rebuild=a.rebuild)
        print(f"engine {s.run_binary}" + (f"\ntext normalizer {s.textnorm_lib}" if s.textnorm_lib.exists() else "")
              + (f"\nsteering bundle {s.bundle}" if s.bundle is not None and s.bundle.exists() else "")
              + (f"\nLACI heads {s.laci_heads}" if s.laci and s.laci_heads.is_dir() else ""))
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
    elif a.cmd in ("say", "edit"):
        controls = {}
        for item in a.control:
            name, _, dose = item.partition("=")
            try:
                controls[name] = float(dose)
            except ValueError:
                raise SystemExit(f"--control {item}: write NAME=DOSE, the dose a number") from None
        if a.cmd == "edit":
            if not a.span and not a.text:
                raise SystemExit("edit: give --text (what the clip says now) or --span START END")
            print(serve.edit(s, a.clip, a.target, a.out, a.text, a.span, a.seed, a.takes, controls, a.language, a.plain_row))
            return 0
        print(serve.say(s, a.text, a.out, a.voice, a.seed, a.ref, a.language, a.as_typed, a.rate, a.loudness, a.guidance, controls, a.by_embedding, a.controls_guidance, a.laci, a.start, a.no_short_text))
    elif a.cmd == "align":
        print(serve.align(s, a.clip, a.text, a.language))
    elif a.cmd == "voices":
        print(serve.voices(s))
    elif a.cmd == "controls":
        print(serve.controls(s))
    elif a.cmd == "validate":
        problems = config.validate(s)
        print("\n".join(problems) if problems else f"ok: {s.served_name} from {s.model} with {s.run_binary} at {s.url}")
        for fmt, pkg in config.missing_codecs():
            print(f"note: no library for {fmt} replies (Debian/Ubuntu: sudo apt install {pkg})")
        if not config.has_cublas():
            print("note: the loader does not find cuBLAS (libcublas.so.13, CUDA 13's): the decoder needs it, so the server will not load without it")
        if not s.normalizes:
            print("note: no text normalizer: numbers, dates and currency are spoken as typed (python3 -m rtx build, with Rust 1.88 or later)")
        return 1 if problems else 0
    elif a.cmd == "systemd":
        print(systemd.write(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
