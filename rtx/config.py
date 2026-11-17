"""Settings: recipe.toml's defaults, local.toml's overrides for this machine, then environment variables in upper case."""

import ctypes.util
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# what the server reads of the release, and the files that show a download is whole
INCLUDE = ("lm/*", "autoencoder/*", "registries/*", "gguf/voices/*", "speaker_encoders/*", "README.md")
NEEDED = ("lm/zonos3_mini/model.pth", "autoencoder/generator.pt", "autoencoder/teacher/model.safetensors", "gguf/voices")
# device memory measured on an RTX 5090: after load, and once a request with a clip has loaded the clip encoder
LOAD_GIB = 3.83e9 / 2**30
CLONE_GIB = 9.47e9 / 2**30
# the libraries the server loads when a reply is first asked in their format: (format, library, Debian package)
CODECS = (("mp3", "mp3lame", "libmp3lame0"), ("opus", "opus", "libopus0"), ("aac", "fdk-aac", "libfdk-aac2"))


@dataclass
class Settings:
    model_repo: str = ""
    model_revision: str = ""
    served_name: str = ""
    model_dir: str = ""
    tensorfold_src: str = ""
    tensorfold_sha: str = ""
    zig: str = "zig"
    nvcc: str = "/usr/local/cuda/bin/nvcc"
    sm: str = "120"
    binary: str = ""
    host: str = "127.0.0.1"
    port: int = 30117
    page_port: int = 30118
    page_bind: str = "127.0.0.1"
    tf_cache: str = "~/.cache/zonos3-mini"
    api_key: str = ""
    extra_args: str = ""

    @staticmethod
    def path(p: str) -> Path:
        return Path(os.path.expanduser(p))

    @property
    def model(self) -> Path:
        return self.path(self.model_dir)

    @property
    def cache(self) -> Path:
        return self.path(self.tf_cache)

    @property
    def worktree(self) -> Path:
        return self.cache / "engine" / self.tensorfold_sha[:12]

    @property
    def built_binary(self) -> Path:
        return self.worktree / "zig-out" / "native" / "bin" / "tensorfold-native"

    @property
    def run_binary(self) -> Path:
        return self.path(self.binary) if self.binary else self.built_binary

    @property
    def page_script(self) -> Path:
        return self.worktree / "tools" / "zig" / "zonos3_test_page.py"

    @property
    def pid_file(self) -> Path:
        return self.cache / "server.pid"

    @property
    def page_pid_file(self) -> Path:
        return self.cache / "page.pid"

    @property
    def log_file(self) -> Path:
        return self.cache / "server.log"

    @property
    def local_host(self) -> str:
        return "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host

    @property
    def url(self) -> str:
        return f"http://{self.local_host}:{self.port}/v1"

    @property
    def page_url(self) -> str:
        return f"http://{'127.0.0.1' if self.page_bind in ('0.0.0.0', '') else self.page_bind}:{self.page_port}/"

    def serve_args(self) -> list[str]:
        a = [str(self.run_binary), "serve", str(self.model), "--name", self.served_name, "--host", self.host, "--port", str(self.port),
             "--parallel", "1", "--no-update-check"]
        if self.api_key:
            a += ["--api-key", self.api_key]
        return a + self.extra_args.split()

    def page_args(self) -> list[str]:
        return ["python3", str(self.page_script), "--upstream", f"{self.local_host}:{self.port}", "--bind", self.page_bind, "--port", str(self.page_port)]


def _coerce(kind: type, raw):
    if kind is bool:
        return raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
    return kind(raw)


def load(local_file: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    merged: dict = {}
    for f in (ROOT / "recipe.toml", local_file or ROOT / "local.toml"):
        if f.exists():
            for section in tomllib.loads(f.read_text()).values():
                merged.update(section)
    s = Settings()
    for fld in fields(Settings):
        kind = type(getattr(s, fld.name))
        if fld.name in merged:
            setattr(s, fld.name, _coerce(kind, merged[fld.name]))
        if fld.name.upper() in env:
            setattr(s, fld.name, _coerce(kind, env[fld.name.upper()]))
    return s


def missing_codecs() -> list[tuple[str, str]]:
    """(format, package) for each encoded format whose library the loader does not find."""
    return [(fmt, pkg) for fmt, lib, pkg in CODECS if not ctypes.util.find_library(lib)]


def validate(s: Settings) -> list[str]:
    """The settings' problems, worded for the user (empty: none)."""
    out = []
    if not s.binary and not s.tensorfold_sha:
        out.append("no engine: set tensorfold_sha (local.toml) or binary")
    lacking = [n for n in NEEDED if not (s.model / n).exists()]
    if lacking:
        out.append(f"no release at {s.model} (missing {', '.join(lacking)}): python3 -m rtx fetch")
    if not s.run_binary.exists():
        out.append(f"no engine binary at {s.run_binary}: python3 -m rtx build")
    if s.page_port and not s.binary and s.built_binary.exists() and not s.page_script.exists():
        out.append(f"no test page at {s.page_script}: the pinned engine should hold it")
    if s.host not in ("127.0.0.1", "localhost") and not s.api_key:
        out.append(f"host {s.host} serves other machines without an api_key: set one, or host = \"127.0.0.1\"")
    return out
