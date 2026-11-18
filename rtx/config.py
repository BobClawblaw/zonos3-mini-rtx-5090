"""Settings: recipe.toml's defaults, local.toml's overrides for this machine, then environment variables in upper case."""

import ctypes.util
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# what the server reads of the release, and the files that show a download is whole
INCLUDE = ("lm/*", "autoencoder/*", "registries/*", "gguf/voices/*", "speaker_encoders/*", "web/aligner/*", "README.md")
NEEDED = ("lm/zonos3_mini/model.pth", "autoencoder/generator.pt", "autoencoder/teacher/model.safetensors", "gguf/voices")
# device memory measured on an RTX 5090: after load, and once a request with a clip has loaded the clip encoder
# the engine's own device bytes: the load at one request at once, each request at once past it, the clip encoder
LOAD_BYTES = 3_269_295_616
LANE_BYTES = 495_527_936
CLIP_BYTES = 5_643_510_688
# the libraries the server loads when a reply is first asked in their format:
# (format, library, the variable that names its file instead, Debian package)
CODECS = (("mp3", "mp3lame", "TENSORFOLD_LIBMP3LAME", "libmp3lame0"), ("opus", "opus", "TENSORFOLD_LIBOPUS", "libopus0"),
          ("aac", "fdk-aac", "TENSORFOLD_LIBFDKAAC", "libfdk-aac2"))
# Zyphra's text normalizer, which the server loads from lib/ beside its binary: release 0.2.0, the reference
# repository's third_party/zyphra-textnorm submodule
TEXTNORM_REPO = "https://github.com/Zyphra/zyphra-textnorm"
TEXTNORM_SHA = "170c571e75034c9a9f3bece9885c36d8dae77df2"
TEXTNORM_RUST = (1, 88)
TEXTNORM_VAR = "TENSORFOLD_LIBZYPHRA_TEXTNORM"
# the steering bundle: direction vectors Zyphra keeps in its code repository (MIT), not in the release
STEER_REPO = "https://github.com/Zyphra/ZONOS3-Mini"
STEER_SHA = "96d9e0ec1e39083ed336ac78f24e3dbc93721c0f"
STEER_DIR = "assets/steering"
STEER_BUNDLE = "studio_bundle_mect60k_x10.json"
# from the same repository and commit: LACI's alignment heads, and the short-text registry of its Python package
LACI_DIR = "assets/laci_heads"
SHORT_DIR = "model/src/zonos3/registries"
SHORT_FILE = "short_text.json"


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
    cargo: str = "cargo"
    binary: str = ""
    host: str = "127.0.0.1"
    port: int = 30117
    page_port: int = 30118
    page_bind: str = "127.0.0.1"
    tf_cache: str = "~/.cache/zonos3-mini"
    parallel: int = 4
    api_key: str = ""
    steer: bool = True
    steer_bundle: str = ""
    laci: bool = True
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
    def load_gib(self) -> float:
        """Device memory the server takes when it starts."""
        return (LOAD_BYTES + (self.parallel - 1) * LANE_BYTES) / 2**30

    @property
    def clone_gib(self) -> float:
        """Device memory once a request with a clip has loaded the clip encoder."""
        return self.load_gib + CLIP_BYTES / 2**30

    @property
    def textnorm_src(self) -> Path:
        return self.cache / "textnorm" / TEXTNORM_SHA[:12]

    @property
    def textnorm_lib(self) -> Path:
        """Where the server looks by itself: lib/ beside the bin/ its binary is in."""
        return self.run_binary.parent.parent / "lib" / "libzyphra_textnorm.so"

    @property
    def normalizes(self) -> bool:
        named = os.environ.get(TEXTNORM_VAR)
        return os.path.isfile(named) if named is not None else self.textnorm_lib.exists()

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

    @property
    def steer_src(self) -> Path:
        """Where `build` checks out the publisher's steering folder."""
        return self.cache / "steering" / STEER_SHA[:12]

    @property
    def bundle(self) -> Path | None:
        """The steering bundle the server loads: steer_bundle when set, else the one `build` fetched; None: steer is off."""
        if not self.steer:
            return None
        return self.path(self.steer_bundle) if self.steer_bundle else self.steer_src / STEER_DIR / STEER_BUNDLE

    @property
    def laci_heads(self) -> Path:
        """The folder of alignment-head files `build` fetched: what `laci` requests and short texts need."""
        return self.steer_src / LACI_DIR

    @property
    def short_text(self) -> Path:
        return self.steer_src / SHORT_DIR / SHORT_FILE

    def serve_args(self) -> list[str]:
        a = [str(self.run_binary), "serve", str(self.model), "--name", self.served_name, "--host", self.host, "--port", str(self.port),
             "--parallel", str(self.parallel), "--no-update-check"]
        if self.api_key:
            a += ["--api-key", self.api_key]
        if self.bundle is not None:
            a += ["--steer-bundle", str(self.bundle)]
        if self.laci:
            a += ["--laci-heads", str(self.laci_heads), "--short-text", str(self.short_text)]
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
    """(format, package) for each encoded format whose library neither its variable nor the loader finds."""
    return [(fmt, pkg) for fmt, lib, var, pkg in CODECS if not (os.path.isfile(os.environ.get(var, "")) or ctypes.util.find_library(lib))]


def has_cublas() -> bool:
    """Whether the loader finds cuBLAS, which the engine opens for the decoder's linear layers."""
    return bool(ctypes.util.find_library("cublas"))


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
    if s.bundle is not None and not s.bundle.exists():
        out.append(f"no steering bundle at {s.bundle}: python3 -m rtx build fetches it (or steer = false)")
    if s.laci and not (s.laci_heads.is_dir() and s.short_text.exists()):
        out.append(f"no LACI heads at {s.laci_heads} or no {s.short_text}: python3 -m rtx build fetches them (or laci = false)")
    if not 1 <= s.parallel <= 16:
        out.append(f"parallel = {s.parallel}: the server takes 1 to 16 requests at once")
    if s.host not in ("127.0.0.1", "localhost") and not s.api_key:
        out.append(f"host {s.host} serves other machines without an api_key: set one, or host = \"127.0.0.1\"")
    return out
