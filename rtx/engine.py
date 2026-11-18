"""The engine: a worktree of the TensorFold checkout at the pinned commit, built under tf_cache (never in the
checkout). The family needs no libtfgrammar: it speaks, with no tools or schemas."""

import os
import re
import shutil
import subprocess
from pathlib import Path

from .config import INCLUDE, TEXTNORM_REPO, TEXTNORM_RUST, TEXTNORM_SHA, Settings

# where tensorfold_sha lives until the family reaches TensorFold's main: a tag, since the PR branch is rebased
TENSORFOLD_REPO = "https://github.com/BobClawblaw/TensorFold"
TENSORFOLD_REF = "refs/tags/zonos3-openai-a158fcdaf"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def fetch(s: Settings) -> Path:
    """The release at its pinned revision: the 15 GB the server reads, without its GGUF and web builds."""
    _run(["hf", "download", s.model_repo, "--revision", s.model_revision, "--local-dir", str(s.model), "--include", *INCLUDE])
    return s.model


def build(s: Settings, rebuild: bool = False) -> Path:
    if s.binary:
        return s.run_binary
    if not s.tensorfold_sha:
        raise SystemExit("no tensorfold_sha pinned (local.toml)")
    src = s.path(s.tensorfold_src)
    if not (src / ".git").exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--filter=blob:none", TENSORFOLD_REPO, str(src)])
    have = subprocess.run(["git", "-C", str(src), "cat-file", "-e", f"{s.tensorfold_sha}^{{commit}}"], capture_output=True)
    if have.returncode != 0:  # a clone older than the pin
        _run(["git", "-C", str(src), "fetch", TENSORFOLD_REPO, TENSORFOLD_REF])
    if not s.worktree.exists():
        s.worktree.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "-C", str(src), "worktree", "add", "--detach", str(s.worktree), s.tensorfold_sha])
    if rebuild or not s.built_binary.exists():
        _run([s.path(s.zig).as_posix(), "build", "native", "-Doptimize=ReleaseFast", f"-Dnvcc={s.path(s.nvcc)}", f"-Dsm={s.sm}"], cwd=s.worktree)
    textnorm(s, rebuild)
    return s.built_binary


def _rust(cargo: str) -> tuple[int, int] | None:
    try:
        out = subprocess.run([cargo, "--version"], capture_output=True, text=True).stdout
    except OSError:
        return None
    m = re.match(r"cargo (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def textnorm(s: Settings, rebuild: bool = False) -> Path | None:
    """Zyphra's text normalizer as a shared library in the engine's lib/. Without Rust the engine still serves:
    text is then spoken as typed."""
    if s.textnorm_lib.exists() and not rebuild:
        return s.textnorm_lib
    cargo = s.path(s.cargo).as_posix() if "/" in s.cargo else s.cargo
    have, need = _rust(cargo), ".".join(map(str, TEXTNORM_RUST))
    if have is None or have < TEXTNORM_RUST:
        found = "no cargo" if have is None else f"cargo {have[0]}.{have[1]}"
        print(f"note: text normalization not built ({found}; it needs Rust {need} or later: set cargo in local.toml and run build again). "
              "Numbers, dates and currency are spoken as typed until then.")
        return None
    src = s.textnorm_src
    if not (src / ".git").exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", TEXTNORM_REPO, str(src)])
    _run(["git", "-C", str(src), "checkout", "--detach", TEXTNORM_SHA])
    cmd = [cargo, "rustc", "--release", "--locked", "--lib", "-p", "zyphra-textnorm", "--features", "capi", "--crate-type", "cdylib"]
    print("+", " ".join(cmd), flush=True)
    # a cargo named by its path runs the rustc beside it, not an older one earlier on PATH
    env = {**os.environ, "PATH": os.path.dirname(cargo) + os.pathsep + os.environ.get("PATH", "")} if "/" in cargo else None
    if subprocess.run(cmd, cwd=src, env=env).returncode != 0:
        print("note: the text normalizer did not build (see above); numbers, dates and currency are spoken as typed until it does.")
        return None
    s.textnorm_lib.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "target" / "release" / "libzyphra_textnorm.so", s.textnorm_lib)
    return s.textnorm_lib
