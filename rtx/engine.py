"""The engine: a worktree of the TensorFold checkout at the pinned commit, built under tf_cache (never in the
checkout). The family needs no libtfgrammar: it speaks, with no tools or schemas."""

import subprocess
from pathlib import Path

from .config import INCLUDE, Settings

# where tensorfold_sha lives until the family reaches TensorFold's main: a tag, since the PR branch is rebased
TENSORFOLD_REPO = "https://github.com/BobClawblaw/TensorFold"
TENSORFOLD_REF = "refs/tags/zonos3-openai-9757d2378"


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
    return s.built_binary
