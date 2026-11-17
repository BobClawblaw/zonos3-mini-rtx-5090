#!/usr/bin/env bash
#
# setup.sh: download the release at its pinned revision, check out the engine at its pinned commit and build it for
# sm_120. Idempotent; rerun after changing recipe.toml.
#
# Env: MODELS (./models), TF_SRC (./engine), ZIG (zig on PATH, 0.17.0), NVCC (/usr/local/cuda/bin/nvcc)
#
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
pin() { python3 -c "import tomllib, sys; print(tomllib.load(open('recipe.toml', 'rb'))[sys.argv[1]][sys.argv[2]])" "$@"; }

MODELS="${MODELS:-$PWD/models}"
TF_SRC="${TF_SRC:-$PWD/engine}"
ZIG="${ZIG:-zig}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"

"$ZIG" version | grep -qx "$(pin engine zig)" || { echo "need Zig $(pin engine zig) (ZIG=...)" >&2; exit 1; }

# the files the server reads (about 15 GB); the release's GGUF and web builds (14 GB more) are left out
hf download "$(pin model repo)" --revision "$(pin model revision)" --local-dir "$MODELS/ZONOS3-Mini" \
  --include "lm/*" "autoencoder/*" "registries/*" "gguf/voices/*" "speaker_encoders/*" "README.md"

[ -d "$TF_SRC/.git" ] || git clone --branch "$(pin engine branch)" "$(pin engine repo)" "$TF_SRC"
git -C "$TF_SRC" fetch -q origin "$(pin engine branch)"
git -C "$TF_SRC" checkout -q "$(pin engine sha)"
SM="$(pin engine sm)"   # read here: recipe.toml is not in $TF_SRC, and an empty -Dsm builds for nvcc's default arch
(cd "$TF_SRC" && "$ZIG" build native -Dnvcc="$NVCC" -Dsm="$SM" -Doptimize=ReleaseFast)
echo "built $TF_SRC/zig-out/native/bin/tensorfold-native at $(git -C "$TF_SRC" rev-parse --short HEAD)"
