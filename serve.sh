#!/usr/bin/env bash
#
# serve.sh: serve Zyphra's ZONOS3 Mini text to speech on TensorFold native (CUDA) on one RTX 5090. Foreground.
#
# Knobs (env): PORT (30117), HOST (127.0.0.1), NAME (zonos3-mini), MODELS (./models), TF_SRC (./engine)
# The server speaks one request at a time; a request that arrives while another is spoken gets 503.
#
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODELS="${MODELS:-$PWD/models}"
TF_SRC="${TF_SRC:-$PWD/engine}"
exec "$TF_SRC/zig-out/native/bin/tensorfold-native" serve "$MODELS/ZONOS3-Mini" \
  --name "${NAME:-zonos3-mini}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-30117}" \
  --parallel 1 --no-update-check
