#!/usr/bin/env bash
#
# run.sh: the server, its browser test page and the OpenAI-style front together, in the foreground. Ctrl-C stops
# all three.
#
# Knobs (env): PORT (30117, the server's own API), UI_PORT (30118, the page), FRONT_PORT (30119, the front that
#   OpenAI text-to-speech clients such as Hermes point at), UI_BIND and FRONT_BIND (127.0.0.1; 0.0.0.0 lets other
#   machines reach the page or the front, neither of which has a password), FRONT_VOICE (default: the voice for
#   names the server does not have), and serve.sh's own (NAME, MODELS, TF_SRC).
#
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PORT="${PORT:-30117}"
UI_PORT="${UI_PORT:-30118}"
UI_BIND="${UI_BIND:-127.0.0.1}"
FRONT_PORT="${FRONT_PORT:-30119}"
FRONT_BIND="${FRONT_BIND:-127.0.0.1}"
TF_SRC="${TF_SRC:-$PWD/engine}"
[ -x "$TF_SRC/zig-out/native/bin/tensorfold-native" ] || { echo "no server yet: run ./setup.sh first" >&2; exit 1; }

server="" page="" front=""
stop() {
  trap - EXIT INT TERM
  [ -n "$page" ] && kill "$page" 2>/dev/null
  [ -n "$front" ] && kill "$front" 2>/dev/null
  [ -n "$server" ] && kill "$server" 2>/dev/null
  wait 2>/dev/null
}
trap stop EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

HOST=127.0.0.1 PORT="$PORT" ./serve.sh &
server=$!
until curl -s -m 2 -o /dev/null "127.0.0.1:$PORT/v1/models"; do
  kill -0 "$server" 2>/dev/null || { echo "the server stopped before it was ready" >&2; exit 1; }
  sleep 1
done
python3 "$TF_SRC/tools/zig/zonos3_test_page.py" --upstream "127.0.0.1:$PORT" --bind "$UI_BIND" --port "$UI_PORT" &
page=$!
python3 tools/openai_front.py --upstream "127.0.0.1:$PORT" --bind "$FRONT_BIND" --port "$FRONT_PORT" --voice "${FRONT_VOICE:-default}" &
front=$!
command -v ffmpeg >/dev/null || echo "no ffmpeg on this machine: the front will answer WAV and PCM only, not MP3 or Opus" >&2
echo "ZONOS3 Mini is up (Ctrl-C stops it):"
echo "  test page                 http://$UI_BIND:$UI_PORT/"
echo "  for OpenAI speech clients http://$FRONT_BIND:$FRONT_PORT/v1"
echo "  the server's own API      http://127.0.0.1:$PORT/v1"
wait -n "$server" "$page" "$front"
echo "one of the three stopped; stopping the others" >&2
exit 1
