#!/usr/bin/env bash
#
# run.sh: the server and its browser test page together, in the foreground. Ctrl-C stops both.
#
# Knobs (env): PORT (30117, the API), UI_PORT (30118, the page), UI_BIND (127.0.0.1; 0.0.0.0 lets other machines
#   open the page, which has no password), and serve.sh's own (NAME, MODELS, TF_SRC).
#
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PORT="${PORT:-30117}"
UI_PORT="${UI_PORT:-30118}"
UI_BIND="${UI_BIND:-127.0.0.1}"
TF_SRC="${TF_SRC:-$PWD/engine}"
[ -x "$TF_SRC/zig-out/native/bin/tensorfold-native" ] || { echo "no server yet: run ./setup.sh first" >&2; exit 1; }

server="" page=""
stop() {
  trap - EXIT INT TERM
  [ -n "$page" ] && kill "$page" 2>/dev/null
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
echo "ZONOS3 Mini is up: the test page is http://$UI_BIND:$UI_PORT/ and the API http://127.0.0.1:$PORT/v1 (Ctrl-C stops both)"
wait -n "$server" "$page"
echo "one of the two stopped; stopping the other" >&2
exit 1
