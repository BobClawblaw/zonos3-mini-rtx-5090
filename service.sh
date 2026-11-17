#!/usr/bin/env bash
#
# service.sh: optional. Keep run.sh running as a systemd user service (no root): it starts at login and restarts if
# it fails. Without this, ./run.sh in a terminal is all you need.
#
#   ./service.sh install    write and start zonos3-mini.service, with the PORT, UI_PORT and UI_BIND set now
#   ./service.sh remove     stop it and delete the unit
#   ./service.sh status | logs
#
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
unit="$HOME/.config/systemd/user/zonos3-mini.service"
case "${1:-}" in
  install)
    mkdir -p "$(dirname "$unit")"
    cat > "$unit" <<UNIT
[Unit]
Description=ZONOS3 Mini text to speech (TensorFold) and its test page

[Service]
WorkingDirectory=$PWD
Environment=PORT=${PORT:-30117} UI_PORT=${UI_PORT:-30118} UI_BIND=${UI_BIND:-127.0.0.1}
ExecStart=$PWD/run.sh
Restart=on-failure
SuccessExitStatus=130 143
RestartSec=10

[Install]
WantedBy=default.target
UNIT
    systemctl --user daemon-reload
    systemctl --user enable --now zonos3-mini.service
    echo "installed: $unit (to keep it running after you log out: loginctl enable-linger $USER)" ;;
  remove)
    systemctl --user disable --now zonos3-mini.service 2>/dev/null || true
    rm -f "$unit"
    systemctl --user daemon-reload
    systemctl --user reset-failed zonos3-mini.service 2>/dev/null || true
    echo "removed" ;;
  status) systemctl --user status zonos3-mini.service ;;
  logs) journalctl --user -u zonos3-mini.service -n 50 --no-pager ;;
  *) sed -n '3,10p' "$0"; exit 1 ;;
esac
