"""The systemd user unit: `up --foreground` at login, restarted on failure, with this machine's paths and the settings
in force when it is written (nothing machine-specific is committed). Enabling it is left to the user:
systemctl --user enable --now zonos3-mini.service"""

import shutil
from pathlib import Path

from .config import ROOT, Settings

UNIT = "zonos3-mini.service"


def unit(s: Settings) -> str:
    python = shutil.which("python3") or "/usr/bin/python3"
    return f"""[Unit]
Description=ZONOS3 Mini text to speech (TensorFold) and its test page

[Service]
WorkingDirectory={ROOT}
Environment=HOST={s.host} PORT={s.port} PAGE_PORT={s.page_port} PAGE_BIND={s.page_bind}
ExecStart={python} -m rtx up --foreground
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
"""


def write(s: Settings) -> str:
    path = Path.home() / ".config" / "systemd" / "user" / UNIT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(unit(s))
    return f"wrote {path} (not enabled): systemctl --user daemon-reload && systemctl --user enable --now {UNIT}"
