#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
    echo 'Python environment missing. Follow the setup in README.md first.'
    read -r -p 'Press Return to close. '
    exit 1
fi
mkdir -p .eval
export QUORUM_DB="$PWD/.eval/dashboard.db"
.venv/bin/python - <<'PY'
import json
import os
import shlex
from pathlib import Path
import threading
import time
import urllib.request
import webbrowser

import uvicorn

# Read simple KEY=value entries without executing shell code or exposing secrets.
config = Path('.env')
if config.exists():
    for line in config.read_text().splitlines():
        parts = shlex.split(line, comments=True)
        if parts and parts[0] == 'export':
            parts = parts[1:]
        if len(parts) == 1 and '=' in parts[0]:
            name, value = parts[0].split('=', 1)
            if name == 'GROQ_API_KEY' or name.startswith('QUORUM_'):
                os.environ.setdefault(name, value)

url = 'http://127.0.0.1:8765'

def available():
    try:
        with urllib.request.urlopen(url + '/demo/measurements', timeout=1) as response:
            data = json.load(response)
        return 'benchmark' in data and 'verification' in data
    except (OSError, ValueError):
        return False

if available():
    webbrowser.open(url + '/#architecture')
    print('Quorum is already running. Opened the dashboard in your browser.')
else:
    def open_when_ready():
        for _ in range(100):
            if available():
                webbrowser.open(url + '/#architecture')
                return
            time.sleep(0.1)
    threading.Thread(target=open_when_ready, daemon=True).start()
    print('Opening Quorum in your browser. Keep this window open in the background.')
    print('Stop the local server with Control-C after your demo.')
    uvicorn.run('quorum.api:app', host='127.0.0.1', port=8765)
PY
