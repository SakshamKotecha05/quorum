"""Load a .env if one is present, before any module reads the environment.

ponytail: 8 lines of stdlib instead of python-dotenv. It handles KEY=value, comments
and blank lines, which is the whole of what this project needs. Real env vars win, so
an export always overrides the file.
"""
import os
from pathlib import Path

for _line in (Path(__file__).resolve().parents[1] / ".env").read_text().splitlines() \
        if (Path(__file__).resolve().parents[1] / ".env").exists() else []:
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))
