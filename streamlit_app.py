from pathlib import Path
import runpy
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "ocean_dashboard"
sys.path.insert(0, str(APP_DIR))

# Cloud-only credentials come from Streamlit Secrets.
try:
    key = str(st.secrets.get("ROBOFLOW_API_KEY", "")).strip()
except Exception:
    key = ""

if key:
    keyfile = Path.home() / ".config/ocean-ai/roboflow_api_key"
    keyfile.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    keyfile.write_text(key)
    keyfile.chmod(0o600)

runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
