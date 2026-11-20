"""Serve Neuron Studio as a self-contained Streamlit application."""

from base64 import b64encode
from pathlib import Path

import streamlit as st


assets = Path(__file__).parent / "dist"
page = (assets / "index.html").read_text(encoding="utf-8")
stylesheet = (assets / "style.css").read_text(encoding="utf-8")
page = page.replace('<link rel="stylesheet" href="style.css">', f"<style>{stylesheet}</style>")
page = page.replace('href="./"', 'href="#network"')

# Keep the scripts deferred and in their original order inside the iframe.
for filename in ("model.js", "app.js"):
    encoded = b64encode((assets / filename).read_bytes()).decode("ascii")
    page = page.replace(f'src="{filename}"', f'src="data:text/javascript;base64,{encoded}"')

st.set_page_config(page_title="Neuron Studio", layout="wide")
st.iframe(page)
