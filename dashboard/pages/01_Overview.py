"""Overview page (Grok Build dashboard pages).

This will become the landing page when full multipage navigation is wired.
For now the primary logic lives in dashboard/app.py with a radio selector.
Keeping this file ensures the pages/ layout matches the spec.
"""
import streamlit as st
from pathlib import Path

st.header("Overview (multipage stub)")
st.write("Switch to the main `streamlit run dashboard/app.py` for the current MVP UI.")
st.caption(str(Path.cwd()))
