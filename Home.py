import os
from pathlib import Path

import streamlit as st

cwd = os.getcwd()

st.set_page_config(page_title="EV Placement", page_icon=":zap:")


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


# Display Problem Introduction from markdown file
intro_markdown = read_markdown_file("README.md")
st.markdown(intro_markdown, unsafe_allow_html=True)
