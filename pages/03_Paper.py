import streamlit as st

from ev_station_solver.helper_functions import get_pdf

# aper
st.set_page_config(page_title="EV Placement - Paper", page_icon=":closed_book:")

st.title("Paper")

pdf_file_name = "Placement_EV_Chargers.pdf"

with open(pdf_file_name, "rb") as f:
    st.download_button(label="Download", data=f, file_name=pdf_file_name)

st.markdown(get_pdf(pdf_file_name), unsafe_allow_html=True)
