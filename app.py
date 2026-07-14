import os
import streamlit as st

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Flick AI",
    page_icon=" ",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center;'> Flick AI</h1>
    <h4 style='text-align:center;color:gray;'>
    AI-Powered RAG Document Assistant
    </h4>
    """,
    unsafe_allow_html=True
)

st.write("---")

# -----------------------------
# PDF UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📄 Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    os.makedirs("uploads", exist_ok=True)

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF Uploaded Successfully")

    st.info(f" File Name: {uploaded_file.name}")

    st.info(f" File Size: {round(uploaded_file.size / 1024,2)} KB")

    st.write("### PDF is ready for processing.")
