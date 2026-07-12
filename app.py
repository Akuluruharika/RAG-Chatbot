import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Flick AI",
    page_icon=" ",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------
col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("""
    <h1 style='text-align:center;'>
      Flick AI
    </h1>
   
    """, unsafe_allow_html=True)

with col2:
    dark_mode = st.toggle(" ")

# -----------------------------
# THEME
# -----------------------------
if dark_mode:

    st.markdown("""
    <style>

    .stApp {
        background-color: #0E1117;
        color: white;
    }

    h1, h2, h3, p, div, label {
        color: white !important;
    }

    .stTextInput input {
        color: white !important;
    }

    </style>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <style>

    .stApp {
        background-color: white;
        color: black;
    }

    h1, h2, h3, p, div, label {
        color: black !important;
    }

    .stTextInput input {
        color: black !important;
    }

    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# PDF UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    " Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    try:

        # Create folders
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("chroma_db", exist_ok=True)

        # Save PDF
        pdf_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(" PDF Uploaded Successfully")

        # -----------------------------
        # LOAD PDF
        # -----------------------------
        with st.spinner("Loading PDF..."):

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

        st.success(
            f" PDF Loaded ({len(documents)} pages)"
        )

        # -----------------------------
        # SPLIT TEXT
        # -----------------------------
        with st.spinner("Creating chunks..."):

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = splitter.split_documents(
                documents
            )

        st.success(
            f" Created {len(docs)} chunks"
        )

        # -----------------------------
        # EMBEDDINGS
        # -----------------------------
        with st.spinner("Generating embeddings..."):

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        st.success("Embeddings Ready")

        # -----------------------------
        # VECTOR DATABASE
        # -----------------------------
        with st.spinner(
            "Creating Vector Database..."
        ):

            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

        st.success(
            " Vector Database Created"
        )

        # -----------------------------
        # QUESTION INPUT
        # -----------------------------
        query = st.text_input(
            " Ask a question about the PDF"
        )

        if query:

            with st.spinner(
                "Searching document..."
            ):

                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )

                relevant_docs = retriever.invoke(
                    query
                )

                context = "\n\n".join(
                    [
                        doc.page_content
                        for doc in relevant_docs
                    ]
                )

            # -----------------------------
            # MISTRAL
            # -----------------------------
            llm = ChatOllama(
                model="mistral"
            )

            prompt = f"""
You are Flick AI.

Answer ONLY using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

            with st.spinner(
                "Generating answer..."
            ):

                response = llm.invoke(
                    prompt
                )

            # -----------------------------
            # DISPLAY ANSWER
            # -----------------------------
            st.markdown(
                "## AI Response"
            )

            if hasattr(
                response,
                "content"
            ):
                st.success(
                    response.content
                )
            else:
                st.success(
                    str(response)
                )

            # -----------------------------
            # SOURCES
            # -----------------------------
            st.markdown(
                "## Retrieved Sources"
            )

            for i, doc in enumerate(
                relevant_docs
            ):

                with st.expander(
                    f"Source {i+1}"
                ):
                    st.write(
                        doc.page_content
                    )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )