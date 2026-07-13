import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Flick AI",
    page_icon=" ",
    layout="wide"
)
col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("""
    <h1 style='text-align:center;'>
     Flick AI
    </h1>
    """, unsafe_allow_html=True)

with col2:
    dark_mode = st.toggle("🌙")

# -----------------------------
# THEME
# -----------------------------
if dark_mode:

    st.markdown("""
    <style>

    .stApp {
        background-color:#0E1117;
        color:white;
    }

    h1,h2,h3,p,div,label{
        color:white !important;
    }

    .stTextInput input{
        color:white !important;
    }

    </style>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <style>

    .stApp{
        background:white;
        color:black;
    }

    h1,h2,h3,p,div,label{
        color:black !important;
    }

    .stTextInput input{
        color:black !important;
    }

    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# PDF UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    try:

        os.makedirs(
            "uploads",
            exist_ok=True
        )

        os.makedirs(
            "chroma_db",
            exist_ok=True
        )

        pdf_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(pdf_path,"wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )

        st.success(
            "✅ PDF Uploaded Successfully"
        )

        with st.spinner(
            "Loading PDF..."
        ):

            loader = PyPDFLoader(
                pdf_path
            )

            documents = loader.load()

        st.success(
            f"✅ PDF Loaded ({len(documents)} pages)"
        )

        with st.spinner(
            "Creating Chunks..."
        ):

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = splitter.split_documents(
                documents
            )

        st.success(
            f"✅ {len(docs)} Chunks Created"
        )

        with st.spinner(
            "Generating Embeddings..."
        ):

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        st.success(
            "✅ Embeddings Ready"
        )

        with st.spinner(
            "Creating Vector Database..."
        ):

            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

        st.success(
            "✅ Vector Database Created"
        )

        query = st.text_input(
            "💬 Ask a question about the PDF"
        )

        if query:

            with st.spinner(
                "Searching document..."
            ):

                retriever = vectorstore.as_retriever(
                    search_kwargs={
                        "k":3
                    }
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
                           
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=st.secrets["GEMINI_API_KEY"],
                temperature=0
            )

            prompt = f"""
You are Flick AI, an intelligent RAG-Based Document Assistant.

Use ONLY the information provided in the context below.

If the answer is not available in the context,
reply with:

"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{query}

Answer:
"""

            with st.spinner(
                "Generating Answer..."
            ):

                response = llm.invoke(
                    prompt
                )
            st.markdown(
                "##  AI Response"
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

            st.markdown(
                "##  Retrieved Sources"
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
            f" Error: {str(e)}"
        )
