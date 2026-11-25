from pathlib import Path

import streamlit as st

from rag import collection_stats, ingest_path, search_documents


st.set_page_config(
    page_title="Local Document Search",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Local Document Search")
st.write(
    "Add documents, index them locally, then search for relevant passages "
    "using vector search."
)

with st.sidebar:
    st.header("Knowledge base")

    uploaded_files = st.file_uploader(
        "Add PDF, Markdown, or text files",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Index uploaded files", type="primary"):
        upload_folder = Path("data/uploads")
        upload_folder.mkdir(parents=True, exist_ok=True)

        total_indexed = 0

        for uploaded_file in uploaded_files:
            destination = upload_folder / Path(uploaded_file.name).name
            destination.write_bytes(uploaded_file.getvalue())
            total_indexed += ingest_path(destination)

        st.success(f"Indexed {total_indexed} text chunks.")

    if st.button("Index the sample documents"):
        total_indexed = ingest_path(Path("data/sample_docs"))
        st.success(f"Indexed {total_indexed} text chunks.")

    st.metric("Indexed text chunks", collection_stats())

    st.caption(
        "Chroma creates local embeddings the first time you index or search. "
        "Your vector database is saved in data/chroma/."
    )

st.subheader("Search your documents")

question = st.text_input(
    "Enter a question or topic",
    placeholder="What are the steps in a RAG workflow?",
)

if st.button("Search", type="primary", disabled=not question.strip()):
    try:
        matches = search_documents(question)

        st.write(f"Found {len(matches)} relevant passages:")

        for number, match in enumerate(matches, start=1):
            st.markdown(
                f"**{number}. {match['source']}** "
                f"(similarity: {match['score']:.2f})"
            )
            st.info(match["text"])

    except Exception as error:
        st.error(str(error))

st.divider()
st.caption(
    "This demo shows document ingestion, embeddings, persistent vector storage, "
    "and semantic retrieval. It does not generate an AI-written answer."
)