# Local Document Search

A local document search demo built with Python, Streamlit, and ChromaDB. Add PDF, Markdown, or text files, index them, and search for relevant passages using vector search.

## What it demonstrates

- Reading PDF, Markdown, and plain text files
- Splitting documents into overlapping chunks
- Creating embeddings locally with ChromaDB's default embedding function
- Storing text, vectors, and source filenames in a persistent Chroma database
- Searching for relevant passages and showing their source files

## What it does not do

This version retrieves and displays relevant passages. It does not use a chat model to generate an answer.

The Streamlit app runs locally and uses ChromaDB. The Databricks notebook in `notebooks/databricks_rag_demo.py` demonstrates the same sample search flow using a Delta table and Databricks AI Search.

Databricks Free Edition is intended for personal learning and has fair-use limits. This notebook uses only sample text. It retrieves passages; it does not generate an AI-written answer.

## Cost and privacy

The local Streamlit app runs without a cloud service or API key. The optional Databricks notebook uses your Databricks Free Edition workspace, which has fair-use limits and is for personal, non-commercial learning. The notebook sends its sample passages to Databricks for embedding, so use only non-sensitive sample data there. Python packages and Chroma's embedding model are downloaded when setting up the project. The embedding model runs locally after download.

The local Chroma database is stored in `data/chroma/`. Uploaded files are stored in `data/uploads/`. These folders are excluded from Git so local data is not published by accident.

Do not put private documents in `data/sample_docs/`; those sample files are part of the project and can be published to GitHub.

## Requirements

- Python 3.13
- VS Code
- An internet connection for installing packages and downloading Chroma's embedding model the first time

Sentence Transformers, PyTorch, Ollama, and cloud services are not required.

## Run locally on macOS

Open this folder in VS Code, then open **Terminal → New Terminal**.

Create and activate the project environment:

```bash
python3 -m venv .venv
source .venv/bin/activate