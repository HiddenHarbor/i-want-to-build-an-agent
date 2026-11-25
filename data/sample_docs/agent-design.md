# A small agent design

An AI agent combines a language model with tools and a loop that decides what to do next. A useful first project should have a narrow job, a clear success condition, and a small set of allowed actions.

For a document assistant, retrieval is the tool: the system embeds a question, finds relevant passages in a vector index, and asks a model to answer from those passages. Showing the source documents makes the answer easier to verify.

## RAG workflow

1. Load documents and extract their text.
2. Split text into overlapping chunks so important context near boundaries is retained.
3. Convert each chunk to an embedding vector.
4. Store vectors, text, and source metadata in a vector database.
5. Embed a user question and retrieve the closest chunks by cosine similarity.
6. Give the retrieved context to the language model and return an answer with source names.

## Quality and safety

Retrieval quality depends on document quality, chunk size, embedding model, and the question. The assistant should admit when the sources do not contain an answer. Treat retrieved text as untrusted input: it may contain misleading instructions, so the model should use it as evidence rather than follow it as policy.

## Cost-conscious development

Running the model, embeddings, and vector database locally avoids metered cloud services and API keys. The tradeoff is that model speed and capacity depend on the developer's computer. Small models are a practical starting point for a portfolio demo.
