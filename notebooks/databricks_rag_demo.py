# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks document retrieval demo
# MAGIC
# MAGIC This notebook stores sample passages in a Delta table, indexes them
# MAGIC with Databricks AI Search, and retrieves relevant passages.
# MAGIC It uses sample text only and does not generate an AI-written answer.

# COMMAND ----------

print("Current catalog:", spark.sql("SELECT current_catalog()").first()[0])
print("Current schema:", spark.sql("SELECT current_schema()").first()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create a small sample Delta table
# MAGIC
# MAGIC This cell overwrites only the demo table named below.

# COMMAND ----------

rows = [
    (
        "passage-001",
        "agent-design.md",
        "For a document assistant, retrieval finds relevant passages in a vector index. The language model can then use those passages to answer a question, and source names help people verify the answer.",
    ),
    (
        "passage-002",
        "agent-design.md",
        "A RAG workflow loads documents, splits them into overlapping chunks, converts each chunk into an embedding vector, stores the vectors with text and metadata, retrieves the closest passages for a question, and provides those passages to a language model.",
    ),
    (
        "passage-003",
        "demo-notes.txt",
        "A useful document search stores every passage with its source filename. It should return relevant passages even when the user's question uses different words from the document.",
    ),
]

documents_df = spark.createDataFrame(rows, ["id", "source", "text"])

documents_df.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.default.rag_demo_documents"
)

display(spark.table("workspace.default.rag_demo_documents"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Install the Databricks AI Search SDK
# MAGIC
# MAGIC Run this cell once. Databricks restarts the Python session afterward.

# COMMAND ----------

# MAGIC %pip install databricks-ai-search
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create the AI Search endpoint
# MAGIC
# MAGIC Run this setup cell once per workspace. Free Edition allows one AI Search endpoint.

# COMMAND ----------

from databricks.ai_search.client import AISearchClient

ai_search = AISearchClient()

ai_search.create_endpoint(
    name="rag-demo-endpoint",
    endpoint_type="STANDARD",
)

print("Requested the rag-demo-endpoint endpoint.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Enable table change tracking
# MAGIC
# MAGIC Standard AI Search endpoints require change data feed on the source table.

# COMMAND ----------

spark.sql("""
ALTER TABLE workspace.default.rag_demo_documents
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

print("Change data feed enabled.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create the Delta Sync index
# MAGIC
# MAGIC Run this setup cell once. It uses Databricks' managed Qwen embedding model.

# COMMAND ----------

index = ai_search.create_delta_sync_index(
    endpoint_name="rag-demo-endpoint",
    source_table_name="workspace.default.rag_demo_documents",
    index_name="workspace.default.rag_demo_index",
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column="text",
    embedding_model_endpoint_name="databricks-qwen3-embedding-0-6b",
)

print(index.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Check index readiness
# MAGIC
# MAGIC If the index is still provisioning, rerun this cell later.
# MAGIC Continue to the search cell when `ready` is `True`.

# COMMAND ----------

index = ai_search.get_index(
    endpoint_name="rag-demo-endpoint",
    index_name="workspace.default.rag_demo_index",
)

print(index.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Search the index
# MAGIC
# MAGIC The results include the matching text and its source filename.

# COMMAND ----------

results = index.similarity_search(
    query_text="How can I tell which file a passage came from?",
    columns=["id", "source", "text"],
    num_results=2,
)

print(results)