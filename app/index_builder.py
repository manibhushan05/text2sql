import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient
from llama_settings import get_service_context

def load_charter_json(path="charter1.json"):
    with open(path) as f:
        metadata = json.load(f)

    docs = []
    for table, table_info in metadata["tables"].items():
        doc_chunks = [f"Table: {table}", f"Description: {table_info['description']}"]
        for col_name, col_info in table_info["columns"].items():
            doc_chunks.append(f"Column: {col_name} ({col_info['type']}) - {col_info['description']}")
        docs.append("\n".join(doc_chunks))

    for rel in metadata.get("relationships", []):
        docs.append(f"Foreign Key: {rel['from_table']}.{rel['from_column']} → {rel['to_table']}.{rel['to_column']}")

    for q in metadata.get("sample_queries", []):
        docs.append(f"Example Query: {q['description']} → {q['query']}")

    return [Document(text=d) for d in docs]

def build_index():
    get_service_context()
    documents = load_charter_json()
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    chroma_path = base_dir / "storage"
    chroma_client = PersistentClient(path=str(chroma_path))
    vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("charter_meta"))

    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
    index.storage_context.persist()
    print("✅ Index built and persisted.")
