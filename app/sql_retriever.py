from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

def get_sql_context_from_query(user_query: str):
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    chroma_path = base_dir / "storage"
    storage_context = StorageContext.from_defaults(persist_dir=str(chroma_path))
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=3)

    nodes = retriever.retrieve(user_query)
    return "\n".join([node.text for node in nodes])
