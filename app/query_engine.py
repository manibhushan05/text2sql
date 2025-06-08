# query_engine.py
from .llama_settings import get_service_context
from .sql_retriever import get_sql_context_from_query
from llama_index.core.llms import ChatMessage


def generate_sql(user_question: str):
    llm, _ = get_service_context()
    context = get_sql_context_from_query(user_question)

    system_prompt = (
        "You are a SQL analyst. Based on the provided schema context, "
        "generate a valid PostgreSQL SQL query for the user's question. "
        "Don't explain — only output the query."
    )

    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser Question: {user_question}\n\nSQL:"

    response = llm.chat([ChatMessage(role="user", content=prompt)])
    return response.message.content.strip()
