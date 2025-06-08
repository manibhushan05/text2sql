from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .index_builder import build_index
from .query_engine import generate_sql

app = FastAPI(title="Text-to-SQL API with Gemini + RAG")


class QueryRequest(BaseModel):
    question: str


class SQLResponse(BaseModel):
    sql: str


@app.on_event("startup")
async def startup_event():
    build_index()


@app.post("/generate_sql", response_model=SQLResponse)
def generate_sql_endpoint(request: QueryRequest):
    try:
        sql_query = generate_sql(request.question)
        return {"sql": sql_query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
