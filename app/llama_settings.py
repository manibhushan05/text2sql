import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

def get_service_context():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not found in environment variables")

    os.environ["GOOGLE_API_KEY"] = api_key

    llm = GoogleGenAI(model="gemini-2.0-flash")
    embed_model = GoogleGenAIEmbedding(model="text-embedding-004")
    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model
