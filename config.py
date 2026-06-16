import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class BedrockModel(str, Enum):
    SONNET = "us.anthropic.claude-sonnet-4-6"
    HAIKU = "anthropic.claude-haiku-4-5-20251001-v1:0"
    OPENAI_EMBEDDING = "text-embedding-ada-002"

class Settings:
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    DEFAULT_MODEL = BedrockModel.SONNET
    DEFAULT_RERANK_MODEL = ""
    PINECONE_FETCH_K = 20
    PINECONE_TOP_K = 6
    PINECONE_SCORE_THRESHOLD = 0.75
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

settings = Settings()

