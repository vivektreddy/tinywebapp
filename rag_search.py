from pinecone import Pinecone
import os
from openai import OpenAI
from config import settings

class RAGSearcher:
    def __init__(self):
        self.pc = Pinecone(api_key = settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX)
        self.oai = OpenAI(api_key = settings.OPENAI_API_KEY)

    def search(self, search_query):

        embedding = self.oai.embeddings.create(model = "text-embedding-ada-002", \
        input = search_query).data[0].embedding
        results = self.index.query(vector = embedding, \
        top_k = settings.PINECONE_TOP_K, \
        include_metadata = True)
        

        matches = []
        for res in results.matches:
            meta = res.metadata or {}

            matches.append({
                "excerpt": meta.get("text", ""),
                "title": meta.get("program_name") or meta.get("section"),
                "url": meta.get("url", ""),
            })

        return matches
