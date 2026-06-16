from pinecone import Pinecone
from openai import OpenAI
from config import settings

class RAGSearcher:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX)
        self.oai = OpenAI(api_key=settings.OPENAI_API_KEY)

    def search(self, search_query):
        embedding = self.oai.embeddings.create(
            model="text-embedding-ada-002", input=search_query
        ).data[0].embedding

        results = self.index.query(
            vector=embedding,
            top_k=settings.PINECONE_FETCH_K,
            include_metadata=True,
        )

        seen_urls = {}
        for res in results.matches:
            if res.score < settings.PINECONE_SCORE_THRESHOLD:
                continue
            meta = res.metadata or {}
            url = meta.get("url", "")
            if url in seen_urls and seen_urls[url]["score"] >= res.score:
                continue
            seen_urls[url] = {
                "score": res.score,
                "excerpt": meta.get("text", ""),
                "title": meta.get("program_name") or meta.get("section"),
                "url": url,
            }

        matches = sorted(seen_urls.values(), key=lambda m: m["score"], reverse=True)[:settings.PINECONE_TOP_K]
        return [{"excerpt": m["excerpt"], "title": m["title"], "url": m["url"]} for m in matches]
