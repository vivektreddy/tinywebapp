import time
from pinecone import Pinecone
from openai import OpenAI
from config import settings
import telemetry
from telemetry import get_tracer


class RAGSearcher:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX)
        self.oai = OpenAI(api_key=settings.OPENAI_API_KEY)

    def search(self, search_query):
        tracer = get_tracer()
        t_rag = time.monotonic()

        with tracer.start_as_current_span("openai.embeddings", attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "text-embedding-ada-002",
            "rag.query": search_query,
        }) as emb_span:
            resp = self.oai.embeddings.create(model="text-embedding-ada-002", input=search_query)
            embedding = resp.data[0].embedding
            emb_span.set_attribute("gen_ai.usage.input_tokens", getattr(resp.usage, "prompt_tokens", 0))

        with tracer.start_as_current_span("pinecone.query", attributes={
            "db.system": "pinecone",
            "db.name": settings.PINECONE_INDEX,
            "db.vector.query.top_k": settings.PINECONE_FETCH_K,
        }) as pine_span:
            results = self.index.query(
                vector=embedding,
                top_k=settings.PINECONE_FETCH_K,
                include_metadata=True,
            )
            pine_span.set_attribute("pinecone.result_count", len(results.matches))
            pine_span.set_attribute(
                "pinecone.top_score",
                round(results.matches[0].score, 4) if results.matches else 0.0,
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
        telemetry.rag_latency_histo.record(int((time.monotonic() - t_rag) * 1000))
        return [{"excerpt": m["excerpt"], "title": m["title"], "url": m["url"]} for m in matches]
