"""Chunk, embed, and upsert raw_pages.jsonl into Pinecone."""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import tiktoken
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

INPUT = Path(__file__).parent / "data" / "raw_pages.jsonl"
PROGRESS = Path(__file__).parent / "data" / "embed_progress.json"

CHUNK_TOKENS = 500
BATCH_SIZE = 100
EMBED_MODEL = "text-embedding-ada-002"
PINECONE_DIM = 1536

SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def chunk_text(text: str, section: str, enc, max_tokens: int = CHUNK_TOKENS, overlap_sentences: int = 1) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    prefix = f"[{section}] "
    prefix_tokens = len(enc.encode(prefix))

    chunks, current, current_tokens = [], [], prefix_tokens
    for sent in sentences:
        sent_tokens = len(enc.encode(sent + " "))
        if current and current_tokens + sent_tokens > max_tokens:
            chunks.append(prefix + " ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_tokens = prefix_tokens + sum(len(enc.encode(s + " ")) for s in current)
        current.append(sent)
        current_tokens += sent_tokens
    if current:
        chunks.append(prefix + " ".join(current))
    return chunks


def chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_progress() -> set:
    if PROGRESS.exists():
        return set(json.loads(PROGRESS.read_text()))
    return set()


def save_progress(done: set):
    PROGRESS.write_text(json.dumps(list(done)))


def main():
    oai = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX"]

    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=PINECONE_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")),
        )
        print(f"Created Pinecone index: {index_name}")

    index = pc.Index(index_name)
    enc = tiktoken.get_encoding("cl100k_base")
    done_ids = load_progress()

    records = [json.loads(l) for l in INPUT.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(records)} records from {INPUT}")

    vectors = []
    for rec in records:
        chunks = chunk_text(rec["text"], rec["section"], enc)
        for chunk in chunks:
            cid = chunk_id(chunk)
            if cid in done_ids:
                continue
            vectors.append({
                "id": cid,
                "text": chunk,
                "metadata": {
                    "url": rec["url"],
                    "section": rec["section"],
                    "program_name": rec["program_name"],
                    "last_updated": rec["scraped_at"],
                    "text": chunk,
                },
            })

    print(f"{len(vectors)} chunks to embed (skipping {len(done_ids)} already done)")

    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i: i + BATCH_SIZE]
        texts = [v["text"] for v in batch]
        resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
        embeddings = [e.embedding for e in resp.data]

        upsert_data = [
            (v["id"], emb, v["metadata"])
            for v, emb in zip(batch, embeddings)
        ]
        index.upsert(vectors=upsert_data)

        for v in batch:
            done_ids.add(v["id"])
        save_progress(done_ids)

        print(f"  Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} vectors)")
        time.sleep(0.5)

    print(f"\nDone. Total vectors in index: ~{len(done_ids)}")


if __name__ == "__main__":
    main()
