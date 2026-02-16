import time
import json
from datetime import timezone
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder  # <-- добавили
from schema import EventModel
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")


def to_ts(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def event_to_doc(e: EventModel) -> Document:
    text = f"{e.title}\nКатегория: {e.category}\n{e.summary or ''}".strip()
    ts = to_ts(e.date)
    if ts is None:
        ts = 0
    return Document(
        page_content=text,
        metadata={
            "event_id": e.event_id,
            "event_ts": ts,
            "title": e.title,
            "category": e.category,
        },
    )

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": DEVICE, "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)



events = []
with open("events.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "summary" in d and isinstance(d["summary"], str):
            d["summary"] = d["summary"][:120]  # под лимит схемы
        events.append(EventModel(**d))

docs = [event_to_doc(e) for e in events]
ids = [e.event_id for e in events]

vectorstore = Chroma(
    collection_name="events",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

if vectorstore._collection.count() == 0:
    vectorstore.add_documents(docs, ids=ids)
    vectorstore.persist()

bm25 = BM25Retriever.from_documents(docs)
bm25.k = 20

query = "Хочу найти девушку, куда бы сходить познакомиться"
now_ts = int(time.time())

dense_docs = vectorstore.similarity_search(query, k=50)
dense_docs = [d for d in dense_docs if d.metadata.get("event_ts", 0) >= now_ts]
dense_docs = dense_docs[:20]

bm25_docs = bm25.invoke(query)
bm25_docs = [d for d in bm25_docs if d.metadata.get("event_ts", 0) >= now_ts]

# merge + dedupe
seen = set()
merged = []
for d in dense_docs + bm25_docs:
    eid = d.metadata.get("event_id")
    if not eid or eid in seen:
        continue
    merged.append(d)
    seen.add(eid)

# -------------------------
# RERANK (Cross-Encoder, RU-friendly, lightweight)
# -------------------------
# Берём кандидатов побольше, чем top5, чтобы реранкер имел смысл
candidates = merged[:30]  # можно 20-50

reranker = CrossEncoder("DiTy/cross-encoder-russian-msmarco", device=DEVICE)
pairs = [(query, d.page_content) for d in candidates]
scores = reranker.predict(pairs)

ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
top5 = [doc for doc, _ in ranked[:5]]

event_ids = [d.metadata["event_id"] for d in top5]
print(event_ids)
