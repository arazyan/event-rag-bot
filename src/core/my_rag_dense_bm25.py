import logging
import os
import time
import json
from datetime import timezone
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import torch

from src.utils.schema import EventModel


class EventRetriever:
    def __init__(self, jsonl_path="data/events.jsonl", chroma_dir="data/chroma_db"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Инициализация RAG на устройстве: {self.device}")

        self.jsonl_path = jsonl_path
        self.docs = []

        # 1. Загружаем модели один раз
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": self.device, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = Chroma(
            collection_name="events",
            embedding_function=self.embeddings,
            persist_directory=chroma_dir,
        )

        # NOTE: device=cpu manually added
        self.reranker = CrossEncoder(
            "DiTy/cross-encoder-russian-msmarco", device="cpu"
        )

        logging.info(f"[DEBUG] CHROMADB ELEMs count: {self.vectorstore._collection.count()}")

        # 2. Подгружаем историю из JSONL и синхронизируем с ChromaDB
        self._load_and_sync()

    def _to_ts(self, dt):
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    def _event_to_doc(self, ev: EventModel) -> Document:
        text = f"{ev.title}\nКатегория: {ev.category}\n{ev.summary or ''}".strip()
        ts = self._to_ts(ev.date) or 0
        return Document(
            page_content=text,
            metadata={
                "event_id": ev.event_id,
                "event_ts": ts,
                "title": ev.title,
                "category": ev.category,
            },
        )

    def _load_and_sync(self):
        """Читает JSONL, обновляет BM25 и пушит в ChromaDB, если она пуста."""
        if not os.path.exists(self.jsonl_path):
            self.bm25 = None  # NOTE: self.bm25=[]
            return

        events = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(EventModel(**json.loads(line)))

        self.docs = [self._event_to_doc(e) for e in events]

        if self.vectorstore._collection.count() == 0 and self.docs:
            ids = [doc.metadata["event_id"] for doc in self.docs]
            self.vectorstore.add_documents(self.docs, ids=ids)

        if self.docs:
            self.bm25 = BM25Retriever.from_documents(self.docs)
            self.bm25.k = 20
        else:
            self.bm25 = None  # NOTE: here as well

    def add_event(self, ev: EventModel):
        """Добавляет свежий пост из Телеграма в базу на лету."""
        doc = self._event_to_doc(ev)
        self.vectorstore.add_documents([doc], ids=[ev.event_id])
        self.docs.append(doc)
        # Пересобираем BM25 для новых слов
        self.bm25 = BM25Retriever.from_documents(self.docs)
        self.bm25.k = 20
        print(f"[INFO] Добавлено в индекс поиска: {ev.title}")
        logging.info(f"[INFO] Добавлено в индекс поиска: {ev.title}")
        logging.info(f"[DEBUG] CHROMADB ELEMs count: {self.vectorstore._collection.count()}")

    def search(self, query: str, top_k=5):
        """Ищет релевантные события и возвращает список ID."""
        now_ts = int(time.time())

        # Поиск по смыслу (Dense)
        dense_docs = self.vectorstore.similarity_search(query, k=50)
        dense_docs = [d for d in dense_docs if d.metadata.get("event_ts", 0) >= now_ts][
            :20
        ]

        # Поиск по словам (BM25)
        bm25_docs = []
        if self.bm25:
            bm25_docs = self.bm25.invoke(query)
            bm25_docs = [
                d for d in bm25_docs if d.metadata.get("event_ts", 0) >= now_ts
            ]

        # Слияние
        seen, merged = set(), []
        for d in dense_docs + bm25_docs:
            eid = d.metadata.get("event_id")
            if not eid or eid in seen:
                continue
            merged.append(d)
            seen.add(eid)

        if not merged:
            return []

        # Реранкинг (самая умная часть)
        candidates = merged[:30]
        pairs = [(query, d.page_content) for d in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in ranked[:top_k]]

        return [d.metadata["event_id"] for d in top_docs]
