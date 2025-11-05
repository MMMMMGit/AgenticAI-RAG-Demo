# rag.py
"""
RAG Module: Fully offline with Sentence Transformers + Ollama (phi3)
- No OpenAI
- No API keys
- No internet after setup
- Uses local embeddings + local LLM
"""

import json
import os
from typing import List, Dict, Any

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------
# Local Embedding Model (no internet, no key)
# ----------------------------------------------------------------------
print("Loading local embedding model... (first time may take 10 sec)")
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # ~80MB, CPU-friendly
print("Embedding model loaded.")

# ----------------------------------------------------------------------
# Chroma Setup – NEW API (Chroma 0.5+)
# ----------------------------------------------------------------------
CHROMA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_PATH)  # Fixed deprecation
collection = client.get_or_create_collection(name="event_history")


def build_index(history_file: str) -> None:
    """
    Build Chroma index from event_history.json using local embeddings.
    Runs once at startup.
    """
    if not os.path.exists(history_file):
        print(f"[ERROR] History file not found: {history_file}")
        return

    with open(history_file, "r", encoding="utf-8") as f:
        events = json.load(f)

    print(f"Indexing {len(events)} historical events...")

    ids = []
    docs = []
    embeddings = []

    for ev in events:
        doc = json.dumps(ev, ensure_ascii=False)
        ids.append(ev["event_id"])
        docs.append(doc)
        embeddings.append(EMBEDDING_MODEL.encode(doc).tolist())

    # Add all at once
    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings
    )
    print(f"Indexed {len(events)} events into offline Chroma.")


class RAG:
    def __init__(self, top_n: int = 6):
        self.top_n = top_n

    def retrieve(self, request: Dict) -> List[Dict]:
        """
        Retrieve top-N similar past events using local embeddings.
        Returns: list of {historical_event, similarity_score}
        """
        # Build query document
        query_doc = json.dumps({
            "event_type": request.get("event_type"),
            "attendee_count": request.get("attendee_count"),
            "duration_days": request.get("duration_days"),
            "budget": request.get("budget"),
            "required_amenities": request.get("required_amenities", []),
            "special_requirements": request.get("special_requirements", []),
            "event_style": request.get("event_style")
        }, ensure_ascii=False)

        query_emb = EMBEDDING_MODEL.encode(query_doc).tolist()

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=self.top_n * 2,
            include=["documents", "distances"]
        )

        similar = []
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            event = json.loads(doc)
            # Convert cosine distance → similarity (0–100)
            similarity = max(0, round(100 * (1 - dist), 1))
            similar.append({
                "historical_event": event,
                "similarity_score": similarity
            })
        return similar

    # ------------------------------------------------------------------
    # BONUS – EASY: LLM summary of retrieved events
    # ------------------------------------------------------------------
    def summarize_retrieval(self, similar_events: List[Dict]) -> str:
        """Use local Ollama (phi3) to explain relevance."""
        if not similar_events:
            return "No similar past events found."

        # Top 3 for summary
        top3 = similar_events[:3]
        events_summary = "\n".join([
            f"- {e['historical_event']['event_name']} "
            f"({e['historical_event']['event_type']}, "
            f"{e['historical_event']['attendee_count']} attendees, "
            f"satisfaction: {e['historical_event']['overall_satisfaction']}/5)"
            for e in top3
        ])

        prompt = f"""
You are an expert event planner. In 2-3 concise sentences, explain why these past events are relevant to a new event with:
- Type: {top3[0]['historical_event'].get('event_type', 'unknown')}
- Size: {top3[0]['historical_event'].get('attendee_count', 0)} attendees
- Style: {top3[0]['historical_event'].get('event_style', 'unknown')}

Past events:
{events_summary}

Keep it professional and under 80 words.
"""

        try:
            resp = ollama.chat(
                model="phi3",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            return f"[LLM Error: {str(e)}]"

    # ------------------------------------------------------------------
    # BONUS – MEDIUM: Filter low-satisfaction events
    # ------------------------------------------------------------------
    # In retrieve(): if event['overall_satisfaction'] < 3.0: similarity *= 0.7

    # ------------------------------------------------------------------
    # BONUS – HARD: Hybrid query (vector + keyword filter)
    # ------------------------------------------------------------------
    # Add: where={"event_type": request["event_type"]} in collection.query()