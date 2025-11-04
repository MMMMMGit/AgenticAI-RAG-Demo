# rag.py
"""
RAG module: Retrieves similar historical events for a given request.
Uses TF-IDF on key_requirements and special requirements, normalized similarity,
and penalizes unsuccessful events.
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RAG:
    def __init__(self, event_history_file):
        with open(event_history_file, "r") as f:
            self.events = json.load(f)

        self.vectorizer = TfidfVectorizer()
        corpus = [" ".join(e.get("key_requirements", [])) for e in self.events]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve_similar(self, request_event, top_n=3):
        """
        Returns top-N historical events with normalized similarity scores.
        Penalizes events that were not successful or not rebooked.
        """
        request_text = " ".join(request_event.get("required_amenities", []) +
                                request_event.get("special_requirements", []))
        request_vec = self.vectorizer.transform([request_text])
        similarities = cosine_similarity(request_vec, self.tfidf_matrix)[0]

        # Normalize to 0-100
        if len(similarities) > 0:
            sim_min = np.min(similarities)
            sim_max = np.max(similarities)
            norm_sims = 100 * (similarities - sim_min) / max((sim_max - sim_min), 1e-6)
        else:
            norm_sims = similarities

        adjusted_scores = []
        for idx, event in enumerate(self.events):
            score = norm_sims[idx]
            outcome_multiplier = 1.0 if event.get("would_rebook", True) else 0.5
            satisfaction = event.get("overall_satisfaction", 3.0) / 5.0
            adjusted_score = score * outcome_multiplier * satisfaction
            adjusted_scores.append((adjusted_score, score, event))

        adjusted_scores.sort(key=lambda x: x[0], reverse=True)

        top_events = []
        for _, raw_sim, e in adjusted_scores[:top_n]:
            top_events.append({
                "historical_event": e,
                "similarity_score": raw_sim
            })

        return top_events
