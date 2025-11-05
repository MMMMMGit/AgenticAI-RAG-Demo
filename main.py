# main.py
"""
FastAPI orchestrator: Offline RAG + local Ollama LLM.
Loads data, retrieves similar events, scores with agents, hybrid ranking.
No API keys, no internet.
"""

import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from rag import RAG, build_index
from agents import (
    CapacityAgent, AmenityAgent, LocationAgent,
    CostAgent, SpecialRequirementAgent, FeedbackAgent
)

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
app = FastAPI(title="Offline Smart Venue Recommender – RAG + Local LLM")

DATA_DIR = "./data"
VENUES_FILE = os.path.join(DATA_DIR, "venues.json")
REQUESTS_FILE = os.path.join(DATA_DIR, "current_requests.json")
HISTORY_FILE = os.path.join(DATA_DIR, "event_history.json")

with open(VENUES_FILE, "r", encoding="utf-8") as f:
    venues = json.load(f)
with open(REQUESTS_FILE, "r", encoding="utf-8") as f:
    current_requests = json.load(f)

# Build index on startup
rag = RAG(top_n=6)
build_index(HISTORY_FILE)

# Agents
capacity_agent = CapacityAgent()
amenity_agent = AmenityAgent()
location_agent = LocationAgent()
cost_agent = CostAgent()
special_agent = SpecialRequirementAgent()
feedback_agent = FeedbackAgent()

# Weights (unchanged)
AGENT_WEIGHTS = {
    "capacity": 0.25,
    "amenity": 0.20,
    "location": 0.15,
    "cost": 0.25,
    "special": 0.15
}
HYBRID_WEIGHTS = {
    "agent": 0.45,
    "rag": 0.45,
    "feedback": 0.10
}

class EventRequest(BaseModel):
    event_id: str
    top_n: int = 3


@app.post("/api/venues/recommend")
def recommend(payload: EventRequest):
    request = next((r for r in current_requests if r["event_id"] == payload.event_id), None)
    if not request:
        raise HTTPException(status_code=404, detail="Event request not found")

    # RAG retrieval (offline)
    similar_events = rag.retrieve(request)
    if not similar_events:
        raise HTTPException(status_code=404, detail="No similar past events found")

    # Score candidates
    recommendations = []
    for item in similar_events:
        hist_event = item["historical_event"]
        venue = next((v for v in venues if v["venue_id"] == hist_event.get("venue_id")), None)
        if not venue:
            continue

        # Agent analyses
        cap = capacity_agent.analyze(request, venue)
        ame = amenity_agent.analyze(request, venue)
        loc = location_agent.analyze(request, venue)
        cst = cost_agent.analyze(request, venue)
        spc = special_agent.analyze(request, venue)
        fb_adj = feedback_agent.analyze(hist_event)

        # Agent composite score
        agent_score = (
            cap["score"] * AGENT_WEIGHTS["capacity"] +
            ame["score"] * AGENT_WEIGHTS["amenity"] +
            loc["score"] * AGENT_WEIGHTS["location"] +
            cst["score"] * AGENT_WEIGHTS["cost"] +
            spc["score"] * AGENT_WEIGHTS["special"]
        )

        # Hybrid score
        rag_score = item["similarity_score"]
        hybrid_score = (
            HYBRID_WEIGHTS["agent"] * agent_score +
            HYBRID_WEIGHTS["rag"] * rag_score +
            HYBRID_WEIGHTS["feedback"] * fb_adj
        )

        recommendations.append({
            "venue_id": venue["venue_id"],
            "venue_name": venue["name"],
            "ranking_score": round(hybrid_score, 1),
            "analysis": {
                "capacity_agent": cap,
                "amenity_agent": ame,
                "location_agent": loc,
                "cost_agent": cst,
                "special_requirement_agent": spc,
                "feedback_adjustment": fb_adj,
                "rag_similarity_score": rag_score,
                "rag_summary": rag.summarize_retrieval([item]),  # Local LLM summary
                "historical_event": hist_event
            }
        })

    if not recommendations:
        raise HTTPException(status_code=404, detail="No suitable venues found")

    # Sort and return top N
    recommendations.sort(key=lambda x: x["ranking_score"], reverse=True)
    return {"recommendations": recommendations[:payload.top_n]}

# ----------------------------------------------------------------------
# BONUS – EASY: Explanation quality – already in agents (LLM reasons)
# BONUS – MEDIUM: Date Availability Agent – check request['preferred_dates'] vs hist
# BONUS – HARD: Client History Agent – load from client_profiles.json by client_id