# main.py
"""
Main orchestration: Computes hybrid recommendations using weighted agents, RAG, and feedback.
Returns top-N venues with detailed analysis.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import RAG
from agents import CapacityAgent, AmenityAgent, LocationAgent, CostAgent, SpecialRequirementAgent, FeedbackAgent
import json
import os

app = FastAPI(title="Smart Venue Recommendation Engine")

class EventRequest(BaseModel):
    event_id: str
    top_n: int = 3

DATA_DIR = "./data"
with open(os.path.join(DATA_DIR, "venues.json"), "r") as f:
    venues = json.load(f)
with open(os.path.join(DATA_DIR, "current_requests.json"), "r") as f:
    current_requests = json.load(f)

rag = RAG(os.path.join(DATA_DIR, "event_history.json"))

# Instantiate agents
capacity_agent = CapacityAgent()
amenity_agent = AmenityAgent()
location_agent = LocationAgent()
cost_agent = CostAgent()
special_agent = SpecialRequirementAgent()
feedback_agent = FeedbackAgent()

# -------------------- Configurable Weights --------------------
AGENT_WEIGHTS = {
    "capacity": 0.25,
    "amenity": 0.2,
    "location": 0.15,
    "cost": 0.25,
    "special": 0.15
}
HYBRID_WEIGHTS = {
    "agent": 0.45,
    "rag": 0.45,
    "feedback": 0.1
}

@app.post("/api/venues/recommend")
def recommend(event_request: EventRequest):
    request = next((r for r in current_requests if r["event_id"] == event_request.event_id), None)
    if not request:
        raise HTTPException(status_code=404, detail="Event request not found")

    similar_events = rag.retrieve_similar(request, top_n=event_request.top_n*2)

    recommendations = []
    for candidate in similar_events:
        hist_event = candidate["historical_event"]
        venue = next((v for v in venues if v["venue_id"] == hist_event.get("venue_id")), None)
        if not venue:
            continue

        # -------------------- Agent Scoring --------------------
        capacity = capacity_agent.analyze(request, venue)["score"]
        amenity = amenity_agent.analyze(request, venue)["score"]
        location = location_agent.analyze(request, venue)["score"]
        cost = cost_agent.analyze(request, venue)["score"]
        special = special_agent.analyze(request, venue)["score"]
        feedback_adjust = feedback_agent.analyze(hist_event)

        agent_score = (
            capacity * AGENT_WEIGHTS["capacity"] +
            amenity * AGENT_WEIGHTS["amenity"] +
            location * AGENT_WEIGHTS["location"] +
            cost * AGENT_WEIGHTS["cost"] +
            special * AGENT_WEIGHTS["special"]
        )

        rag_score = candidate["similarity_score"]

        hybrid_score = (
            HYBRID_WEIGHTS["agent"] * agent_score +
            HYBRID_WEIGHTS["rag"] * rag_score +
            HYBRID_WEIGHTS["feedback"] * feedback_adjust
        )

        recommendations.append({
            "venue_id": venue.get("venue_id"),
            "venue_name": venue.get("name"),
            "ranking_score": round(hybrid_score,1),
            "analysis": {
                "capacity_agent": capacity_agent.analyze(request, venue),
                "amenity_agent": amenity_agent.analyze(request, venue),
                "location_agent": location_agent.analyze(request, venue),
                "cost_agent": cost_agent.analyze(request, venue),
                "special_requirement_agent": special_agent.analyze(request, venue),
                "feedback_adjustment": feedback_adjust,
                "rag_similarity_score": round(rag_score,1),
                "historical_event": hist_event
            }
        })

    if not recommendations:
        raise HTTPException(status_code=404, detail="No suitable venue recommendations found")

    recommendations.sort(key=lambda x: x["ranking_score"], reverse=True)
    return {"recommendations": recommendations[:event_request.top_n]}
