# agents.py
"""
Agents: Rule-based scoring + local Ollama (phi3) for natural language reasoning.
Fully offline – no OpenAI.
"""

import ollama
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def _llm_explain(prompt: str) -> str:
    """
    Central call to local Ollama for agent explanations.
    Temperature=0.2 for consistent, professional output.
    """
    try:
        resp = ollama.chat(
            model="phi3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        return f"[Local LLM error: {e}]"


class CapacityAgent:
    def analyze(self, event, venue):
        max_cap = venue.get("max_capacity", 0)
        attendees = event.get("attendee_count", 0)
        if max_cap == 0:
            score = 0
            reason = "No capacity data available"
        else:
            utilization = attendees / max_cap
            score = max(0, int(100 - utilization * 100))
            reason = f"{attendees} attendees vs {max_cap} capacity ({utilization:.1%} utilization)"

        prompt = f"Explain capacity score {score}/100 for {attendees} attendees in a {max_cap}-person venue. Keep it 1 sentence."
        llm = _llm_explain(prompt)
        return {"score": score, "reason": reason, "llm_explanation": llm}

    # ------------------------------------------------------------------
    # BONUS – EASY: Confidence scoring for capacity
    # ------------------------------------------------------------------
    # return {"confidence": 100 if max_cap > 0 else 50, ...}


class AmenityAgent:
    def analyze(self, event, venue):
        req = set(event.get("required_amenities", []))
        pref = set(event.get("preferred_amenities", []))
        avail = set(venue.get("amenities", []))

        matched_req = req & avail
        matched_pref = pref & avail
        score = int((
            len(matched_req) / max(len(req), 1) * 70 +
            len(matched_pref) / max(len(pref), 1) * 30
        ))
        reason = f"Required: {len(matched_req)}/{len(req)} matched, Preferred: {len(matched_pref)}/{len(pref)}"

        prompt = f"Explain amenity score {score}/100: required {list(req)}, available {list(avail)}. 1 sentence."
        llm = _llm_explain(prompt)
        return {"score": score, "reason": reason, "llm_explanation": llm}


class LocationAgent:
    def analyze(self, event, venue):
        pref = event.get("location_preference")
        actual = venue.get("region")
        score = 100 if actual == pref else 50  # Partial match
        reason = f"Preferred: {pref}, Venue: {actual}"

        prompt = f"Explain location score {score}/100: preferred {pref}, venue in {actual}. 1 sentence."
        llm = _llm_explain(prompt)
        return {"score": score, "reason": reason, "llm_explanation": llm}


class CostAgent:
    def analyze(self, event, venue):
        budget = event.get("budget", 0)
        daily = venue.get("daily_rate", 0)
        days = event.get("duration_days", 1)
        est = daily * days
        if budget == 0:
            score = 0
            reason = "No budget specified"
        else:
            score = min(100, int(budget / max(est, 1) * 100))
            reason = f"Estimated ${est} vs budget ${budget} ({score}% fit)"

        prompt = f"Explain cost score {score}/100: estimated ${est}, budget ${budget}. 1 sentence."
        llm = _llm_explain(prompt)
        return {"score": score, "reason": reason, "llm_explanation": llm}

    # ------------------------------------------------------------------
    # BONUS – MEDIUM: Budget Optimization Agent
    # ------------------------------------------------------------------
    # Add: suggestions = ["Upgrade to half-day for 20% savings"]


class SpecialRequirementAgent:
    def analyze(self, event, venue):
        reqs = set(event.get("special_requirements", []))
        avail = set(venue.get("amenities", [])) | set(venue.get("features", []))
        matched = reqs & avail
        score = int(len(matched) / max(len(reqs), 1) * 100)
        reason = f"Matched {len(matched)}/{len(reqs)} special requirements"

        prompt = f"Explain special req score {score}/100: needs {list(reqs)}, available {list(avail)}. 1 sentence."
        llm = _llm_explain(prompt)
        return {"score": score, "reason": reason, "llm_explanation": llm}


class FeedbackAgent:
    def analyze(self, hist):
        """Sentiment analysis on historical feedback (local, no LLM)."""
        pos = hist.get("positive_feedback", [])
        neg = hist.get("negative_feedback", [])
        score = sum(analyzer.polarity_scores(c)["compound"] * 5 for c in pos)
        score += sum(analyzer.polarity_scores(c)["compound"] * 5 for c in neg)
        adjustment = max(-20, min(20, score))
        
        # ------------------------------------------------------------------
        # BONUS – EASY: LLM summary of feedback
        # ------------------------------------------------------------------
        # prompt = f"Summarize feedback: pos {pos}, neg {neg}"
        # llm_summary = _llm_explain(prompt)
        
        return adjustment  # Numeric adjustment for hybrid score