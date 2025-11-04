# agents.py
"""
Agents for the Venue Recommendation System.
Each agent evaluates a venue against a specific criterion.
FeedbackAgent now uses sentiment analysis to interpret historical feedback.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------- Capacity Agent --------------------
class CapacityAgent:
    def analyze(self, event, venue):
        """
        Compare venue max capacity vs requested attendees.
        Returns a dict with score (0-100) and reasoning.
        """
        max_capacity = venue.get("max_capacity", 0)
        attendee_count = event.get("attendee_count", 0)

        if max_capacity == 0:
            score = 0
            reason = "Venue capacity not specified"
        else:
            utilization = attendee_count / max_capacity
            score = max(0, int(100 - utilization * 100))
            reason = f"Venue capacity {max_capacity} vs requested {attendee_count}"

        # EASY: Add confidence scoring in future (e.g., reliability of capacity info)
        return {"score": score, "reason": reason}

# -------------------- Amenity Agent --------------------
class AmenityAgent:
    def analyze(self, event, venue):
        """
        Compare required and preferred amenities.
        Returns dict with score (0-100) and reasoning.
        """
        required = set(event.get("required_amenities", []))
        preferred = set(event.get("preferred_amenities", []))
        available = set(venue.get("amenities", []))

        matched_required = required & available
        matched_preferred = preferred & available

        # Weighted: required = 70%, preferred = 30%
        score = int((len(matched_required)/max(len(required),1)*0.7 + 
                     len(matched_preferred)/max(len(preferred),1)*0.3)*100)
        reason = f"Matched required: {list(matched_required)}, matched preferred: {list(matched_preferred)}"

        # MEDIUM: Can add semantic matching for similar amenities
        return {"score": score, "reason": reason}

# -------------------- Location Agent --------------------
class LocationAgent:
    def analyze(self, event, venue):
        """
        Evaluate if venue is in preferred location.
        Returns dict with score (0-100) and reasoning.
        """
        preferred_region = event.get("location_preference")
        venue_region = venue.get("region")
        score = 100 if venue_region == preferred_region else 50
        reason = f"Venue in {venue_region}, preferred {preferred_region}"

        # EASY: Consider nearby alternatives if exact match unavailable
        return {"score": score, "reason": reason}

# -------------------- Cost Agent --------------------
class CostAgent:
    def analyze(self, event, venue):
        """
        Compare estimated cost vs event budget.
        Returns dict with score (0-100) and reasoning.
        """
        budget = event.get("budget", 0)
        daily_rate = venue.get("daily_rate", 0)
        duration = event.get("duration_days", 1)
        estimated_cost = daily_rate * duration

        if budget == 0:
            score = 0
            reason = "Event budget not specified"
        else:
            score = min(100, int(budget / max(estimated_cost, 1) * 100))
            reason = f"Estimated cost ${estimated_cost} vs budget ${budget}"

        # HARD: Budget optimization suggestions can be added later
        return {"score": score, "reason": reason}

# -------------------- Special Requirement Agent (HARD) --------------------
class SpecialRequirementAgent:
    def analyze(self, event, venue):
        """
        Match special requirements against venue features/amenities.
        """
        special_reqs = set(event.get("special_requirements", []))
        available = set(venue.get("amenities", [])) | set(venue.get("features", []))
        matched = special_reqs & available
        score = int(len(matched)/max(len(special_reqs),1)*100)
        reason = f"Matched special requirements: {list(matched)}"
        return {"score": score, "reason": reason}

# -------------------- Feedback Agent --------------------
class FeedbackAgent:
    """
    Uses sentiment analysis to interpret historical feedback.
    Returns a numeric adjustment for hybrid scoring.
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, historical_event):
        positive_feedback = historical_event.get("positive_feedback", [])
        negative_feedback = historical_event.get("negative_feedback", [])

        total_score = 0

        # Positive comments
        for comment in positive_feedback:
            sentiment = self.analyzer.polarity_scores(comment)["compound"]
            total_score += sentiment * 5

        # Negative comments
        for comment in negative_feedback:
            sentiment = self.analyzer.polarity_scores(comment)["compound"]
            total_score += sentiment * 5

        # Cap adjustment to avoid extreme swings
        total_score = max(-20, min(20, total_score))
        return total_score
