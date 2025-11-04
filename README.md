# Smart Venue Recommender – Multi-Agent RAG Engine

**TravelEdge Assignment** – A **hybrid scoring system** combining **5 AI agents**, **TF-IDF RAG**, and **historical feedback** to recommend the **top 3 venues** for any event.

---

## Features
- **5 Rule-Based Agents**:
  1. **Capacity Agent** – Checks attendee count vs max capacity
  2. **Amenity Agent** – Exact match of required amenities
  3. **Location Agent** – Matches preferred region
  4. **Cost Agent** – Budget vs estimated cost
  5. **Special Requirements Agent** – Custom feature matching
- **Feedback Agent** – Boosts venues with positive past feedback
- **RAG (TF-IDF)** – Semantic retrieval of similar past events
- **Hybrid Score** = `0.45×Agent + 0.45×RAG + 0.1×Feedback`
- **Streamlit UI** – Gold/Silver/Bronze cards, expandable reasoning
- **FastAPI Backend** – `/api/venues/recommend`
---

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/YOURUSERNAME/venue_recommender.git
cd venue_recommender

# 2. Create environment
conda env create -f environment.yml
conda activate venue-rec

# 3. Start FastAPI backend
uvicorn main:app --reload

# 4. Launch Streamlit UI
streamlit run app.py