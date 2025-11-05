# streamlit_app.py
import streamlit as st
import requests

# ------------------ Config ------------------
TOP_N = 3
API_URL = "http://127.0.0.1:8000/api/venues/recommend"

# ------------------ CSS (unchanged) ------------------
st.markdown("""
<style>
    .venue-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .venue-card-silver {background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);}
    .venue-card-bronze {background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);}
    .venue-name {font-size: 1.4rem; font-weight: bold; margin-bottom: 0.5rem;}
    .hybrid-score {font-size: 2rem; font-weight: bold; margin-top: 0.5rem;}
    .llm-explain {background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Smart Venue Recommendations", layout="wide")
st.title("Smart Venue Recommendations")

if 'show_results' not in st.session_state:
    st.session_state.show_results = False
    st.session_state.recommendations = []
    st.session_state.current_event_id = ""

# ------------------ Input ------------------
if not st.session_state.show_results:
    st.markdown("### Enter Event ID")
    col1, col2 = st.columns([3, 1])
    with col1:
        event_id = st.text_input("Event ID", placeholder="e.g., EVT-2026-001")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Get Recommendations", type="primary", use_container_width=True):
            if event_id:
                with st.spinner("Thinking..."):
                    try:
                        resp = requests.post(API_URL, json={"event_id": event_id, "top_n": TOP_N})
                        resp.raise_for_status()
                        data = resp.json()
                        st.session_state.recommendations = data["recommendations"]
                        st.session_state.current_event_id = event_id
                        st.session_state.show_results = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Enter an Event ID")

# ------------------ Results ------------------
else:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"## Top {TOP_N} for **{st.session_state.current_event_id}**")
    with col2:
        if st.button("New Search", use_container_width=True):
            st.session_state.show_results = False
            st.rerun()

    recs = st.session_state.recommendations
    cols = st.columns(TOP_N)
    medals = ["1st", "2nd", "3rd"]
    card_classes = ["venue-card", "venue-card venue-card-silver", "venue-card venue-card-bronze"]

    for idx, (col, rec) in enumerate(zip(cols, recs)):
        with col:
            st.markdown(f"""
            <div class="{card_classes[idx]}">
                <div style="text-align: center;">
                    <div style="font-size: 3rem;">{medals[idx]}</div>
                    <div class="venue-name">{rec['venue_name']}</div>
                    <div class="hybrid-score">{rec['ranking_score']}</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">Hybrid Score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            a = rec["analysis"]

            # RAG Summary (LLM)
            st.markdown("### Why This Venue?")
            st.markdown(f"<div class='llm-explain'>{a['rag_summary']}</div>", unsafe_allow_html=True)

            # Agent Scores
            st.markdown("### Agent Breakdown")
            for key, label, icon in [
                ("capacity_agent", "Capacity", "People"),
                ("amenity_agent", "Amenities", "Target"),
                ("location_agent", "Location", "Map Pin"),
                ("cost_agent", "Budget", "Money"),
                ("special_requirement_agent", "Special Reqs", "Star")
            ]:
                info = a.get(key, {})
                score = info.get("score", 0)
                llm = info.get("llm_explanation", "No explanation")
                color = "Green" if score >= 80 else "Yellow" if score >= 60 else "Red"
                st.markdown(f"{icon} **{label}**: {color} {score}/100")
                with st.expander("LLM Explanation"):
                    st.caption(llm)

            # Feedback
            with st.expander("Past Event Feedback"):
                pos = a["historical_event"].get("positive_feedback", [])[:2]
                neg = a["historical_event"].get("negative_feedback", [])[:2]
                for p in pos: st.markdown(f"Positive: {p}")
                for n in neg: st.markdown(f"Negative: {n}")