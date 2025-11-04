import streamlit as st
import requests

# ------------------ Config ------------------
TOP_N = 3
API_URL = "http://127.0.0.1:8000/api/venues/recommend"

# ------------------ Custom CSS for better styling ------------------
st.markdown("""
<style>
    .venue-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .venue-card-silver {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .venue-card-bronze {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .venue-name {
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .hybrid-score {
        font-size: 2rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    .metric-container {
        background: rgba(255,255,255,0.15);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .agent-name {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# ------------------ Streamlit Page Setup ------------------
st.set_page_config(page_title="Smart Venue Recommendations", layout="wide")
st.title("🏟️ Smart Venue Recommendations")

# Initialize session state
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'current_event_id' not in st.session_state:
    st.session_state.current_event_id = ""

# ------------------ Input Section ------------------
if not st.session_state.show_results:
    st.markdown("### Enter Event Details")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        event_id = st.text_input("Event ID", value="", placeholder="e.g., EVT-2026-001")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
            if event_id:
                with st.spinner("Fetching recommendations..."):
                    try:
                        response = requests.post(API_URL, json={"event_id": event_id, "top_n": TOP_N})
                        response.raise_for_status()
                        st.session_state.recommendations = response.json().get("recommendations", [])
                        st.session_state.current_event_id = event_id
                        st.session_state.show_results = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error fetching recommendations: {e}")
            else:
                st.warning("Please enter an Event ID")

# ------------------ Results Section ------------------
else:
    # Header with back button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"## Top {TOP_N} Venue Recommendations for **{st.session_state.current_event_id}**")
    with col2:
        if st.button("← New Search", use_container_width=True):
            st.session_state.show_results = False
            st.rerun()
    
    recommendations = st.session_state.recommendations
    
    if not recommendations:
        st.warning("No recommendations found.")
    else:
        # Create columns for side-by-side display
        cols = st.columns(TOP_N)
        
        card_classes = ["venue-card", "venue-card venue-card-silver", "venue-card venue-card-bronze"]
        medals = ["🥇", "🥈", "🥉"]
        
        for idx, (col, rec) in enumerate(zip(cols, recommendations)):
            with col:
                # Header card with gradient
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
                
                analysis = rec["analysis"]
                
                # ------------------ Agent Scores ------------------
                st.markdown("### 📊 Agent Scores")
                
                agent_config = [
                    ("capacity_agent", "Capacity Agent", "👥"),
                    ("amenity_agent", "Amenity Agent", "🎯"),
                    ("location_agent", "Location Agent", "📍"),
                    ("cost_agent", "Cost Agent", "💰"),
                    ("special_requirement_agent", "Special Requirements", "⭐")
                ]
                
                for agent_key, agent_display, icon in agent_config:
                    agent_info = analysis.get(agent_key, {})
                    score = agent_info.get("score", 0)
                    reason = agent_info.get("reason", "N/A")
                    
                    # Color code based on score
                    if score >= 80:
                        color = "🟢"
                    elif score >= 60:
                        color = "🟡"
                    else:
                        color = "🔴"
                    
                    st.markdown(f"{icon} **{agent_display}**: {color} {score}/100")
                    st.progress(score/100)
                    with st.expander("View reasoning"):
                        st.caption(reason)
                
                # ------------------ RAG & Feedback ------------------
                st.markdown("### 📈 Additional Metrics")
                
                rag_score = analysis.get("rag_similarity_score", 0)
                feedback_adj = analysis.get("feedback_adjustment", 0)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("RAG Score", f"{rag_score:.2f}")
                with col_b:
                    st.metric("Feedback", f"{feedback_adj:+.1f}", 
                             delta=f"{feedback_adj:+.1f}" if feedback_adj != 0 else None)
                
                # ------------------ Historical Event Summary ------------------
                hist_event = analysis.get("historical_event", {})
                with st.expander("📜 Historical Event Details"):
                    st.markdown(f"**Event ID:** {hist_event.get('event_id', 'N/A')}")
                    st.markdown(f"**Overall Satisfaction:** {hist_event.get('overall_satisfaction', 'N/A')}")
                    st.markdown(f"**Would Rebook:** {hist_event.get('would_rebook', 'N/A')}")
                    
                    pos = hist_event.get("positive_feedback", [])
                    neg = hist_event.get("negative_feedback", [])
                    
                    if pos or neg:
                        st.markdown("---")
                        st.markdown(f"✅ **Positive Feedback:** {len(pos)} items")
                        if pos:
                            for p in pos[:3]:
                                st.markdown(f"- {p}")
                        
                        st.markdown(f"❌ **Negative Feedback:** {len(neg)} items")
                        if neg:
                            for n in neg[:3]:
                                st.markdown(f"- {n}")