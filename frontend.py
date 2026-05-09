import streamlit as st
import requests
import streamlit_analytics2 as st_analytics
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Sarthak's AI Portfolio", page_icon="👨‍💻")

# URL for your FastAPI Backend
# Chat endpoint
BACKEND_URL = "https://interactive-ai-career-agent.onrender.com/chat"

# Health check endpoint
HEALTH_URL = "https://interactive-ai-career-agent.onrender.com"

# --- 2. ANALYTICS & TRACKING ---
# Wrap EVERYTHING in the track block to capture all interactions
with st_analytics.track(unsafe_password="your_secret_password"):

    # --- 3. SESSION STATE (Memory) ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- 4. SIDEBAR & STATUS ---
    with st.sidebar:
        st.title("🤖 Agent Status")
        try:
            # Health check to FastAPI
            response = requests.get(HEALTH_URL, timeout=2)
            if response.status_code == 200:
                st.success("System: Online")
            else:
                st.warning("System: Issues detected")
        except:
            st.error("System: Offline (Run main.py)")
        
        st.info("""
        **How it works:**
        This is an Agentic RAG system. It uses a self-correcting loop via LangGraph to verify answers against my portfolio before responding.
        """)
        
        if st.sidebar.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # --- 5. MAIN UI HEADER ---
    st.title("👨‍💻 Agentic Portfolio AI")
    st.caption("Architecture: Streamlit ➔ FastAPI ➔ LangGraph ➔ Groq")
    st.divider()

    # --- 6. DISPLAY CHAT HISTORY ---
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # --- 7. USER INPUT & API CALL ---
    if prompt := st.chat_input("Ask about my projects, skills, or experience..."):
        # Display user message
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call the FastAPI Backend
        with st.chat_message("assistant"):
            with st.spinner("Agent is searching and thinking..."):
                try:
                    payload = {"question": prompt}
                    response = requests.post(BACKEND_URL, json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "⚠️ No response from agent.")
                        
                        # Display AI Answer
                        st.markdown(answer)
                        st.session_state.messages.append(AIMessage(content=answer))
                        
                        # Display metadata (Reasoning steps/Attempt count)
                        if "count" in data:
                            st.caption(f"Retrieved after {data['count']} internal search attempts.")
                    else:
                        st.error(f"Backend Error: {response.status_code}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Connection Refused: Ensure your FastAPI server (main.py) is running on port 8000.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    # --- 8. CUSTOM STYLING ---
    st.markdown("""
        <style>
            .stChatMessage { border-radius: 12px; }
            .stSidebar { background-color: #f0f2f6; }
            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
