import streamlit as st
import time
from rag_tools import run_tool

st.set_page_config(
    page_title="Medical QA Chatbot",
    layout="wide",
    page_icon="\U0001F3E5",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI layout
with st.container():
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.markdown("## ⚙️ Settings")
        selected_model = st.radio(
            "Select Model",
            ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"],
            key="model_selector",
        )
        rag_toggle = st.toggle("Enable RAG", value=True, key="rag_toggle")
        st.markdown("---")
        st.markdown("## 💡 Key Features")
        st.markdown(
            """
        - **BioBERT for Embeddings**
        - **Prebuilt ChromaDB Vector Store**
        - **PySpur Tools Available:**
            - Treatment Recommender
            - Symptom Cause Analyzer
            - Clinical Trial Matcher
            - Chat Memory Symptom Reasoner
        """
        )

    with right_col:
        st.markdown(
            """
        <div style='text-align: center; padding-top: 10px;'>
            <h2>❤️ Medical QA Chatbot</h2>
        </div>
        <br><br>
        """,
            unsafe_allow_html=True,
        )

        chat_container = st.container()
        user_query = None

        # Show chat history
        for entry in st.session_state.chat_history:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(entry["user"])
            if "response" in entry:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(entry["response"])

        user_query = st.chat_input("Ask a question...")

        if user_query:
            st.session_state.chat_history.append({"user": user_query})
            try:
                response = run_tool(user_query, st.session_state.chat_history, selected_model, rag_toggle)
            except Exception as e:
                response = f"Error handling query: {str(e)}"
            st.session_state.chat_history[-1]["response"] = response
            st.rerun()

        # Clear button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Sidebar styling
st.markdown("""
<style>
    section[data-testid="stSidebar"] > div:first-child {
        border-right: 2px solid #ddd;
        padding-right: 1rem;
    }
</style>
""", unsafe_allow_html=True)
