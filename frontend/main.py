import streamlit as st
import requests
import json

# --- Configuration ---
FASTAPI_BASE_URL = "http://127.0.0.1:8000"
GENERATE_URL = f"{FASTAPI_BASE_URL}/generate"
FEEDBACK_URL = f"{FASTAPI_BASE_URL}/feedback"

# --- Page Setup ---
st.set_page_config(
    page_title="Math Professor Agent",
    page_icon="🧠",
    layout="centered",
)

st.title("Math Professor Agent 🧠")

# --- Session State Initialization ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "answer" not in st.session_state:
    st.session_state.answer = None
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# --- Main App Logic ---

with st.form("question_form"):
    user_question = st.text_input(
        "Ask a math question:",
        placeholder="e.g., What is the difference between integration and differentiation?"
    )
    submit_button = st.form_submit_button("Ask the Professor")

if submit_button and user_question:
    st.session_state.answer = None
    st.session_state.thread_id = None
    st.session_state.feedback_submitted = False

    with st.spinner("The agent is thinking..."):
        try:
            response = requests.post(GENERATE_URL, json={"query": user_question})
            response.raise_for_status()
            data = response.json()
            st.session_state.answer = data.get("answer")
            st.session_state.thread_id = data.get("thread_id")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the agent's backend. Ensure the server is running. Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if st.session_state.answer:
    st.markdown("---")
    st.subheader("🤖 Here's the explanation:")
    st.markdown(st.session_state.answer)

    if not st.session_state.feedback_submitted:
        st.markdown("---")
        st.write("Was this explanation helpful? (Click an option below)")

        def submit_feedback(feedback_text: str):
            """Generic function to submit any feedback string."""
            if not feedback_text:
                st.warning("Please provide feedback before submitting.")
                return
            try:
                feedback_data = {"thread_id": st.session_state.thread_id, "feedback": feedback_text}
                response = requests.post(FEEDBACK_URL, json=feedback_data)
                response.raise_for_status()
                st.session_state.feedback_submitted = True
                st.toast("Thank you for your feedback! ✨")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to submit feedback. Error: {e}")

        

        # 1. Quick Feedback Buttons
        col1, col2 = st.columns(2)
        with col1:
            st.button("👍 Helpful", on_click=submit_feedback, args=("helpful",), use_container_width=True)
        with col2:
            st.button("👎 Unhelpful", on_click=submit_feedback, args=("unhelpful",), use_container_width=True)

        # 2. Detailed Feedback Form
        with st.form("detailed_feedback_form"):
            detailed_feedback = st.text_area("Or, provide more detailed feedback:", height=100)
            submit_detailed_feedback = st.form_submit_button("Submit Detailed Feedback")

            if submit_detailed_feedback:
                submit_feedback(detailed_feedback)
        
        

if st.session_state.feedback_submitted:
    st.success("Feedback submitted! You can ask another question above.")