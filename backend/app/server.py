import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Import the compiled graph 
from app.agents.agent import app, AgentState

# Initialize FastAPI app
api = FastAPI(
    title="Math Professor Agent API",
    description="An API for interacting with a multi-agent system for math questions.",
)

# --- Pydantic Models for API Requests ---
class GenerateRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    thread_id: str
    feedback: str

# --- API Endpoints ---

@api.post("/generate", summary="Start a new math query and get an answer")
async def generate_answer(request: GenerateRequest):
    """
    Takes a user's math question, runs it through the agent graph,
    and returns the final answer along with a thread_id for providing feedback.
    """
    # Each new request gets a unique ID
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # The input for the graph
    input_data = {"messages": [("user", request.query)]}

    # Asynchronously invoke the graph
    
    final_state = await app.ainvoke(input_data, config)

    return {
        "answer": final_state["messages"][-1].content,
        "thread_id": thread_id,
        "comment": "Answer generated. Provide feedback using the /feedback endpoint."
    }

@api.post("/feedback", summary="Submit feedback for a generated answer")
async def submit_feedback(request: FeedbackRequest):
    """
    Takes feedback for a specific run (identified by thread_id)
    and continues the graph to log the feedback.
    """
    config = {"configurable": {"thread_id": request.thread_id}}

    
    app.update_state(config, {"feedback": request.feedback})

    
    await app.ainvoke(None, config)

    return {
        "message": "Feedback successfully recorded.",
        "thread_id": request.thread_id
    }


@api.get("/")
def read_root():
    return {"status": "Math Professor Agent API is running."}