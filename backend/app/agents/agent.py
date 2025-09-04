import os
import json
from typing import TypedDict, Annotated, List, Dict, Any,Optional
from pprint import pprint
import re
import csv
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END,add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage
from langchain.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

import warnings
warnings.filterwarnings('ignore')

from app.knowledge_base.query_kb import query_kbf

load_dotenv()

# Check for API keys
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    raise ValueError("Please set GOOGLE_API_KEY and TAVILY_API_KEY environment variables in .env")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_question: str
    feedback: Optional[str] 

raw_tavily = TavilySearchResults(max_results=3)

def tavily_search(query: str) -> str:
    """Wrapper around Tavily that formats output nicely."""
    results = raw_tavily.invoke(query)
    formatted = []
    for idx, r in enumerate(results, 1):
        title = r.get("title", "No Title")
        url = r.get("url", "")
        snippet = r.get("content", r.get("snippet", ""))
        formatted.append(f"{idx}. {title}\n{snippet}\n(Source: {url})")
    return "\n\n".join(formatted)

search_web = StructuredTool.from_function(
    func=tavily_search,
    name="search_web",
    description="Search the web if the knowledge base does not contain the answer."
)

tools = [query_kbf, search_web]

# --- Initialize the LLM ---

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

MATH_KEYWORDS = [
    r"\d+\s*[\+\-\*/^]\s*\d+",   
    r"integral|differentiate|derivative|limit",
    r"matrix|vector|determinant|eigen",
    r"probability|statistics|mean|variance|median",
    r"equation|quadratic|polynomial|factor|roots|zeroes",
    r"algebra|geometry|trigonometry|calculus",
]

def quick_math_check(query: str) -> bool:
    """Check if query looks mathematical using regex keywords."""
    query = query.lower()
    for pattern in MATH_KEYWORDS:
        if re.search(pattern, query):
            return True
    return False

def ai_gateway_agent(state: AgentState) -> Dict[str, Any]:
    print("--- 1. AI Gateway Agent: Checking input for relevance ---")
    user_message = state["messages"][-1]
    query = user_message.content.strip()

    # Step 1: Quick check
    if quick_math_check(query):
        return {
            "messages": [HumanMessage(content=query)],
            "user_question": query
        }

    # Step 2: If ambiguous, ask LLM to decide
    prompt = PromptTemplate.from_template("""
    You are an AI guardrail. Decide if the user query is a mathematical problem.

    Respond with only:
    - "MATH" if it is mathematical
    - "NON_MATH" otherwise

    Query: {query}
    """)
    
    guardrail_chain = prompt | llm
    response = guardrail_chain.invoke({"query": query})
    guardrail_result = response.content.strip().upper()

    if guardrail_result == "MATH":
        return {
            "messages": [HumanMessage(content=query)],
            "user_question": query
        }
    else:
        return {
            "messages": [AIMessage(content="This system only answers math-related questions.")],
            "user_question": query
        }
    
def main_math_agent(state: AgentState) -> Dict[str, Any]:
    """
    2. Main Math Routing Agent (Agentic RAG with tools):
    - LLM decides tool usage: query_kbf first, then search_web if KB fails.
    - If nothing works, respond with 'NO_ANSWER_FOUND'.
    """
    print("--- 2. Main Math Agent: Agentic RAG with Tools ---")
    messages = state["messages"]

    # Prompt gives the LLM explicit strategy
    prompt = PromptTemplate.from_template("""
    You are a Math Routing Agent with access to two tools:

    1. `query_kbf` → retrieves from the knowledge base of NCERT & JEE math.
    2. `search_web` → searches the web using Tavily.

    Rules:
    - Always try `query_kbf` first.
    - If the KB says "Not enough information in knowledge base." → call `search_web`.
    - If both fail, respond with "NO_ANSWER_FOUND".

    Decide tool use step by step based on the user's question.

    User's Question: {question}
    """)

    # Bind tools to LLM
    agent_llm = prompt | llm.bind_tools([query_kbf, search_web])
    response = agent_llm.invoke(messages)
    return {"messages": [response]}
    
def get_feedback_for_question(question: str):
    feedbacks = []
    try:
        with open("feedback_log.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:   
                    continue
                q, ans, fb = row[0], row[1], row[2]  
                if q.lower() in question.lower() or question.lower() in q.lower():
                    feedbacks.append(fb)
    except FileNotFoundError:
        return None

    return feedbacks[-1] if feedbacks else None
    
def explanation_agent(state: AgentState) -> Dict[str, Any]:
    """
    Final Explanation Agent: concise, grounded, no follow-ups, no hallucinations.
    """
    print("--- 3. Final Explanation Agent: Generating the final solution ---")
    messages = state["messages"]

    # Get the most recent tool output
    last_tool_msg = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)

    # Guard:
    if not last_tool_msg:
        return {"messages": [AIMessage(content="Not enough information in the provided context.")]}
    
    tool_output = (last_tool_msg.content or "").strip()

    # Guard: detect failure signals or too-thin context
    insufficient_signals = [
        "not enough information in knowledge base",
        "no answer found",
        "not_found",
        "no answer found even after web search",
    ]
    if (
        not tool_output
        or len(tool_output) < 40
        or any(sig in tool_output.lower() for sig in insufficient_signals)
    ):
        return {"messages": [AIMessage(content="Not enough information in the provided context.")]}
    
            # 🔹 Check for feedback
    feedback_instruction = get_feedback_for_question(state['user_question'])
    
    prompt_template = PromptTemplate.from_template("""
You are a helpful mathematical professor.

Answer **only** using the facts present in Tool Context. 
Do not use prior knowledge. If the Tool Context is insufficient to fully answer, reply exactly:
"Not enough information in the provided context."

Formatting rules:
- Keep it easy and remember not to be too long.
- Present clear, numbered steps when appropriate.
- Use LaTeX for equations, e.g. $x = (-b \pm \sqrt(b^2 - 4ac))/(2a)$.
- No markdown code blocks.
- Do not ask follow-up questions or add extra commentary.
- Refer to the feedbacks and try to improve on that : {feedback_rule}

User's Question: {user_question}

Tool Context:
{tool_context}

Final Explanation:
""")
    
    feedback_rule = (
        f"Additional refinement based on past feedback: {feedback_instruction}"
        if feedback_instruction else
        "No past feedback available."
    )
    explanation_chain = prompt_template | llm  
    final = explanation_chain.invoke({
        "user_question": state["user_question"],
        "tool_context": tool_output,
        "feedback_rule": feedback_rule
    })

    content = getattr(final, "content", str(final)).strip()

    return {"messages": [AIMessage(content=content)]}

def feedback_agent(state: AgentState) -> Dict[str, Any]:
    """Logs the feedback received from the API call."""
    print("--- 4. Feedback Agent: Logging human validation ---")
    final_answer = state["messages"][-1].content
    user_question = state["user_question"]
    feedback = state.get("feedback")

    if feedback:
        with open("feedback_log.csv", "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([user_question, final_answer, feedback])
        print(f"✅ Feedback recorded: {feedback}\n")
    else:
        print("⚠️ No feedback was provided.")
    
    return {}

def build_graph():
    memory = MemorySaver()
    tool_node = ToolNode(tools)

# --- CORRECT GRAPH ASSEMBLY ---
    workflow = StateGraph(AgentState)

# 1. Add all the nodes to the graph
    workflow.add_node("ai_gateway_agent", ai_gateway_agent)
    workflow.add_node("main_math_agent", main_math_agent)
    workflow.add_node("explanation_agent", explanation_agent)
    workflow.add_node("feedback_agent", feedback_agent)
    workflow.add_node("tools", tool_node) # Add the missing tool execution node

# 2. Set the entry point
    workflow.set_entry_point("ai_gateway_agent")

    # 3. Define how to route from the gateway
    # This requires a simple router to check if the question was valid
    def route_gateway(state: AgentState) -> str:
        if "only answers math-related questions" in state["messages"][-1].content:
            return END
        return "main_math_agent"

# 4. Add the conditional edges (The CORE of the fix)
    workflow.add_conditional_edges(
        "ai_gateway_agent",
        route_gateway # Use the router to decide where to go after the gateway
    )

# Rename your router for clarity and use it here
    def route_main_agent(state: AgentState) -> str:
        """This function routes based on whether a tool call was requested."""
        if state["messages"][-1].tool_calls:
            return "tools" # If tool call exists, go to the tool node
        else:
            return "explanation_agent" # Otherwise, go to the final explanation

    workflow.add_conditional_edges(
        "main_math_agent",
        route_main_agent # Use your logic to route from the main agent
    )

# 5. Add the final direct edges
    workflow.add_edge("tools", "main_math_agent") # This creates the crucial loop!

    workflow.add_edge("explanation_agent", "feedback_agent")
# Feedback agent is the new END
    workflow.add_edge("feedback_agent", END)
    
    return workflow.compile(checkpointer=memory,interrupt_before=["feedback_agent"])
    
app = build_graph()

