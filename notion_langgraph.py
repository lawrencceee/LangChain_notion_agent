from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from notion_client import Client
from datetime import datetime, timedelta
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, Sequence
import streamlit as st
import os
import json
import re

# --- Setup ---
load_dotenv()
notion = Client(auth=os.getenv("NOTION_API_KEY"))
os.environ["LANGCHAIN_PROJECT"] = "NotionAgentProject"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14")
DATABASE_ID = "notion_database_id"

# --- Date helpers ---
today = datetime.now().date()
yesterday = today - timedelta(days=1)
three_days_ago = today - timedelta(days=3)
last_week = today - timedelta(days=7)

# --- State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], add_messages]
    user_input: str
    intent: str | None
    extracted_data: dict | None
    notion_filter: dict | None
    query_results: list | None
    action_taken: str | None
    error: str | None

# --- Functions ---
def get_intent_and_payload(state: AgentState) -> AgentState:
    prompt = f"""
    Classify the following user input and extract relevant fields.
    Return JSON with:
    - intent: "query", "create", or "update"
    - job_title, company, status, date, reference, last_updated_time
    Input: "{state['user_input']}"
    """
    response = llm.invoke(prompt).content.strip()
    if response.startswith("```"):
        response = re.sub(r"```json|```", "", response).strip()
    data = json.loads(response)
    return {**state, "intent": data.get("intent"), "extracted_data": data}

def validate_data(state: AgentState) -> AgentState:
    data = state.get("extracted_data", {})
    intent = state.get("intent")
    if intent == "create" and (not data.get("company") or not data.get("job_title")):
        return {**state, "error": "Missing company or job title."}
    return state

def handle_query(state: AgentState) -> AgentState:
    try:
        notion_filter = get_filter_from_llm(state["user_input"])
        results = query_notion_database(notion_filter)
        results.sort(key=lambda x: x.get("Date of application", ""), reverse=True)
        return {**state, "notion_filter": notion_filter, "query_results": results, "action_taken": "query"}
    except Exception as e:
        return {**state, "error": str(e)}

def handle_create(state: AgentState) -> AgentState:
    data = state.get("extracted_data", {})
    try:
        create_notion_page(
            company=data.get("company", "Unknown"),
            job_title=data.get("job_title", "Unknown"),
            status=data.get("status", "Applied"),
            date=data.get("date"),
            reference=data.get("reference", "Unknown")
        )
        return {**state, "action_taken": "create"}
    except Exception as e:
        return {**state, "error": str(e)}

def handle_update(state: AgentState) -> AgentState:
    data = state.get("extracted_data", {})
    try:
        update_notion_status(data["job_title"], data["company"], data["status"])
        return {**state, "action_taken": "update"}
    except Exception as e:
        return {**state, "error": str(e)}

def handle_error(state: AgentState) -> AgentState:
    return state

def get_filter_from_llm(nl_prompt: str) -> dict:
    prompt = f"""
    You are a system that converts natural language into Notion filter JSON.
    Today is {today.isoformat()}. "Last week" is {last_week.isoformat()} to {today.isoformat()}.

    Example:
    Input: "What jobs did I apply last week?"
    Output:
    {{
      "filter": {{
        "and": [
          {{"property": "Status", "status": {{"equals": "Applied"}}}},
          {{"property": "Date of application", "date": {{"on_or_after": "{last_week.isoformat()}"}}}},
          {{"property": "Date of application", "date": {{"before": "{today.isoformat()}"}}}}
        ]
      }}
    }}

    Input: "Did I apply to analyst job in citi?"
    Output:
    {{
      "filter": {{
        "and": [
          {{"property": "Status", "status": {{"is_not_empty": true}}}},
          {{"property": "Job", "title": {{"contains": "Analyst"}}}},
          {{"property": "Company", "rich_text": {{"equals": "Citi"}}}}
        ]
      }}
    }}

    Input: "How many jobs did I apply?"
    Output:
    {{
      "filter": {{
        "property": "Status",
        "status": {{
          "is_not_empty": true
        }}
      }}
    }}
 
    Input: "What jobs did I apply in mastercard?"
    Output:
    {{
      "filter": {{
        "and": [
          {{"property": "Status", "status": {{"is_not_empty": true}}}},
          {{"property": "Company", "rich_text": {{"equals": "Mastercard"}}}}
        ]
      }}
    }}

    When generating the notion query, arrange the list in ascending of the Date of application.
    Now convert this input:
    \"{nl_prompt}\"
    """
    response = llm.invoke(prompt).content

    # Try to extract only the JSON part from LLM response
    try:
        json_str = response.strip()

        # If the LLM included markdown (```json ... ```), remove it
        if json_str.startswith("```"):
            json_str = re.sub(r"```json|```", "", json_str).strip()

        # Attempt to load JSON
        return json.loads(json_str)

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON. Raw output:\n{response}\n\nError: {e}")

def query_notion_database(filter_obj: dict) -> list:

    results = notion.databases.query(
        database_id=DATABASE_ID,
        **filter_obj
    )

    records = []
    for result in results["results"]:
        props = result["properties"]
        record = {}
        for name, prop in props.items():
            prop_type = prop.get("type")
            if prop_type == "title":
                record[name] = prop["title"][0]["text"]["content"] if prop["title"] else ""
            elif prop_type == "rich_text":
                record[name] = prop["rich_text"][0]["text"]["content"] if prop["rich_text"] else ""
            elif prop_type == "status":
                record[name] = prop["status"]["name"] if prop["status"] else ""
            elif prop_type == "date":
                record[name] = prop["date"]["start"] if prop["date"] else ""
            else:
                record[name] = "[Unsupported]"
        records.append(record)
    return records

def create_notion_page(job_title: str, company: str, reference: str | None = None, date: str | None = None, status: str = "Applied"):
    notion.pages.create(
        parent={"database_id": DATABASE_ID},
        properties={
            "Company": {
                "rich_text": [
                    {"text": {"content": company}}
                ]
            },
            "Job": {
                "title": [
                    {"text": {"content": job_title}}
                ]
            },
            "Reference": {
                "rich_text": [
                    {"text": {"content": reference or ""}}  # fallback to empty string
                ]
            },
            "Status": {
                "status": {"name": status}
            },
            "Date of application": {
                "date": {"start": date or datetime.now().date().isoformat()}
            }
        }
    )

def update_notion_status(job_title: str, company: str, new_status: str) -> tuple:
    search = notion.databases.query(
        database_id=DATABASE_ID,
        filter={
            "and": [
                {
                    "property": "Job",
                    "title": {
                        "contains": job_title
                    }
                },
                {
                    "property": "Company",
                    "rich_text": {
                        "contains": company
                    }
                }
            ]
        }
    )


    if not search["results"]:
        raise ValueError(f"No entry found with job title: {job_title} and company containing '{company}'")
    
    page = search["results"][0]
    page_id = page["id"]

    # Extract title
    job_title_data = page["properties"].get("Job", {}).get("title", [])
    full_job_title = job_title_data[0]["text"]["content"] if job_title_data else "(unknown job title)"

    company_data = page["properties"].get("Company", {}).get("rich_text", [])
    full_company_name = company_data[0]["text"]["content"] if company_data else "(unknown company)"

    # Update the page
    notion.pages.update(
        page_id=page_id,
        properties={
            "Status": {
                "status": {"name": new_status}  # also correct
            },
            "Last updated time": {
                "date": {"start": datetime.now().date().isoformat()}
            }
        }
    )

    return full_job_title, full_company_name

def router(state: AgentState) -> str:
    if state.get("error"):
        return "error"
    
    intent = state.get("intent")
    if intent in ["query", "create", "update"]:
        return intent  # ✅ Return the KEY from the conditional_edges dict
    return "error"

# Function to analyze job application records
def analyze_records(records: list, nl_prompt: str) -> str:
    statuses = [r.get("Status", "") for r in records]
    total = len(statuses)
    rejected = sum(1 for s in statuses if s.lower() == "rejected")
    if total == 0:
        return "No job applications found."
    rate = rejected / total * 100
    return f"You applied to {total} jobs. {rejected} were rejected. Rejection rate: {rate:.1f}%."

# --- LangGraph setup ---
graph = StateGraph(AgentState)
graph.add_node("intent", get_intent_and_payload)
graph.add_node("validate", validate_data)
graph.add_node("handle_query", handle_query)
graph.add_node("handle_create", handle_create)
graph.add_node("handle_update", handle_update)
graph.add_node("handle_error", handle_error)

graph.add_edge(START, "intent")
graph.add_edge("intent", "validate")
graph.add_conditional_edges("validate", router, {
    "query": "handle_query",
    "create": "handle_create",
    "update": "handle_update",
    "handle_error": "handle_error"
})
graph.add_edge("handle_query", END)
graph.add_edge("handle_create", END)
graph.add_edge("handle_update", END)
graph.add_edge("handle_error", END)

app = graph.compile()

# --- Streamlit UI ---
st.title("Notion LangGraph Agent")
prompt = st.text_input("Ask about your job applications:")

if st.button("Run") and prompt:
    try:
        result = app.invoke({
            "messages": [],
            "user_input": prompt,
            "intent": None,
            "extracted_data": None,
            "notion_filter": None,
            "query_results": None,
            "action_taken": None,
            "error": None,
            "confirmation_data": None,
            "needs_confirmation": False
        })

        # 🛑 Error Handling
        if result.get("error"):
            st.error(f"❌ {result['error']}")
        
        # 🔍 Query Results
        elif result.get("intent") == "query" and result.get("query_results"):
            query_results = result["query_results"]
            query_results.sort(key=lambda x: x.get("Date of application", ""), reverse=True)
            st.dataframe(query_results)
            summary = analyze_records(query_results, prompt)
            st.write(summary)

        # ✅ Creation Feedback
        elif result.get("intent") == "create":
            if result.get("action_taken") == "create":
                st.success("✅ Job entry created in Notion.")
            elif result.get("needs_confirmation") and result.get("confirmation_data"):
                st.warning(f"⚠️ {result['confirmation_data']['message']}")

        # 🔄 Update Feedback
        elif result.get("intent") == "update":
            if result.get("action_taken") == "update":
                extracted = result.get("extracted_data", {})
                job = extracted.get("job_title", "(job title unknown)")
                company = extracted.get("company", "(company unknown)")
                status = extracted.get("status", "(status unknown)")
                st.success(f"✅ Status for {job} at {company} has been updated to {status}.")
            elif result.get("needs_confirmation") and result.get("confirmation_data"):
                st.warning(f"⚠️ {result['confirmation_data']['message']}")

        # ❓ Fallback
        else:
            st.warning("⚠️ Could not determine user intent or no action was taken.")

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
