from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from notion_client import Client
from datetime import datetime, timedelta
import streamlit as st
import os
import json
import re

# Load API keys
load_dotenv()
notion = Client(auth=os.getenv("NOTION_API_KEY"))
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "NotionAgentProject"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize LLM
llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14")

# Set database ID
DATABASE_ID = "YOUR DATABASE ID"

# Get the date information
today = datetime.now().date()
yesterday = today - timedelta(days=1)
three_days_ago = today - timedelta(days=3)
last_week = today - timedelta(days=7)

# Function to add page to Notion database
def create_notion_page(job_title: str, company: str, reference: str, date: str = None, status: str = "Applied"):
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
                    {"text": {"content": reference}}
                ]
            },
            "Status": {
                "status": {"name": status}
            },
            "Date of application": {
                "date": {"start": date or datetime.now().date().isoformat()}  # ISO date string
            }
        }
    )

# Function to update the query in Notion database
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

# Function to convert natural language prompt to Notion filter JSON using LLM
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

# Function to query Notion database with a filter
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

# Function to let LLM decide what to do based on query
def get_intent_and_payload(nl_prompt: str) -> dict:
    prompt = f"""
    Classify the following user input and extract relevant fields.

    Return JSON with:
    - intent: "query", "create", or "update"
    - job_title: if available
    - company: if available
    - status: if available
    - date: optional, ISO format if present
    - reference: optional, link or reference to the job posting
    - last_updated_time: optional, ISO format if present
    
    Examples:
    Input: "I applied to Backend Engineer at Amazon yesterday"
    Output:
    {{
      "intent": "create",
      "job_title": "Backend Engineer",
      "company": "Amazon",
      "status": "Applied",
      "date": "{yesterday.isoformat()}"
    }}

    Input: "Google Software Engineer just rejected me"
    Output:
    {{
      "intent": "update",
      "job_title": "Software Engineer",
      "company": "Google",
      "status": "Rejected",
      "last_updated_time": "{datetime.now().date().isoformat()}"
    }}

    Input: "I applied to Sales Consultant in Apple 3 days ago. Reference is https://www.apple.com/"
    Output:
    {{
      "intent": "create",
      "job_title": "Sales Consultant",
      "company": "Apple",
      "status": "Applied",
      "reference": "https://www.apple.com/",
      "date": "{three_days_ago.isoformat()}"
    }}

    Input: "{nl_prompt}"
    """
    response = llm.invoke(prompt).content.strip()
    if response.startswith("```"):
        response = re.sub(r"```json|```", "", response).strip()
    return json.loads(response)

# Function to analyze job application records
def analyze_records(records: list, nl_prompt: str) -> str:
    statuses = [r.get("Status", "") for r in records]
    total = len(statuses)
    rejected = sum(1 for s in statuses if s.lower() == "rejected")
    if total == 0:
        return "No job applications found."
    rate = rejected / total * 100
    return f"You applied to {total} jobs. {rejected} were rejected. Rejection rate: {rate:.1f}%."

# Streamlit UI
st.title("Notion AI Agent")
nl_prompt = st.text_input("Ask a question:")

if st.button("Run") and nl_prompt:
    try:
        action = get_intent_and_payload(nl_prompt)

        if action["intent"] == "query":
            notion_filter = get_filter_from_llm(nl_prompt)
            desired_fields = ["Job", "Company", "Status", "Date of application"]

            filtered_records = []
            for record in query_notion_database(notion_filter):
                filtered = {k: v for k, v in record.items() if k in desired_fields}
                filtered_records.append(filtered)

            filtered_records.sort(key=lambda x: x.get("Date of application", ""), reverse=True)
            st.dataframe(filtered_records)
            st.write(analyze_records(filtered_records, nl_prompt))

        elif action["intent"] == "create":
            create_notion_page(
                company=action.get("company", "Unknown"),
                job_title=action.get("job_title", "Unknown"),
                status=action.get("status", "Applied"),
                date=action.get("date"),
                reference=action.get("reference", "Unknown")
            )
            st.success("✅ Job entry created in Notion.")

        elif action["intent"] == "update":
            full_job_title, full_company_name = update_notion_status(
                job_title=action["job_title"],
                new_status=action["status"],
                company=action["company"]
            )
            st.success(f"✅ Status for {full_job_title} at {full_company_name} has been updated to {action['status']}.")

        else:
            st.warning("⚠️ Could not determine user intent.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
