from dotenv import load_dotenv
from docx import Document
from pathlib import Path
from langchain.chat_models import ChatOpenAI

import streamlit as st
import os
import asyncio

load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "AIAgentProject"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14")

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

st.title("AI Agent generating Cover Letters")
st.write("Input the info below:")

input_text = st.text_input("Enter your question:", "")
company_name = st.text_input("Enter the company name:", "")
job_title = st.text_input("Enter the job title:", "")
company_address = st.text_input("Enter the company address:", "")
file_name = st.text_input("Enter a filename (without extension):", value="")

if input_text and file_name and company_name and job_title:
    prompt = f"""
    Write a professional cover letter for the position of {job_title} at {company_name}, located at {company_address}.
    My name is 'XXX', email is 'XXX', phone number is 'XXX'.  Just put these info in the cover letter. Date use today's date.
    Make the letter personalized, concise, and convincing. Use the following additional instructions from the user:

    {input_text}
    """

    generated_text = llm.invoke(prompt).content
    st.write("Response:", generated_text)

    # Save to Desktop
    desktop_path = Path.home() / "Desktop"
    file_path = desktop_path / f"{file_name}.docx"

    doc = Document()
    doc.add_paragraph(generated_text)
    doc.save(file_path)

    st.success(f"Document saved to {file_path}")
    with open(file_path, "rb") as f:
        st.download_button("Download Word File", f, file_name=f"{file_name}.docx")
