# LangChain agent
# 📘 Project Title

Main.py: An agnet to generate cover letter in word.doc format based on user's input and can customise based on user's preference.
Notion.py: Created an agent that can dynamically retrieve, analyze and update the Notion database based on user's query with Streamlit UX.

## 🚀 Features

- 🤖 AI-powered Notion assistant using OpenAI
- 📊 Retrieves and analyzes job applications
- ✍️ Creates, updates, and deletes entries in your Notion database
- 📁 Generates professional cover letters in Word format

## 📸 Screenshot

# Notion Agent

Create an empty notion database
![image](https://github.com/user-attachments/assets/6908ba55-abcd-4ba2-b267-fec02b113df7)

Run the Notion Agent and ask it to create job application entry
![image](https://github.com/user-attachments/assets/cab1a643-1fab-4414-b97e-f242a427d576)

Notion will automatically create an entry based on user's query
![image](https://github.com/user-attachments/assets/3b2582e6-4f05-4df4-875d-a40d4ff59a46)

Tell Notion Agent to update the status to "Rejected"
![image](https://github.com/user-attachments/assets/bb963eda-1c9a-4bde-a059-2e623167b80c)

The job status in Notion database will automatically be updated
![image](https://github.com/user-attachments/assets/2be89456-9e21-4a1c-9618-fccd8aeb04ce)

Notion  Agent can also retrieve and analyse the result based on user's query
![image](https://github.com/user-attachments/assets/dd621d30-01f9-4833-a1cd-e6c377621f27)
![image](https://github.com/user-attachments/assets/8d8b6207-f2a9-4c1a-833d-80dc7fde8940)



# Cover Letter Agent

Layout of the cover letter agent

![image](https://github.com/user-attachments/assets/0c94d15b-3fdf-4754-9bbc-9fd3e52ef31f)

After inputting the query, it will automatically generate the cover letter based on the input
![image](https://github.com/user-attachments/assets/dad8c654-21e7-49f5-89f6-480ca5299fdc)

It will be downloaded to Desktop by default, user can also click the "Download Word File" button to download the file

![image](https://github.com/user-attachments/assets/a8d41fc9-a07a-4e39-9205-701cfd800a5b)

Sample of the word doc generated
![image](https://github.com/user-attachments/assets/1ea2779c-4809-4305-8bcb-f27c950807a1)



## 🛠️ Tech Stack

- Python
- Streamlit
- OpenAI API (GPT-4.1)
- LangChain & LangSmith
- Notion API
