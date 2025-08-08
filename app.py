import os
import sys
import uuid
import chromadb
import pandas as pd
from flask import Flask, request, render_template
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Flask app initialization
app = Flask(__name__)

# Get API key from Render's environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Initialize Vector Database on Startup ---
client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="portfolio")

if not collection.count():
    try:
        df = pd.read_csv("my_portfolio.csv")
        for _, row in df.iterrows():
            collection.add(documents=row["Techstack"],
                           metadatas={"links": row["Links"]},
                           ids=[str(uuid.uuid4())])
        print("✅ Vector database populated with portfolio data.")
    except FileNotFoundError:
        print("❌ Error: my_portfolio.csv not found.")
        sys.exit(1)
else:
    print("✅ Vector database already exists.")

@app.route('/')
def index():
    return render_template('robot_ui.html')

@app.route('/generate', methods=['POST'])
def generate_content():
    if not GROQ_API_KEY:
        return "❌ Error: Groq API key is not set. Please add it to Render's environment variables.", 500

    job_url = request.form.get('job_url')
    if not job_url:
        return "Please provide a job URL.", 400

    # --- Validate Groq API Key ---
    try:
        llm = ChatGroq(
            temperature=0,
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192"
        )
        llm.invoke("Test LLM connection.")
    except Exception as e:
        return f"❌ Error: Invalid Groq API key or model unavailable. Details: {e}", 500

    # --- 2. Scrape and Extract Job Information ---
    try:
        loader = WebBaseLoader(job_url)
        page_data = loader.load().pop().page_content
    except Exception as e:
        return f"❌ Error scraping URL. Error: {e}", 500

    prompt_extract = PromptTemplate.from_template(
        """### SCRAPED TEXT FROM WEBSITE: {page_data}
        ### INSTRUCTION: Extract the job posting details and return them in JSON format with keys: `role`, `experience`, `skills` and `description`. Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):"""
    )
    json_parser = JsonOutputParser()
    chain_extract = prompt_extract | llm | json_parser
    job = chain_extract.invoke(input={'page_data': page_data})
    
    # --- 3. Find Relevant Portfolio Links ---
    job_skills = job.get('skills', [])
    relevant_links = collection.query(query_texts=job_skills, n_results=2).get('metadatas', [])

    # --- 4. Generate Cold Email ---
    prompt_email = PromptTemplate.from_template(
        """### JOB DESCRIPTION: {job_description}
        ### INSTRUCTION: You are Mohan, a business development executive at AtliQ. Write a cold email to the client, describing AtliQ's capabilities in fulfilling their needs. Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
        ### EMAIL (NO PREAMBLE):"""
    )
    chain_email = prompt_email | llm
    email_content = chain_email.invoke({"job_description": str(job), "link_list": relevant_links})

    return email_content.content

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
