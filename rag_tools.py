# rag_tools.py
# PySpur-based tool orchestration. All routing is via run_tool().
# Contains all tool logic, LLM completions, and web search integration.

import os
import spacy
import requests
from dotenv import load_dotenv
from groq import Client
from pyspur.nodes.decorator import tool_function
from embedding import embed_text
from vector_store import query_discharge_notes, query_clinical_trials

# Load environment variables from .env if present (for local development)
load_dotenv()

# LLM API config
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Client(api_key=GROQ_API_KEY)

# SerpAPI config
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

SYMPTOM_HINTS = [
    "chest pain", "shortness of breath", "fatigue", "dizziness",
    "nausea", "vomiting", "palpitations", "sweating", "jaw pain",
    "arm pain", "back pain", "tightness", "pressure in chest",
    "arrhythmia", "tachycardia", "bradycardia", "angina",
    "edema", "dyspnea", "syncope", "lightheadedness",
    "ejection fraction", "myocardial infarction", "heart failure",
    "cardiomyopathy", "cardiac arrest"
]

nlp = spacy.load("en_core_web_sm")

def generate_response(messages: list, model: str = "llama3-70b-8192"):
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "[Error] GROQ_API_KEY is missing or not loaded from .env"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Groq API Error] {e}"

@tool_function(name="chat_memory_tool")
def chat_memory_tool(memory: str, chat_history=None, model: str = "llama-3.3-70b-versatile") -> str:
    doc = nlp(memory)
    found_symptoms = set(
        keyword for chunk in doc.noun_chunks for keyword in SYMPTOM_HINTS if keyword in chunk.text.lower()
    )
    symptom_context = (
        f"Previously mentioned symptoms include: {', '.join(found_symptoms)}."
        if found_symptoms else "No clear symptoms found in memory."
    )
    messages = [
        {"role": "system", "content": "You are a medical assistant summarizing prior symptoms from memory."},
        {"role": "assistant", "content": memory},
        {"role": "user", "content": (
            f"The patient previously reported: {memory}\n\n"
            f"Symptoms extracted: {symptom_context}\n"
            "Please provide a clear, concise, and helpful summary of these symptoms and suggest next steps."
        )}
    ]
    return generate_response(messages, model=model)

@tool_function(name="treatment_tool")
def treatment_tool(query: str, chat_history=None, model: str = "llama-3.3-70b-versatile", use_rag: bool = True) -> str:
    try:
        query_embedding = embed_text(query)
        if use_rag:
            top_docs = query_discharge_notes(query_embedding, n_results=5)
            top_docs = [doc[:1500] for doc in top_docs]
            combined_context = "\n\n".join(top_docs)
            messages = [
                {"role": "system", "content": "You are a medically accurate and safety-focused clinical assistant."},
                {"role": "assistant", "content": combined_context},
                {"role": "user", "content": (
                    f"Patient condition: {query}.\n\n"
                    "Based on the following discharge notes, recommend essential treatment."
                )}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a medically accurate and safety-focused clinical assistant."},
                {"role": "user", "content": f"Patient condition: {query}. What treatment is recommended?"}
            ]
        result = generate_response(messages, model=model)
        if use_rag:
            return result + "\n\n_Powered by Groq API + RAG_"
        else:
            return result + "\n\n_Powered by Groq API_"
    except Exception as e:
        return f"Error: {str(e)}"

@tool_function(name="symptom_search_tool")
def symptom_search_tool(symptom_description: str, chat_history=None, model: str = "llama-3.3-70b-versatile") -> str:
    load_dotenv()
    serp_api_key = os.environ.get("SERPAPI_KEY")
    if not serp_api_key:
        return "[Error] Missing SERPAPI_KEY in environment variables."
    
    def perform_search(query):
        params = {
            "engine": "google",
            "q": f"{query} possible causes site:mayoclinic.org OR site:webmd.com OR site:nih.gov",
            "api_key": serp_api_key
        }
        response = requests.get("https://serpapi.com/search", params=params)
        return response.json().get("organic_results", [])

    try:
        results = perform_search(symptom_description)
        if not results:
            return "No reliable medical source found."

        snippets_with_citations = []
        sources = []
        for res in results[:3]:
            if 'snippet' in res and 'link' in res:
                url = res['link']
                domain = url.split("//")[-1].split("/")[0].replace("www.", "")
                snippets_with_citations.append(f"{res['snippet']} (Source: {domain})")
                sources.append(url)

        search_context = "\n\n".join(snippets_with_citations)
        messages = [
            {"role": "system", "content": "You are a medical assistant using trusted web sources to explain symptom causes."},
            {"role": "assistant", "content": search_context},
            {"role": "user", "content": f"What could be the cause of: {symptom_description}?"}
        ]
        answer = generate_response(messages, model=model)
        return answer + "\n\n**Sources:**\n" + "\n".join(f"- {url}" for url in sources) + "\n\n_Powered by SERP API + Groq API_"
    except Exception as e:
        return f"Search error: {str(e)}"

@tool_function(name="trial_matcher_tool")
def trial_matcher_tool(discharge_note: str, chat_history=None, model: str = "llama-3.3-70b-versatile", use_rag: bool = True) -> str:
    try:
        query_embedding = embed_text(discharge_note)
        results = query_clinical_trials(query_embedding, n_results=3)
        documents = results["documents"]
        metadatas = results["metadatas"]

        if not documents:
            return "No matching clinical trials were found for the provided note."

        summaries = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            nct_id = meta.get("nct_id", "Unknown ID")
            truncated_doc = doc.strip()[:1500]
            if use_rag:
                summary_prompt = (
                    f"You are a clinical assistant reviewing a matched clinical trial.\n"
                    f"Summarize the trial using **bullet points only** for the following fields:\n"
                    f"- NCT ID\n- Study Title\n- Conditions\n- Inclusion Criteria\n- Exclusion Criteria\n\n"
                    f"Use bullets under each field. Maintain a clean format. Respond only with the summary.\n\n"
                    f"Trial Description:\nNCT ID: {nct_id}\n{truncated_doc}"
                )
                messages = [
                    {"role": "system", "content": "You are a medically precise clinical research assistant."},
                    {"role": "user", "content": summary_prompt}
                ]
                summary = generate_response(messages, model=model)
                summaries.append(f"### Trial {i+1}:\n{summary}")
            else:
                summaries.append(f"### Trial {i+1}:\nNCT ID: {nct_id}\n\n{truncated_doc}")
        result = "\n\n---\n\n".join(summaries)
        if use_rag:
            return result + "\n\n_Powered by Groq API + RAG_"
        else:
            return result + "\n\n_Powered by Groq API_"
    except Exception as e:
        return f"Error during trial matching: {str(e)}"

# Tool routing via keyword logic
TOOL_ROUTER = {
    "symptom": ("symptom_search_tool", False),
    "treatment": ("treatment_tool", True),
    "trial": ("trial_matcher_tool", True)
}

TOOL_FUNCTIONS = {
    "chat_memory_tool": chat_memory_tool,
    "treatment_tool": treatment_tool,
    "symptom_search_tool": symptom_search_tool,
    "trial_matcher_tool": trial_matcher_tool
}

def run_tool(query: str, chat_history, model: str, use_rag: bool) -> str:
    for keyword, (tool_name, supports_rag) in TOOL_ROUTER.items():
        if keyword in query.lower():
            tool_func = TOOL_FUNCTIONS[tool_name]
            if supports_rag:
                return tool_func(query, chat_history, model=model, use_rag=use_rag)
            else:
                return tool_func(query, chat_history, model=model)
    return chat_memory_tool(query, chat_history, model=model)
