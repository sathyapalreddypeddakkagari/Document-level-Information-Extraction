# # Import the Groq API client
# from groq import Groq
#
# # Import utilities
# import json
# import os
# from dotenv import load_dotenv  # To load environment variables from a .env file
#
# # Step 1: Load API Key
# # Load environment variables from the .env file
# load_dotenv()
#
# # Retrieve the GROQ API key from environment variables
# key = os.getenv("GROQ_API_KEY")
#
# # Initialize the Groq client using the API key
# client = Groq(api_key=key)

import os
import json
import streamlit as st
from groq import Groq

# Read API key from Streamlit Secrets (Cloud) or .env (local)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found. Set it in Streamlit Secrets or .env")

client = Groq(api_key=GROQ_API_KEY)


# Step 2: Load and Preprocess Training Examples
# Function to load all JSON training files from a given folder
def load_json(folder):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):  # Only process JSON files
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                data.append(doc)
    return data  # Returns a list of loaded JSON objects

# Function to flatten nested lists of documents into a single list
def flatten_list(train_data):
    train_doc = []
    for file_docs in train_data:
        train_doc.extend(file_docs)
    return train_doc

# Load training data from the specified folder
train_folder = "DocIE_dataset_final_version/train"
train_data = load_json(train_folder)       # List of lists of documents
train_doc = flatten_list(train_data)       # Flattened list of all documents
trn_doc = train_doc[:6]                    # Select top 6 for few-shot prompting

# Prepare filtered documents (few-shot samples) with essential fields
filtered_docs = []
for doc in trn_doc:
    filtered_docs.append({
        'domain': doc['domain'],
        'title': doc['title'],
        'doc': doc['doc'],
        'triples': doc['triples']
    })

# Step 3: Build Relation Label Set
# Create a set of all unique relation labels from training data
label_set = set()
for doc in train_doc:
    for triple in doc["triples"]:
        label_set.add(triple["relation"])

# Step 4: Prompt Generation
# Generate the system-level prompt with allowed relation types
def generate_system_prompt(label_set):
    allowed_labels = ", ".join(label_set)
    return f"""
You are a skillful assistant specialized in extracting relations from documents.

Only extract relations with these types: {allowed_labels}.
Format each as [subject] — [relation] — [object].
Extract only top 20 important relations.
No extra explanations.
"""

# Generate the few-shot user prompt by embedding 3–4 labeled examples
def generate_multi_shot_prompt(filtered_docs, num_shots=4):
    prompt = "You are an expert at extracting relations from documents.\n\nHere are some examples:"
    for i, doc in enumerate(filtered_docs[:num_shots]):
        prompt += f"""

Example {i+1}:
Domain: {doc['domain']}
Title: {doc['title']}

Document:
---
{doc['doc']}
---

Extracted Relations:
"""
        for triple in doc['triples']:
            prompt += f"- {triple['head']} — {triple['relation']} — {triple['tail']}\n"

    prompt += "\n---\nNow given the following new document:"
    return prompt

# Generate system and base prompts once
system_prompt = generate_system_prompt(label_set)
base_prompt = generate_multi_shot_prompt(filtered_docs, num_shots=3)

# Step 5: Inference Function for Relation Extraction
# This function takes a new document (text) and uses the LLM to extract relations
# Given a new document (text), use the LLM to extract relations
def extract_relations_from_text(text):
    # Append the new document to the few-shot prompt
    final_prompt = base_prompt + f"""

New Document:
---
{text}
---
Relations that best explain the document:
"""
    # Call the Groq chat model with the full prompt
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},  # Role-based context
            {"role": "user", "content": final_prompt}
        ],
        model="llama-3.3-70b-versatile",  # Model selected for relation extraction
    )

    # Extract and return the response (relation list)
    return chat_completion.choices[0].message.content
