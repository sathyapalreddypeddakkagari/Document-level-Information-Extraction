# DocIE: AI driven Document-Level Information Extraction 📑🤖

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://doc-ie.streamlit.app/](https://document-level-information-extraction-ait.streamlit.app/)) 
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DocIE** is an end-to-end information extraction pipeline designed to transform unstructured documents into structured insights. Built with **Streamlit**, it integrates custom-trained NLP models and cutting-edge LLMs to perform Named Entity Recognition (NER), Relation Extraction (RE), and intelligent Summarization.

---

## 🚀 Key Features

- **Custom NER Engine**: Leverages a fine-tuned **spaCy** model to identify entities like `PERSON`, `ORG`, and `EVENT` with high precision.
- **LLM-Powered Relation Extraction**: Uses **LLaMA-3.3-70B** (via Groq API) to map complex relationships between entities using few-shot prompting.
- **N-Gram Summarization**: Implements a trigram frequency-based scoring algorithm to extract the most significant sentences from long documents.
- **Multi-Format Support**: Seamlessly process both `.txt` and `.pdf` files.
- **Interactive UI**: A clean, dashboard-style interface built on Streamlit for real-time analysis.

---

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **NLP Framework**: [spaCy](https://spacy.io/) (Custom Model)
* **Inference Engine**: [Groq Cloud API](https://groq.com/)
* **Large Language Model**: LLaMA-3.3-70B-Versatile
* **Language**: Python 3.11
* **Python Libraries**: numpy, pandas, requests, python-dotenv, pillow, tqdm, protobuf

---

## 📂 Project Structure

```text
DocIE/
├── .streamlit/              # Streamlit configuration (optional)
│   └── config.toml
├── data/                    # Sample documents for testing
│   └── test_train_dev.zip
├── models/                  # Directory for the fine-tuned model
│   └── Finetuned_NER_model/
├── src/                     # Core logic source files
│   ├── app.py               # Main UI
│   ├── spacy_ner.py         # NER module
│   ├── relation_ext.py      # RE module
│   └── summary_doc.py       # Summarization module
├── .env.example             # Template for GROQ_API_KEY
├── .gitignore               # Files to exclude from Git (e.g., .env, __pycache__)
├── packages.txt             # OS-level dependencies
├── requirements.txt         # Python dependencies
├── runtime.txt              # Specifies Python version according to dependencies
└── README.md                # Project documentation
