
# Document-level Information Extraction (DocIE)

# Import Streamlit for building the web app UI
import streamlit as st

# Import custom modules for NER, relation extraction, and summarization
from spacy_ner import *         # Contains process_input_doc() and display_entities()
from relation_ext import *      # Contains extract_relations_from_text()
from summary_doc import *       # Contains summarize_document()

# Set the main title of the Streamlit app
st.title("Document-level Information Extraction (DocIE)")
st.write("Application for document-level information extraction using NER and relation extraction.")

# Allow user to upload a document (only .txt or .pdf files are accepted)
uploaded_file = st.file_uploader("Upload a text document", type=["txt", "pdf"])

# Check if a file was uploaded
if uploaded_file is not None:
    # Read the content of the uploaded file and decode it as UTF-8 text
    text = uploaded_file.read().decode("utf-8")
    
    # Display the uploaded document's content summary
    st.subheader("Uploaded Document")
    st.write("Document length:", len(text), "characters")
    
    # Store the document text in session state so it's accessible across sections
    st.session_state["doc_text"] = text

    # === NER Section ===
    st.subheader("Named Entity Recognition (NER)")
    st.write("Extract named entities from the document using a pre-trained NER model.")

    # NER button: Runs the NER model on the document text
    if st.button("NER"):
        result = process_input_doc(text)      # Process text using the NER model
        html = display_entities(result)       # Generate HTML visualization of entities
        st.session_state["ner_html"] = html   # Store the result in session state

    # If NER was already run, display the HTML-rendered output
    if "ner_html" in st.session_state:
        st.markdown(st.session_state["ner_html"], unsafe_allow_html=True)

    # === Relation Extraction Section ===
    st.subheader("Relation Extraction")
    st.write("Extract relations between entities in the document using LLM 'llama-3.3-70b-versatile'.")

    # RE button: Calls the relation extraction function
    if st.button("RE"):
        relations = extract_relations_from_text(text)  # Extract relations using an LLM or rule-based system
        st.session_state["re_text"] = relations        # Store in session state

    # Display the extracted relations if available
    if "re_text" in st.session_state:
        st.write("Obtained Relations from the Document")
        st.markdown(st.session_state["re_text"], unsafe_allow_html=True)

    # === Document Summary Section ===
    st.subheader("Document Summary")
    st.write("Small summary of the document using N-gram frequency scoring.")

    # Summary button: Calls the summarization function
    if st.button("Summary"):
        summary = summarize_document(text, n=3, top_k=3)  # Summarize the document with trigrams and top 3 sentences
        st.session_state["summary_text"] = summary        # Store summary in session state

    # Display the summary if it has been computed
    if "summary_text" in st.session_state:
        for sentence in st.session_state["summary_text"]:
            st.markdown(sentence)  # Render each summary sentence
