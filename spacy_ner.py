# Import the spaCy library and its visualization module
import spacy
from spacy import displacy

# Load the fine-tuned Named Entity Recognition (NER) model
# The model is a Spacy "en_core_Web_lg" custom-trained model stored in the folder "Finetuned_NER_model"
nlp_custom = spacy.load("Finetuned_NER_model")

# Function to prompt the user for a document input
def get_input_doc():
    # Ask the user to enter some text (i.e., a document)
    input_doc = input("Please enter the document text: ")
    return input_doc  # Return the entered text

# Function to process the user's input using the loaded NER model
def process_input_doc(input_doc):
    # Use the custom NER model to process the input document
    doc = nlp_custom(input_doc)
    return doc  # Return the processed spaCy Doc object

# Function to display the recognized entities visually using displacy
def display_entities(doc):
    # Render the entities found in the document using displaCy
    # style="ent" highlights named entities in the text
    # jupyter=False indicates that the output should be returned as HTML (not for use in Jupyter Notebook)
    return displacy.render(doc, style="ent", jupyter=False)
