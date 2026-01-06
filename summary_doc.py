# Import necessary modules
import re
import string
from collections import Counter, defaultdict
import spacy

# Import the custom fine-tuned spaCy model for NER (used here for sentence splitting)
from spacy_ner import nlp_custom

def summarize_document(text, n=2, top_k=5):
    """
    Summarize the document using N-gram frequency scoring.

    Args:
        text (str): The full input document as a string.
        n (int): The size of N-grams to use for scoring (default is 2 for bigrams).
        top_k (int): The number of most important sentences to return.

    Returns:
        list: A list of the top-k important sentences forming the summary.
    """

    # Step 1: Sentence Segmentation
    # Ensure that the custom spaCy pipeline has a sentencizer component for sentence detection
    if "sentencizer" not in nlp_custom.pipe_names:
        nlp_custom.add_pipe("sentencizer")

    # Process the document using spaCy to split into sentences
    doc = nlp_custom(text)
    sentences = list(doc.sents)  # List of spaCy Span objects representing sentences

    # Step 2: Preprocess Sentences
    # Convert sentences to lowercase and remove punctuation for N-gram scoring
    clean_sentences = [
        re.sub(rf"[{string.punctuation}]", "", sent.text.lower())
        for sent in sentences
    ]

    # Step 3: Compute N-gram Frequencies
    ngram_counts = Counter()
    for sent in clean_sentences:
        tokens = sent.split()
        ngrams = zip(*[tokens[i:] for i in range(n)])  # Create n-grams from tokens
        ngram_counts.update(ngrams)  # Update overall N-gram frequency counts

    # Step 4: Score Sentences
    sent_scores = defaultdict(int)  # Dictionary to store sentence scores
    for i, sent in enumerate(clean_sentences):
        tokens = sent.split()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        for ng in ngrams:
            sent_scores[i] += ngram_counts[ng]  # Add frequency of each N-gram to sentence's score

    # Step 5: Select Top-k Sentences
    # Rank sentence indices by score (descending), then select top_k
    top_sent_ids = sorted(sent_scores, key=sent_scores.get, reverse=True)[:top_k]
    top_sent_ids.sort()  # Sort again to maintain original document order in summary

    # Return the original (unmodified) sentences based on selected indices
    return [sentences[i].text.strip() for i in top_sent_ids]
