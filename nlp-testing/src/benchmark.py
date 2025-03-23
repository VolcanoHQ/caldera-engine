# src/benchmark.py
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import os
import json
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Replace with your Azure Text Analytics endpoint and key
endpoint = "https://<your-text-analytics-endpoint>.cognitiveservices.azure.com/"
key = "<your-text-analytics-key>"

# Create a client
credential = AzureKeyCredential(key)
client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

# Load spaCy models
nlp_sm = spacy.load("en_core_web_sm")
nlp_lg = spacy.load("en_core_web_lg")
nlp_lg.add_pipe("spacytextblob") # Add the extension to the pipeline.


# Azure client setup function
def get_azure_client(key, endpoint):
    credential = AzureKeyCredential(key)
    client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
    return client

# Main benchmark function
def benchmark_performance(text, task="all", libraries=["spacy", "nltk", "azure"], azure_client=None):
    """
    Benchmark NLP libraries on various tasks
    
    Parameters:
    - text: text to analyze
    - task: specific task to benchmark or "all"
    - libraries: list of libraries to test
    - azure_client: optional Azure client for Azure tests
    
    Returns:
    - Dictionary with results
    """
    results = {}
    
    # Define test functions for each task and library
    tests = {
        "dialogue_detection": {
            "spacy": benchmark_dialogue_detection_spacy,
            "nltk": benchmark_dialogue_detection_nltk,
            "azure": benchmark_dialogue_detection_azure
        },
        "ner": {
            "spacy": benchmark_ner_spacy,
            "nltk": benchmark_ner_nltk,
            "azure": benchmark_ner_azure
        },
        "sentiment": {
            "spacy": benchmark_sentiment_spacy,
            "nltk": benchmark_sentiment_nltk,
            "azure": benchmark_sentiment_azure
        },
        "processing_speed": {
            "spacy": benchmark_speed_spacy,
            "nltk": benchmark_speed_nltk,
            "azure": benchmark_speed_azure
        }
    }
    
    # Run selected tests
    if task == "all":
        for t in tests:
            results[t] = {}
            for lib in libraries:
                if lib in tests[t]:
                    if lib == "azure" and not azure_client:
                        continue  # Skip Azure tests if no client provided
                    results[t][lib] = tests[t][lib](text, azure_client) if lib == "azure" else tests[t][lib](text)
    else:
        results[task] = {}
        for lib in libraries:
            if lib in tests[task]:
                if lib == "azure" and not azure_client:
                    continue  # Skip Azure tests if no client provided
                results[task][lib] = tests[task][lib](text, azure_client) if lib == "azure" else tests[task][lib](text)
                
    return results

# Dialogue Detection implementations
def benchmark_dialogue_detection_spacy(text):
    start_time = time.time()
    doc = nlp_lg(text)
    
    # Simplistic dialogue detection using quotation marks and speech verbs
    dialogue_sentences = []
    for sent in doc.sents:
        text = sent.text.strip()
        # Check for quotation marks
        if ('"' in text or "'" in text) and any(token.pos_ == "VERB" and token.lemma_ in ["say", "tell", "ask", "reply", "respond", "shout", "whisper"] for token in sent):
            dialogue_sentences.append(text)
    
    end_time = time.time()
    return {
        "detected_dialogue": dialogue_sentences,
        "time_taken": end_time - start_time,
        "count": len(dialogue_sentences)
    }

def benchmark_dialogue_detection_nltk(text):
    start_time = time.time()
    
    sentences = nltk.sent_tokenize(text)
    dialogue_sentences = []
    
    for sent in sentences:
        # Simple tokenization and tagging
        tokens = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)
        
        # Check for quotation marks and speech verbs
        has_quotes = any(token in ['"', "'"] for token in tokens)
        speech_verbs = ["say", "tell", "ask", "reply", "respond", "shout", "whisper"]
        has_speech_verb = any(token.lower() in speech_verbs for token, tag in tagged if tag.startswith('VB'))
        
        if has_quotes and has_speech_verb:
            dialogue_sentences.append(sent)
    
    end_time = time.time()
    return {
        "detected_dialogue": dialogue_sentences,
        "time_taken": end_time - start_time,
        "count": len(dialogue_sentences)
    }

def benchmark_dialogue_detection_azure(text, client):
    start_time = time.time()
    
    # Azure doesn't have a direct dialogue detection feature
    # We'll use sentence splitting and then look for quotation patterns
    try:
        # Split text into smaller chunks due to Azure limits
        max_chunk_size = 5000
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        dialogue_sentences = []
        for chunk in chunks:
            # Use Azure's sentence breaking
            responses = client.extract_key_phrases(documents=[chunk])
            
            # For a real implementation, we'd use a better way to get sentences
            # For now, just use NLTK's sentence tokenizer
            sentences = nltk.sent_tokenize(chunk)
            
            for sent in sentences:
                if ('"' in sent or "'" in sent):
                    dialogue_sentences.append(sent)
    except Exception as e:
        print(f"Azure processing error: {e}")
        dialogue_sentences = []
    
    end_time = time.time()
    return {
        "detected_dialogue": dialogue_sentences,
        "time_taken": end_time - start_time,
        "count": len(dialogue_sentences)
    }

# NER implementations
def benchmark_ner_spacy(text):
    start_time = time.time()
    doc = nlp_lg(text)
    
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "PERSON"]
    
    end_time = time.time()
    return {
        "entities": entities,
        "time_taken": end_time - start_time,
        "count": len(entities)
    }

def benchmark_ner_nltk(text):
    start_time = time.time()
    
    entities = []
    sentences = nltk.sent_tokenize(text)
    
    for sent in sentences:
        tokens = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(tagged)
        
        # Extract person entities
        for chunk in chunks:
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                entities.append((' '.join(c[0] for c in chunk), 'PERSON'))
    
    end_time = time.time()
    return {
        "entities": entities,
        "time_taken": end_time - start_time,
        "count": len(entities)
    }

def benchmark_ner_azure(text, client):
    start_time = time.time()
    
    try:
        # Split text into smaller chunks due to Azure limits
        max_chunk_size = 5000
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        entities = []
        for chunk in chunks:
            response = client.recognize_entities(documents=[chunk])[0]
            
            for entity in response.entities:
                if entity.category == "Person":
                    entities.append((entity.text, "PERSON"))
    except Exception as e:
        print(f"Azure processing error: {e}")
        entities = []
    
    end_time = time.time()
    return {
        "entities": entities,
        "time_taken": end_time - start_time,
        "count": len(entities)
    }

# Sentiment analysis implementations
def benchmark_sentiment_spacy(text):
    start_time = time.time()
    doc = nlp_lg(text)

    # Extract sentiment scores using spacytextblob
    sentiment_scores = {
        "compound": doc._.blob.polarity,
        "positive": max(0, doc._.blob.polarity),  # Simplified positive score
        "negative": max(0, -doc._.blob.polarity),  # Simplified negative score
        "neutral": 1 - abs(doc._.blob.polarity),  # Simplified neutral score
        "subjectivity": doc._.blob.subjectivity
    }

    # Placeholder for emotional sentences (you might need to refine this)
    emotional_sentences = []
    
    # Extract the assessments
    assessments = doc._.blob.sentiment_assessments.assessments
    
    # Sort the assessments by the absolute value of their polarity
    sorted_assessments = sorted(assessments, key=lambda x: abs(x[1]), reverse=True)
    
    # Extract the top 5 most emotional sentences
    for assessment in sorted_assessments[:5]:
        emotional_sentences.append(assessment[0])

    end_time = time.time()
    return {
        "sentiment_scores": sentiment_scores,
        "emotional_sentences": emotional_sentences,
        "time_taken": end_time - start_time,
    }


def benchmark_sentiment_nltk(text):
    start_time = time.time()
    sid = SentimentIntensityAnalyzer()
    
    # Split text into paragraphs or sentences
    sentences = nltk.sent_tokenize(text)
    sentiment_scores = [sid.polarity_scores(sentence) for sentence in sentences]
    
    # Find most emotional sentences
    emotional_sentences = sorted(
        [(sentences[i], scores["compound"]) for i, scores in enumerate(sentiment_scores)],
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]
    
    overall_sentiment = {
        "compound": sum(score["compound"] for score in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
        "positive": sum(score["pos"] for score in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
        "negative": sum(score["neg"] for score in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
        "neutral": sum(score["neu"] for score in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    }
    
    end_time = time.time()
    return {
        "sentiment_scores": overall_sentiment,
        "emotional_sentences": emotional_sentences,
        "time_taken": end_time - start_time
    }

def benchmark_sentiment_azure(text, client):
    start_time = time.time()
    
    try:
        # Split text into smaller chunks due to Azure limits
        max_chunk_size = 5000
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        sentiment_scores = []
        for chunk in chunks:
            response = client.analyze_sentiment(documents=[chunk])[0]
            sentiment_scores.append({
                "compound": response.confidence_scores.neutral,  # There is no direct compound score
                "positive": response.confidence_scores.positive,
                "negative": response.confidence_scores.negative,
                "neutral": response.confidence_scores.neutral
            })
        
        # Just using sentence splitting from NLTK for simplicity
        sentences = nltk.sent_tokenize(text)
        sentence_sentiments = []
        
        for i in range(min(len(sentences), 20)):
            if len(sentences[i]) < 5000:
                response = client.analyze_sentiment(documents=[sentences[i]])[0]
                sentence_sentiments.append((
                    sentences[i],
                    max(response.confidence_scores.positive, response.confidence_scores.negative)
                ))
        
        emotional_sentences = sorted(sentence_sentiments, key=lambda x: x[1], reverse=True)[:5]
        
        # Average the sentiment scores
        overall_sentiment = {
            "compound": sum(score["compound"] for score in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
            "positive": sum(score["positive"] for score in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
            "negative": sum(score["negative"] for score in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
            "neutral": sum(score["neutral"] for score in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        }
    except Exception as e:
        print(f"Azure processing error: {e}")
        overall_sentiment = {"compound": 0, "positive": 0, "negative": 0, "neutral": 0}
        emotional_sentences = []
    
    end_time = time.time()
    return {
        "sentiment_scores": overall_sentiment,
        "emotional_sentences": emotional_sentences,
        "time_taken": end_time - start_time
    }

# Processing speed implementations
def benchmark_speed_spacy(text):
    start_time = time.time()
    doc = nlp_lg(text)
    parsed_time = time.time() - start_time
    
    chars_per_second = len(text) / parsed_time if parsed_time > 0 else 0
    
    return {
        "time_taken": parsed_time,
        "chars_per_second": chars_per_second,
        "text_length": len(text)
    }

def benchmark_speed_nltk(text):
    start_time = time.time()
    
    # Perform typical NLTK operations
    sentences = nltk.sent_tokenize(text)
    tokens = []
    for sent in sentences:
        tokens.extend(nltk.word_tokenize(sent))
    tagged = nltk.pos_tag(tokens[:5000])  # Limit to 5000 tokens for speed
    
    parsed_time = time.time() - start_time
    chars_per_second = len(text) / parsed_time if parsed_time > 0 else 0
    
    return {
        "time_taken": parsed_time,
        "chars_per_second": chars_per_second,
        "text_length": len(text)
    }

def benchmark_speed_azure(text, client):
    start_time = time.time()
    
    try:
        # Split text into smaller chunks due to Azure limits
        max_chunk_size = 5000
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        for chunk in chunks[:1]:  # Just test with first chunk for timing
            _ = client.analyze_sentiment(documents=[chunk])
        
        parsed_time = time.time() - start_time
        chars_per_second = len(chunks[0]) / parsed_time if parsed_time > 0 else 0
    except Exception as e:
        print(f"Azure processing error: {e}")
        parsed_time = 0
        chars_per_second = 0
    
    return {
        "time_taken": parsed_time,
        "chars_per_second": chars_per_second,
        "text_length": len(text)
    }