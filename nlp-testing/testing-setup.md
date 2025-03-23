# Firespeaker NLP Testing Setup

## Setting Up the Hands-On Testing Environment with Anaconda and VS Code

### 1. Environment Setup with Conda

```bash
# Create a new conda environment with Python 3.10
conda create -n firespeaker-nlp python=3.10
conda activate firespeaker-nlp

# Create a new directory for our testing
mkdir firespeaker-nlp-testing
cd firespeaker-nlp-testing
```

### 2. Install Required Libraries with Conda

```bash
# Install core libraries
conda install -c conda-forge spacy nltk pandas matplotlib jupyter
conda install -c microsoft azure-ai-textanalytics

# Install VS Code Jupiter integration if needed
conda install -c conda-forge ipykernel

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words'); nltk.download('vader_lexicon')"
```

### 3. Setting Up VS Code

1. Install VS Code extensions:
   - Python extension
   - Jupyter extension
   - Pylance (for better Python language support)

2. Open VS Code and navigate to your project:
   ```bash
   cd firespeaker-nlp-testing
   code .
   ```

3. Create the project structure in VS Code:
   ```
   firespeaker-nlp-testing/
   ├── data/
   │   ├── corpus/          # For text samples
   │   └── ground_truth/    # For annotations
   ├── notebooks/
   │   └── nlp_benchmarks.ipynb
   └── src/
       ├── __init__.py
       ├── benchmark.py     # Core benchmark functions
       ├── evaluators.py    # Evaluation metrics
       └── visualize.py     # Visualization utilities
   ```

4. Create a new Jupyter notebook in VS Code:
   - Click on the Explorer icon in the sidebar
   - Navigate to the notebooks folder
   - Right-click and select "New File"
   - Name it "nlp_benchmarks.ipynb"
   - Select the firespeaker-nlp kernel when prompted

### 4. Create Test Corpus

Create a directory structure for your test corpus:

```bash
mkdir -p data/corpus/classics
mkdir -p data/corpus/modern
mkdir -p data/corpus/dialogue_heavy
mkdir -p data/corpus/descriptive
mkdir -p data/corpus/emotional
```

You can download sample texts from Project Gutenberg and save them in the appropriate directories.

### 5. Create the Benchmark Code Framework

Let's organize our code more modularly for a VS Code project:

1. First, create the `src/benchmark.py` file:

```python
# src/benchmark.py
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import os
import json
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Load spaCy models
nlp_sm = spacy.load("en_core_web_sm")
nlp_lg = spacy.load("en_core_web_lg")

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
    # spaCy doesn't have built-in sentiment analysis
    # We could use a pipeline or extension, but will skip for now
    start_time = time.time()
    doc = nlp_lg(text)
    
    # Placeholder for sentiment
    sentiment_scores = {"compound": 0, "positive": 0, "negative": 0, "neutral": 0}
    emotional_sentences = []
    
    end_time = time.time()
    return {
        "sentiment_scores": sentiment_scores,
        "emotional_sentences": emotional_sentences,
        "time_taken": end_time - start_time
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
```

2. Next, create the `src/evaluators.py` file:

```python
# src/evaluators.py
import os
import json
import time

def create_ground_truth(sample_id, text, dialogues=None, characters=None, emotions=None):
    """
    Create or update ground truth annotations for a text sample
    
    Parameters:
    - sample_id: identifier for the sample
    - text: the text content
    - dialogues: list of dialogue text spans
    - characters: list of character names
    - emotions: list of (text_span, emotion) tuples
    """
    ground_truth = {
        "sample_id": sample_id,
        "text": text,
        "dialogues": dialogues or [],
        "characters": characters or [],
        "emotions": emotions or []
    }
    
    # Save to JSON
    os.makedirs("data/ground_truth", exist_ok=True)
    with open(f"data/ground_truth/{sample_id}.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
    
    return ground_truth

def calculate_metrics(predictions, ground_truth):
    """
    Calculate precision, recall, and F1-score
    
    Parameters:
    - predictions: list of predicted items
    - ground_truth: list of ground truth items
    
    Returns:
    - Dictionary with precision, recall, and F1 metrics
    """
    # Convert to sets for comparison
    pred_set = set(predictions)
    gt_set = set(ground_truth)
    
    # Calculate metrics
    true_positives = len(pred_set.intersection(gt_set))
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def load_samples(directory="data/corpus"):
    """Load sample texts from corpus directory"""
    samples = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                sample_id = os.path.relpath(file_path, directory)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        samples[sample_id] = f.read()
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return samples

def run_complete_evaluation(benchmark_function, libraries=["spacy", "nltk"], azure_client=None):
    """Run the complete evaluation pipeline on all samples"""
    samples = load_samples()
    results = {}
    
    for sample_id, text in samples.items():
        print(f"Processing sample: {sample_id}")
        
        # Load ground truth if available
        ground_truth_path = f"data/ground_truth/{os.path.basename(sample_id)}.json"
        ground_truth = None
        if os.path.exists(ground_truth_path):
            with open(ground_truth_path, "r") as f:
                ground_truth = json.load(f)
        
        # Run benchmarks
        libs = libraries
        if azure_client is None and "azure" in libs:
            libs.remove("azure")
        
        benchmark_results = benchmark_function(text, task="all", libraries=libs, azure_client=azure_client)
        
        # Calculate metrics against ground truth if available
        evaluation_metrics = {}
        if ground_truth:
            for task in ["dialogue_detection", "ner", "sentiment"]:
                evaluation_metrics[task] = {}
                for lib in benchmark_results[task]:
                    # Extract predictions based on task
                    if task == "dialogue_detection":
                        predictions = benchmark_results[task][lib]["detected_dialogue"]
                        evaluation_metrics[task][lib] = calculate_metrics(predictions, ground_truth["dialogues"])
                    elif task == "ner":
                        predictions = [entity[0] for entity in benchmark_results[task][lib]["entities"]]
                        evaluation_metrics[task][lib] = calculate_metrics(predictions, ground_truth["characters"])
                    # Add sentiment evaluation if needed
        
        results[sample_id] = {
            "benchmark_results": benchmark_results,
            "evaluation_metrics": evaluation_metrics
        }
    
    return results

def generate_summary_report(all_results):
    """Generate a summary report of the evaluation results"""
    summary = {
        "dialogue_detection": {lib: {"precision": [], "recall": [], "f1": []} for lib in ["spacy", "nltk", "azure"]},
        "ner": {lib: {"precision": [], "recall": [], "f1": []} for lib in ["spacy", "nltk", "azure"]},
        "sentiment": {lib: {"precision": [], "recall": [], "f1": []} for lib in ["spacy", "nltk", "azure"]},
        "processing_speed": {lib: {"chars_per_second": []} for lib in ["spacy", "nltk", "azure"]}
    }
    
    # Collect metrics across all samples
    for sample_id, result in all_results.items():
        for task in ["dialogue_detection", "ner", "sentiment"]:
            if task in result["evaluation_metrics"]:
                for lib in result["evaluation_metrics"][task]:
                    metrics = result["evaluation_metrics"][task][lib]
                    for metric in ["precision", "recall", "f1"]:
                        if metric in metrics:
                            summary[task][lib][metric].append(metrics[metric])
        
        # Processing speed
        for lib in result["benchmark_results"]["processing_speed"]:
            summary["processing_speed"][lib]["chars_per_second"].append(
                result["benchmark_results"]["processing_speed"][lib]["chars_per_second"]
            )
    
    # Calculate averages
    for task in summary:
        for lib in summary[task]:
            for metric in summary[task][lib]:
                values = summary[task][lib][metric]
                summary[task][lib][metric] = sum(values) / len(values) if values else 0
    
    return summary
```

3. Create the `src/visualize.py` file:

```python
# src/visualize.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_results(results):
    """Create visualizations of benchmark results"""
    # Processing speed comparison
    if "processing_speed" in results:
        speed_data = [(lib, data["chars_per_second"]) for lib, data in results["processing_speed"].items()]
        libs, speeds = zip(*speed_data)
        
        plt.figure(figsize=(10, 6))
        plt.bar(libs, speeds)
        plt.title("Processing Speed Comparison (characters per second)")
        plt.xlabel("Library")
        plt.ylabel("Characters per second")
        plt.show()
    
    # NER comparison
    if "ner" in results:
        ner_data = [(lib, data["count"]) for lib, data in results["ner"].items()]
        if ner_data:
            libs, counts = zip(*ner_data)
            
            plt.figure(figsize=(10, 6))
            plt.bar(libs, counts)
            plt.title("Named Entity Recognition - Detected Entities Count")
            plt.xlabel("Library")
            plt.ylabel("Number of Entities")
            plt.show()
    
    # Dialogue detection comparison
    if "dialogue_detection" in results:
        dialogue_data = [(lib, data["count"]) for lib, data in results["dialogue_detection"].items()]
        if dialogue_data:
            libs, counts = zip(*dialogue_data)
            
            plt.figure(figsize=(10, 6))
            plt.bar(libs, counts)
            plt.title("Dialogue Detection - Detected Dialogue Count")
            plt.xlabel("Library")
            plt.ylabel("Number of Dialogue Segments")
            plt.show()

def visualize_summary(summary):
    """Create visualizations of the summary report"""
    tasks = ["dialogue_detection", "ner", "sentiment"]
    metrics = ["precision", "recall", "f1"]
    
    # Create a bar chart for each task and metric
    for task in tasks:
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            libs = []
            values = []
            
            for lib in summary[task]:
                if summary[task][lib][metric] > 0:  # Only include if there's data
                    libs.append(lib)
                    values.append(summary[task][lib][metric])
            
            if libs:  # Only create chart if we have data
                plt.bar(libs, values)
                plt.title(f"{task.replace('_', ' ').title()} - {metric.title()}")
                plt.xlabel("Library")
                plt.ylabel(metric.title())
                plt.ylim(0, 1)  # Metrics are between 0 and 1
                plt.show()
    
    # Processing speed comparison
    plt.figure(figsize=(10, 6))
    libs = []
    speeds = []
    
    for lib in summary["processing_speed"]:
        if summary["processing_speed"][lib]["chars_per_second"] > 0:
            libs.append(lib)
            speeds.append(summary["processing_speed"][lib]["chars_per_second"])
    
    if libs:  # Only create chart if we have data
        plt.bar(libs, speeds)
        plt.title("Processing Speed Comparison (characters per second)")
        plt.xlabel("Library")
        plt.ylabel("Characters per second")
        plt.show()

def create_comparison_dashboard(all_results):
    """Create a comprehensive dashboard of all benchmark results"""
    # This would be more complex - could use libraries like Plotly or Dash
    # For now, we'll create a basic multi-panel plot
    
    # Prepare data
    libraries = ["spacy", "nltk", "azure"]
    tasks = ["dialogue_detection", "ner", "sentiment", "processing_speed"]
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('NLP Library Comparison for Audiobook Analysis', fontsize=16)
    
    # Flatten for easier indexing
    axs = axs.flatten()
    
    # Plot each task
    for i, task in enumerate(tasks):
        ax = axs[i]
        
        # Collect data for all samples
        lib_data = {lib: [] for lib in libraries}
        
        for sample_id, result in all_results.items():
            if task in result["benchmark_results"]:
                for lib in libraries:
                    if lib in result["benchmark_results"][task]:
                        if task == "processing_speed":
                            lib_data[lib].append(result["benchmark_results"][task][lib]["chars_per_second"])
                        elif task == "dialogue_detection" or task == "ner":
                            lib_data[lib].append(result["benchmark_results"][task][lib]["count"])
                        elif task == "sentiment":
                            # For sentiment, we'll use the absolute compound score
                            lib_data[lib].append(abs(result["benchmark_results"][task][lib]["sentiment_scores"]["compound"]))
        
        # Calculate means for plotting
        means = []
        labels = []
        for lib in libraries:
            if lib_data[lib]:
                means.append(np.mean(lib_data[lib]))
                labels.append(lib)
        
        # Create bar chart
        if means:
            ax.bar(labels, means)
            ax.set_title(f"{task.replace('_', ' ').title()}")
            ax.set_xlabel("Library")
            
            if task == "processing_speed":
                ax.set_ylabel("Characters per second")
            elif task == "dialogue_detection":
                ax.set_ylabel("Dialogue segments detected")
            elif task == "ner":
                ax.set_ylabel("Entities detected")
            elif task == "sentiment":
                ax.set_ylabel("Absolute compound sentiment")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
```

### 6. Create the Jupyter Notebook for Interactive Testing

Create the file `notebooks/nlp_benchmarks.ipynb` in VS Code:

```python
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NLP Engine Benchmarking for Firespeaker Studio\n",
    "\n",
    "This notebook demonstrates the benchmarking of different NLP engines (spaCy, NLTK, Azure) for tasks related to audiobook text analysis:\n",
    "- Dialogue detection\n",
    "- Character identification (NER)\n",
    "- Emotion/sentiment analysis\n",
    "- Processing speed\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Import required modules\n",
    "import sys\n",
    "import os\n",
    "\n",
    "# Add the src directory to the path\n",
    "sys.path.append(os.path.abspath('../src'))\n",
    "\n",
    "import benchmark\n",
    "from evaluators import load_samples, calculate_metrics, run_complete_evaluation, generate_summary_report\n",
    "from visualize import visualize_results, visualize_summary, create_comparison_dashboard\n",
    "\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import json"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setup Azure Client (Optional)\n",
    "\n",
    "If you want to use Azure AI Language, set up your credentials here. Otherwise, skip this section."
   ]
```
