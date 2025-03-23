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