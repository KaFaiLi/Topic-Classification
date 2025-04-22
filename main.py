#!/usr/bin/env python3
"""
Entry point for the email topic classification pipeline.
"""
import os
import glob
import json
import argparse
import csv

from topic_classifier.preprocessing import parse_email, clean_text
from topic_classifier.embeddings import embed_texts
from topic_classifier.clustering import cluster_embeddings, evaluate_clusters
from topic_classifier.representation import select_medoid, summarize_cluster
from topic_classifier.labeling import generate_labels
from transformers import pipeline


def main():
    parser = argparse.ArgumentParser(description="Email Topic Classification Tool")
    parser.add_argument("input_dir", help="Directory of .eml files or path to a CSV file with text content")
    parser.add_argument("--csv_col", default="text", help="CSV column name containing the text")
    parser.add_argument("--method", default="hdbscan", choices=["hdbscan", "kmeans"], help="Clustering method")
    parser.add_argument("--k", type=int, help="Number of clusters for kmeans")
    parser.add_argument("--output", default="results.json", help="Output JSON file with cluster labels")
    args = parser.parse_args()

    # Phase 1: Load and preprocess texts
    texts = []
    if args.input_dir.lower().endswith('.csv'):
        with open(args.input_dir, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get(args.csv_col, '')
                texts.append(clean_text(raw))
    else:
        file_paths = glob.glob(os.path.join(args.input_dir, "*"))
        for fp in file_paths:
            raw = parse_email(fp)
            texts.append(clean_text(raw))

    # Phase 2: Embeddings via OpenAI
    embeddings = embed_texts(texts)

    # Phase 3: Clustering
    labels, _ = cluster_embeddings(embeddings, method=args.method, k=args.k)
    metrics = evaluate_clusters(embeddings, labels)
    print("Clustering metrics:", metrics)

    # Phase 4: Representation
    medoids = select_medoid(embeddings, texts, labels)
    summarizer = pipeline("summarization")
    summaries = {lbl: summarize_cluster(medoids[lbl], summarizer) for lbl in medoids}

    # Phase 5: Labeling
    labels_dict = generate_labels(medoids)

    # Assemble results and save
    results = {
        lbl: {
            "label": labels_dict.get(lbl),
            "medoid": medoids[lbl],
            "summary": summaries[lbl]
        }
        for lbl in sorted(medoids)
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()