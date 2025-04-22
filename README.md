# Email Topic Classification Tool

This project provides an end-to-end pipeline for automated topic classification of emails or any text corpus. It leverages advanced NLP techniques, OpenAI embeddings, clustering, and LLM-powered topic labeling.

## Features
- **Robust Preprocessing:** Cleans email text, removes signatures, boilerplate, and quoted/forwarded text.
- **Flexible Input:** Accepts either a directory of `.eml` email files or a CSV file with a text column.
- **Semantic Embeddings:** Uses OpenAI's `text-embedding-ada-002` for high-quality text embeddings.
- **Clustering:** Groups similar emails using HDBSCAN (default) or KMeans.
- **Cluster Representation:** Extracts representative emails and summaries per topic.
- **LLM Topic Labeling:** Generates concise, human-readable topic labels with GPT (via LangChain).

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=sk-...  # For Linux/macOS
   set OPENAI_API_KEY=sk-...     # For Windows
   ```

## Usage
### Classify a directory of `.eml` files:
```bash
python main.py path/to/email_dir
```

### Classify a CSV file (specify column with text):
```bash
python main.py path/to/emails.csv --csv_col body_text
```

### Options
- `--method`: Clustering algorithm (`hdbscan` or `kmeans`). Default: `hdbscan`.
- `--k`: Number of clusters (for KMeans).
- `--output`: Output JSON file. Default: `results.json`.
- `--csv_col`: Column in CSV containing text. Default: `text`.

## Output
A JSON file with cluster labels, representative email (medoid), and summary for each cluster.

## Requirements
- Python 3.8+
- See `requirements.txt` for Python dependencies.

## Notes
- Summarization and topic labeling use LLMs and may incur API costs.
- For best results, ensure emails are in English or use multilingual models as needed.

## License
MIT
