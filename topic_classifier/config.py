# Configuration for topic classification pipeline

# OpenAI embedding model
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# HDBSCAN clustering params
def get_hdbscan_params():
    return {
        "min_cluster_size": 5,
        "metric": "euclidean",
        "cluster_selection_method": "eom"
    }

# LangChain / OpenAI
OPENAI_API_KEY = ""  # Set via env var or secrets manager
LLM_MODEL = "gpt-3.5-turbo"
