"""
Phase 2: Semantic Embedding Generation
"""
import openai
from topic_classifier.config import OPENAI_EMBEDDING_MODEL, OPENAI_API_KEY

# configure OpenAI API key
openai.api_key = OPENAI_API_KEY

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Return embeddings for a list of texts via OpenAI API.
    """
    response = openai.Embeddings.create(
        input=texts,
        model=OPENAI_EMBEDDING_MODEL
    )
    return [item.embedding for item in response.data]
