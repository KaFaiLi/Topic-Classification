"""
Phase 5: Topic Label Generation using LLMs
"""
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from topic_classifier.config import OPENAI_API_KEY, LLM_MODEL


def generate_labels(exemplars: dict[int, str]) -> dict[int, str]:
    """
    Given medoid texts per cluster, generate concise labels via LLM.
    """
    # initialize LLM
    llm = ChatOpenAI(model_name=LLM_MODEL, openai_api_key=OPENAI_API_KEY)
    # prompt template
    template = (
        """
You are an expert email classifier. Given the following example email text, generate a 3-5 word topic label.

Example:
{text}

Label:"""
    )
    prompt = PromptTemplate(input_variables=["text"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    labels = {}
    for lbl, text in exemplars.items():
        resp = chain.run(text=text)
        labels[lbl] = resp.strip()
    return labels
