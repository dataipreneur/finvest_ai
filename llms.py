import os
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.llms.together import TogetherLLM

load_dotenv()


TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_together_lm():
    return TogetherLLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        api_key=TOGETHER_API_KEY,
    )


def get_groq_lm():
    return Groq(
        model="llama3-8b-8192",
        api_key=GROQ_API_KEY,
    )
