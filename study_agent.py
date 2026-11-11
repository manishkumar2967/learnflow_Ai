"""LLM factory and grounded generation helpers."""
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
load_dotenv()

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=os.getenv("CLAUDE_API_KEY"), temperature=0.2)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)

def generate(instruction: str, context: str) -> str:
    prompt = ("You are a precise, encouraging study assistant. Use only the supplied course material. "
              "If it is insufficient, say so plainly. Write clear exam-ready English.\n\n"
              f"COURSE MATERIAL:\n{context}\n\nTASK:\n{instruction}")
    response = get_llm().invoke([SystemMessage(content="Ground every claim in the provided material."), HumanMessage(content=prompt)])
    return response.content
