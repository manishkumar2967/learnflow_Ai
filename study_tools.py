from rag.vector_store import StudyVectorStore
from agents.study_agent import generate

def retrieve_relevant_chunks(store: StudyVectorStore, query: str, k: int = 5):
    return store.search(query, k)

def search_uploaded_documents(store: StudyVectorStore, query: str) -> str:
    return "\n\n".join(f"[{c.source}, p.{c.page}]\n{c.text}" for c, _ in store.search(query))

def _context(store, topic):
    return search_uploaded_documents(store, topic)

def summarize_document(store, topic: str, detail: str = "short") -> str:
    return generate(f"Create {detail} chapter notes for: {topic}. Include headings, key points, definitions and a short exam answer.", _context(store, topic))

def generate_mcqs(store, topic: str, count: int = 5) -> str:
    return generate(f"Create exactly {count} MCQs on {topic}. Each needs four plausible options A-D, then an Answer and one-sentence Explanation. Base all questions solely on material.", _context(store, topic))

def generate_exam_questions(store, topic: str) -> str:
    return generate(f"For {topic}, generate 3 two-mark, 3 five-mark and 2 ten-mark important questions. Then give concise quick-revision notes.", _context(store, topic))
