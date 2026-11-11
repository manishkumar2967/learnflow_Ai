# LearnFlow AI — Intelligent Study Assistant

A real RAG study companion for college PDFs. It extracts page text, splits it into chunks, creates local Sentence Transformer embeddings, indexes them in FAISS, retrieves the closest passages, then asks Gemini or Claude to generate a grounded response. A LangGraph workflow routes each request to Q&A, notes, MCQs, or exam preparation.

## Quick start

```bash
cd AI_Study_Assistant
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: add GEMINI_API_KEY or CLAUDE_API_KEY, provider, and matching model
streamlit run app.py
```

Visit the local URL shown by Streamlit, upload text-based PDFs, click **Process documents**, then use the dashboard. Scanned/image-only PDFs require OCR before uploading.

## Architecture

```text
PDF upload → PyPDF page extraction → overlapping chunks
    → all-MiniLM-L6-v2 embeddings → FAISS cosine-similarity index
    → LangGraph: input → intent → retrieve → specialist node → formatted response
    → Gemini / Claude response with document/page citations
```

`graph/workflow.py` provides actual conditional routing:

- Q&A: retrieval → grounded answer
- Notes: retrieval → short/detailed notes and definitions
- MCQs: retrieval → four-option questions with answers/explanations
- Exam: retrieval → 2/5/10-mark question set and revision notes

Tool functions in `tools/study_tools.py` provide `search_uploaded_documents`, `retrieve_relevant_chunks`, `summarize_document`, `generate_mcqs`, and `generate_exam_questions`. The graph calls retrieval directly and invokes the appropriate generation node based on intent.

## How it avoids hallucinations

The LLM is passed only retrieved excerpts plus an instruction to state when material is insufficient. The app appends exact document/page provenance for the chunks used. No answers or course content are hardcoded.

## Demonstration workflow

1. Upload two course PDFs and process them; point out the number of pages and chunks.
2. Ask “Explain deadlock conditions.” Show the returned PDF page sources.
3. Open Notes Studio and generate detailed notes for a unit.
4. Open MCQ Practice, create 10 questions, and use answers/explanations to self-score.
5. Open Exam Prep to show 2-, 5-, and 10-mark question generation.
6. Explain the LangGraph nodes in `graph/workflow.py` and the FAISS retrieval chain above.

## Viva questions and concise answers

1. **What is RAG?** Combining retrieved source material with an LLM prompt to ground output.
2. **Why embeddings?** They represent meaning numerically, enabling semantic—not merely keyword—search.
3. **Why FAISS?** It efficiently finds nearest vectors in a local index.
4. **What model creates embeddings?** `all-MiniLM-L6-v2`, a Sentence Transformer.
5. **Why chunk PDFs?** Whole PDFs exceed context limits and dilute retrieval relevance.
6. **Why overlap chunks?** It preserves ideas that cross a chunk boundary.
7. **What is a vector store?** Vectors plus their source metadata and similarity index.
8. **How are citations produced?** Each chunk retains its original PDF filename and page number.
9. **What is LangGraph?** A stateful directed graph for orchestrating agent workflows.
10. **What causes routing?** Intent classification selects QA, notes, MCQ, or exam node.
11. **What are tools here?** Reusable retrieval, search, summary, MCQ, and exam functions.
12. **How are API keys secured?** Environment variables loaded from `.env`, never frontend source.
13. **What happens on an invalid PDF?** Extraction errors are caught and presented in the UI.
14. **Can it answer outside the PDF?** It is instructed to say the material is insufficient.
15. **How could it scale?** Persist FAISS indexes, add OCR, authentication, and background ingestion.

## Three-minute evaluator script

“LearnFlow AI turns course PDFs into a searchable learning workspace. On upload, PyPDF extracts text page-by-page, then the app creates overlapping chunks. Sentence Transformers converts each chunk into an embedding and FAISS stores them for semantic retrieval. When I ask a question, LangGraph classifies my intent, retrieves the most relevant chunks, routes to a dedicated specialist node, and sends only those excerpts to the selected Gemini or Claude model. Notice the page references: those demonstrate grounding. The same graph routes summaries, MCQs, and exam prep through their own nodes. This is not a static chatbot—the answers depend on the uploaded material, with real embeddings, FAISS similarity search, tool functions, and a conditional LangGraph workflow.”
