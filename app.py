import streamlit as st
from rag.document_loader import load_pdf
from rag.text_splitter import split_pages
from rag.vector_store import StudyVectorStore
from graph.workflow import build_workflow

st.set_page_config(page_title="LearnFlow AI", page_icon="🎓", layout="wide")
st.markdown("""<style>
 .stApp {background: linear-gradient(135deg,#0b1020,#151b35); color:#eaf0ff}
 [data-testid='stSidebar'] {background:#10172d} .hero {padding:1.5rem 0 .5rem}
 .hero h1 {font-size:3rem;margin:0;background:linear-gradient(90deg,#a7b8ff,#57e3c4);-webkit-background-clip:text;color:transparent}
 .card {background:#1a2342;border:1px solid #2c3963;border-radius:16px;padding:1rem;margin:.5rem 0}
 </style>""", unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []
if "store" not in st.session_state: st.session_state.store = None

with st.sidebar:
    st.title("🎓 LearnFlow AI")
    page = st.radio("Workspace", ["Ask your PDFs", "Notes Studio", "MCQ Practice", "Exam Prep"])
    st.divider(); files = st.file_uploader("Upload academic PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process documents", type="primary", disabled=not files):
        try:
            with st.spinner("Extracting pages, chunking text, and building FAISS index…"):
                pages = [p for file in files for p in load_pdf(file)]
                chunks = split_pages(pages)
                st.session_state.store = StudyVectorStore(chunks)
                st.session_state.doc_names = [f.name for f in files]
            st.success(f"Ready: {len(files)} PDFs, {len(pages)} pages, {len(chunks)} chunks.")
        except Exception as exc: st.error(str(exc))
    if st.session_state.store:
        st.caption("Indexed: " + ", ".join(st.session_state.doc_names))

st.markdown("<div class='hero'><h1>Intelligent learning, grounded in your material.</h1><p>RAG-powered notes, answers and practice—each tied to your uploaded PDFs.</p></div>", unsafe_allow_html=True)
if not st.session_state.store:
    st.info("Upload one or more text-based PDFs and click **Process documents** to begin.")
    st.stop()

def run(query, **kwargs):
    try:
        with st.spinner("LangGraph is retrieving relevant passages and generating a grounded response…"):
            result = build_workflow().invoke({"query": query, "store": st.session_state.store, **kwargs})
        st.session_state.history.append((query, result["answer"]))
        st.markdown(result["answer"])
    except Exception as exc:
        st.error("Generation failed. Check your .env API key and model configuration. Details: " + str(exc))

if page == "Ask your PDFs":
    for q, a in st.session_state.history[-5:]:
        with st.chat_message("user"): st.write(q)
        with st.chat_message("assistant"): st.markdown(a)
    question = st.chat_input("Ask anything from the uploaded material…")
    if question:
        with st.chat_message("user"): st.write(question)
        with st.chat_message("assistant"): run(question)
elif page == "Notes Studio":
    st.header("Chapter Notes")
    topic = st.text_input("Topic or chapter", placeholder="e.g., Operating system scheduling")
    detail = st.selectbox("Depth", ["short", "detailed"])
    if st.button("Generate notes", type="primary") and topic: run(f"Create notes for {topic}", detail=detail)
elif page == "MCQ Practice":
    st.header("MCQ Practice")
    topic = st.text_input("Topic", placeholder="e.g., Normalization")
    count = st.selectbox("Questions", [5, 10, 20])
    if st.button("Create MCQs", type="primary") and topic: run(f"Generate MCQs on {topic}", count=count)
elif page == "Exam Prep":
    st.header("Exam Preparation")
    topic = st.text_input("Topic or unit", placeholder="e.g., Unit 3: Deadlocks")
    if st.button("Build exam pack", type="primary") and topic: run(f"Prepare exam questions for {topic}")
