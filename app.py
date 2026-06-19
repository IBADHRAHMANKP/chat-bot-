import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain.chains.retrieval import create_retrieval_chain
#from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Load environment ──────────────────────────────────────────────
load_dotenv()

# ── Page configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="NexusAI — Intelligent Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium look ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ─────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}

/* ── Sidebar ────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0 !important;
}

/* ── Header ─────────────────────────────────────── */
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
    animation: fadeSlideIn 0.8s ease-out;
}

.hero-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 300;
    margin-top: 4px;
    margin-bottom: 28px;
    animation: fadeSlideIn 1s ease-out;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Chat bubbles ───────────────────────────────── */
.user-bubble {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: #fff;
    padding: 14px 20px;
    border-radius: 20px 20px 4px 20px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
    line-height: 1.55;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25);
    animation: bubblePop 0.35s ease-out;
}

.assistant-bubble {
    background: rgba(30, 41, 59, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99, 102, 241, 0.12);
    color: #e2e8f0;
    padding: 14px 20px;
    border-radius: 20px 20px 20px 4px;
    margin: 8px 0;
    max-width: 80%;
    font-size: 0.95rem;
    line-height: 1.55;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    animation: bubblePop 0.35s ease-out;
}

@keyframes bubblePop {
    from { opacity: 0; transform: scale(0.92) translateY(8px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}

/* ── Status cards ───────────────────────────────── */
.status-card {
    background: rgba(30, 41, 59, 0.65);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 12px 0;
    color: #cbd5e1;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.status-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 0 24px rgba(99, 102, 241, 0.08);
}

.status-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #818cf8;
    font-weight: 600;
    margin-bottom: 4px;
}

.status-card .value {
    font-size: 1.35rem;
    font-weight: 600;
    color: #f1f5f9;
}

/* ── File uploader ──────────────────────────────── */
section[data-testid="stSidebar"] .stFileUploader > div {
    border: 2px dashed rgba(99, 102, 241, 0.3) !important;
    border-radius: 14px !important;
    background: rgba(99, 102, 241, 0.04) !important;
    transition: border-color 0.3s;
}

section[data-testid="stSidebar"] .stFileUploader > div:hover {
    border-color: rgba(99, 102, 241, 0.55) !important;
}

/* ── Buttons ────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.35) !important;
}

/* ── Chat input ─────────────────────────────────── */
.stChatInput > div {
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 16px !important;
    background: rgba(15, 23, 42, 0.7) !important;
    backdrop-filter: blur(8px) !important;
}

/* ── Divider glow ───────────────────────────────── */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #6366f1, transparent);
    margin: 20px 0;
    border: none;
}

/* ── Scrollbar ──────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load Sentence Transformer embeddings (cached)."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def process_pdfs(uploaded_files):
    """Load PDFs → split → embed → return Chroma vectorstore."""
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        documents.extend(loader.load())
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="pdf_collection",
    )
    return vectorstore, len(chunks)


def get_llm(api_key: str, model_name: str, temperature: float):
    """Return a Gemini LLM via LangChain."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=True,
    )


def build_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

chain = create_retrieval_chain(
    retriever,
    llm,
)

return chain


# ── Session state init ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 NexusAI")
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # API key
    api_key = os.getenv("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "🔑 Gemini API Key",
        value=api_key,
        type="password",
        help="Get yours at https://aistudio.google.com/app/apikey",
    )

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Model settings
    st.markdown("### ⚙️ Model Settings")
    model_name = st.selectbox(
        "Model",
        ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # PDF upload
    st.markdown("### 📄 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDFs to chat with",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF documents to build a knowledge base.",
    )

    if uploaded_files:
        file_names = sorted([f.name for f in uploaded_files])
        if file_names != st.session_state.pdf_names:
            if not api_key_input:
                st.error("⚠️ Please enter your Gemini API key first.")
            else:
                with st.spinner("🔮 Processing documents…"):
                    vectorstore, chunk_count = process_pdfs(uploaded_files)
                    llm = get_llm(api_key_input, model_name, temperature)
                    chain = build_chain(llm, vectorstore)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.chain = chain
                    st.session_state.chunk_count = chunk_count
                    st.session_state.pdf_names = file_names
                    st.session_state.messages = []

                st.success(f"✅ {len(uploaded_files)} PDF(s) processed!")

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Stats
    if st.session_state.vectorstore:
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""<div class="status-card">
                    <div class="label">Documents</div>
                    <div class="value">{len(st.session_state.pdf_names)}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""<div class="status-card">
                    <div class="label">Chunks</div>
                    <div class="value">{st.session_state.chunk_count}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # Clear chat
    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────
st.markdown('<p class="hero-title">NexusAI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Your intelligent PDF assistant — powered by Gemini & LangChain</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

# Onboarding card when no PDFs loaded
if not st.session_state.vectorstore:
    st.markdown(
        """
        <div class="status-card" style="text-align:center; padding:40px 24px;">
            <div style="font-size:3rem; margin-bottom:12px;">📚</div>
            <div style="font-size:1.2rem; font-weight:600; color:#e2e8f0; margin-bottom:8px;">
                Upload PDFs to get started
            </div>
            <div style="font-size:0.9rem; color:#94a3b8;">
                Add your documents in the sidebar, enter your Gemini API key, and start asking questions.
                NexusAI uses RAG (Retrieval-Augmented Generation) to give you accurate, context-aware answers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render chat history
for msg in st.session_state.messages:
    css_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    icon = "👤" if msg["role"] == "user" else "🧠"
    st.markdown(
        f'<div class="{css_class}">{icon}&nbsp;&nbsp;{msg["content"]}</div>',
        unsafe_allow_html=True,
    )

# Chat input
if prompt := st.chat_input("Ask anything about your documents…", key="chat_input"):
    if not api_key_input:
        st.error("⚠️ Please enter your Gemini API key in the sidebar.")
    elif not st.session_state.chain:
        st.warning("📄 Please upload at least one PDF first.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(
            f'<div class="user-bubble">👤&nbsp;&nbsp;{prompt}</div>',
            unsafe_allow_html=True,
        )

        # Get answer
        with st.spinner("🔮 Thinking…"):
            try:
                result = st.session_state.chain.invoke({"input": prompt})
                answer = result.get("answer", result.get("output", ""))

                # Build source info
                sources = result.get("source_documents", [])
                if sources:
                    source_pages = set()
                    for doc in sources:
                        page = doc.metadata.get("page", "?")
                        source_name = doc.metadata.get("source", "Unknown")
                        source_pages.add(f"p.{int(page)+1}")
                    answer += f"\n\n📑 *Sources: {', '.join(sorted(source_pages))}*"

            except Exception as e:
                answer = f"❌ An error occurred: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(
            f'<div class="assistant-bubble">🧠&nbsp;&nbsp;{answer}</div>',
            unsafe_allow_html=True,
        )
        st.rerun()
