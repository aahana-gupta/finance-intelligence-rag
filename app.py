import streamlit as st
import tempfile
import time
import os
from embed import build_index
from rag import generate_answer, get_available_documents, generate_risk_flags

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

def check_rate_limit():
    now = time.time()
    if "request_timestamps" not in st.session_state:
        st.session_state.request_timestamps = []
    st.session_state.request_timestamps = [t for t in st.session_state.request_timestamps if now - t < 60]
    if len(st.session_state.request_timestamps) >= 10:
        return False
    st.session_state.request_timestamps.append(now)
    return True

st.set_page_config(
    page_title="Finance Intelligence Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');

.stApp { background-color: #1c1c1e; color: #f0f0f0; font-family: 'Inter', sans-serif; }
.stSidebar { background-color: #242426; }
.stButton > button { background-color: #2c2c2e; color: #f0f0f0; border: 1px solid #444; width: 100%; }
.stButton > button:hover { background-color: #3a3a3c; border-color: #666; }
.chat-message-user { background-color: #2c2c2e; padding: 12px 16px; border-radius: 8px; margin: 8px 0; }
.chat-message-assistant { background-color: #1e2d1e; padding: 12px 16px; border-radius: 8px; margin: 8px 0; border-left: 3px solid #4ade80; }
.risk-flag { background-color: #2d1f1f; padding: 10px 14px; border-radius: 6px; margin: 6px 0; border-left: 3px solid #f87171; }
.risk-flag b { color: #f87171; font-family: 'Inter', sans-serif; font-weight: 600; }
.doc-badge { background-color: #2c2c2e; padding: 4px 10px; border-radius: 12px; font-size: 13px; display: inline-block; margin: 3px 0; }
div[data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }
h1, h2, h3 { font-family: 'Space Mono', monospace; font-weight: 700; color: #ffffff; letter-spacing: 1px; text-transform: uppercase; }
.stSuccess { background-color: #1e2d1e; }
section[data-testid="stFileUploadDropzone"] { background-color: #2c2c2e !important; border: 1px solid #444 !important; }
</style>
""")

# Sidebar
with st.sidebar:
    st.markdown("## Finance Intelligence")
    st.caption("AI-powered earnings analysis")
    st.divider()

    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if not check_rate_limit():
            st.error("Rate limit exceeded, please wait.")
        elif not uploaded_file.name.lower().endswith(".pdf"):
            st.error("Only PDF files are allowed")
        elif uploaded_file.size > MAX_UPLOAD_SIZE:
            st.error("File exceeds 10MB size limit")
        else:
            with st.spinner("Indexing..."):
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    build_index(tmp_path)
                    st.success("Indexed successfully")
                except ValueError as e:
                    st.error(str(e))
                finally:
                    if tmp_path:
                        os.remove(tmp_path)

    st.divider()

    st.markdown("### Indexed Documents")
    docs = get_available_documents()
    if docs:
        for doc in docs:
            st.markdown(f'<div class="doc-badge">📄 {doc}</div>', unsafe_allow_html=True)
    else:
        st.caption("No documents yet")

    st.divider()

    st.markdown("### Risk Analysis")
    if st.button("Generate Risk Flags"):
        with st.spinner("Analyzing..."):
            st.session_state.risks = generate_risk_flags()

# Main area
st.markdown("## Finance Intelligence Assistant")
st.caption("Ask questions across all uploaded earnings call transcripts")
st.divider()

if "risks" in st.session_state:
    st.markdown("### Risk Flags")
    for doc, flags in st.session_state.risks.items():
        st.markdown(f"**{doc}**")
        cols = st.columns(2)
        for i, f in enumerate(flags):
            with cols[i % 2]:
                st.markdown(
                    f'<div class="risk-flag">🚩 <b>{f["flag"]}</b><br><span style="font-size:12px;color:#ccc">{f["detail"]}</span></div>',
                    unsafe_allow_html=True
                )
    st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message-assistant">{msg["content"]}</div>', unsafe_allow_html=True)

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(f'<div class="chat-message-user">{question}</div>', unsafe_allow_html=True)

    if not check_rate_limit():
        st.error("Rate limit exceeded, please wait.")
    else:
        with st.spinner("Thinking..."):
            answer = generate_answer(question)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(f'<div class="chat-message-assistant">{answer}</div>', unsafe_allow_html=True)