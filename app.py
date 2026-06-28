import streamlit as st
import requests

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
        with st.spinner("Indexing..."):
            response = requests.post(
                "http://127.0.0.1:8000/upload",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            )
        if response.status_code == 200:
            st.success("Indexed successfully")

    st.divider()

    st.markdown("### Indexed Documents")
    docs_response = requests.get("http://127.0.0.1:8000/documents")
    if docs_response.status_code == 200:
        docs = docs_response.json()["documents"]
        if docs:
            for doc in docs:
                st.markdown(f'<div class="doc-badge">📄 {doc}</div>', unsafe_allow_html=True)
        else:
            st.caption("No documents yet")

    st.divider()

    st.markdown("### Risk Analysis")
    if st.button("Generate Risk Flags"):
        with st.spinner("Analyzing..."):
            risk_response = requests.get("http://127.0.0.1:8000/risks")
        if risk_response.status_code == 200:
            st.session_state.risks = risk_response.json()["risks"]

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

    with st.spinner("Thinking..."):
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": question}
        )

    if response.status_code == 200:
        answer = response.json()["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(f'<div class="chat-message-assistant">{answer}</div>', unsafe_allow_html=True)
    else:
        st.error("Something went wrong")