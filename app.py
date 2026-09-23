import streamlit as st
import os
from dotenv import load_dotenv

# Groq for the CHAT model
from langchain_groq import ChatGroq

# Local HuggingFace embeddings (no API limits)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Document loading
from langchain_community.document_loaders import PyPDFLoader

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store
from langchain_community.vectorstores import FAISS

# Chains and memory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# Prompts
from langchain_core.prompts import PromptTemplate

# ---------- 1. SETUP ----------
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY in .env file.")
    st.stop()

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")

# ---------- 2. CUSTOM CSS ----------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(180deg, #0e1117 0%, #1a1f2e 100%); }
    h1 { 
        background: linear-gradient(90deg, #4f9cf9, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .stChatMessage { border-radius: 12px; padding: 10px; }
    div[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f9cf9, #a855f7);
        color: white; border: none; border-radius: 8px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 DocuMind — Automation of PDF Interaction")
st.caption("Ask questions about multiple PDFs using natural language — powered by LangChain + Groq (GPT-OSS 120B).")

# ---------- 3. SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ---------- 4. SIDEBAR ----------
with st.sidebar:
    st.header("📎 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    # Create a unique key from all uploaded file names
    current_files_key = "|".join(sorted([f.name for f in uploaded_files])) if uploaded_files else None

    if uploaded_files and current_files_key != st.session_state.pdf_name:
        with st.spinner(f"Processing {len(uploaded_files)} PDF(s)… chunking, embedding, indexing"):
            # Process EACH uploaded PDF
            all_docs = []
            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                loader = PyPDFLoader(temp_path)
                docs = loader.load()

                # Add source metadata for citation transparency
                for doc in docs:
                    doc.metadata["source_file"] = uploaded_file.name

                all_docs.extend(docs)

                # Safe delete of the temp file after processing
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            # Chunk all documents together
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = splitter.split_documents(all_docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = FAISS.from_documents(chunks, embeddings)

            # Prompt optimization (prevents hallucination)
            custom_prompt = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template="""You are a precise PDF question-answering assistant.

Use ONLY the context below to answer the question. If the answer is not in the 
context, respond with: "I could not find that information in the documents."

When writing mathematical formulas, use LaTeX with $...$ for inline math 
and $$...$$ for block equations.

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer (be concise and cite the source file and page if possible):"""
            )

            llm = ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=0.2,
                max_retries=3,
            )

            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )

            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": custom_prompt},
                return_source_documents=True,
            )
            st.session_state.pdf_name = current_files_key
            st.session_state.messages = []
            st.success(f"✅ {len(uploaded_files)} PDF(s) ready!")

    if st.session_state.pdf_name:
        num_docs = len(st.session_state.pdf_name.split("|"))
        st.info(f"Active: **{num_docs} document(s)**")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# ---------- 5. CHAT INTERFACE ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Ask something about the PDFs…")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    if st.session_state.chain is None:
        with st.chat_message("assistant"):
            st.warning("Please upload PDF(s) first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = st.session_state.chain.invoke({"question": user_q})
                answer = result["answer"]
                st.markdown(answer)

                # Show source pages with filenames
                sources = result.get("source_documents", [])
                if sources:
                    with st.expander(f"📑 Sources — {len(sources)} chunks"):
                        for d in sources:
                            source_name = d.metadata.get("source_file", "Unknown")
                            page_num = d.metadata.get("page", 0) + 1
                            st.markdown(f"**{source_name}** — Page {page_num}:")
                            st.markdown(f"{d.page_content[:300]}…")
                            st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})