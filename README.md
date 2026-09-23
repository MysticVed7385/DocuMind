# 📄 DocuMind — Automation of PDF Interaction

An AI-powered multi-document research assistant that lets you ask natural-language questions across an entire library of PDFs.

🔗 **Live Demo:** [documind-pdf.streamlit.app](https://documind-pdf.streamlit.app)

---

## ✨ Features

- 📚 **Multi-document support** — Upload and query multiple PDFs at once
- 🎯 **Source citations** — Every answer includes filename + page number
- 🧠 **Conversation memory** — Ask follow-up questions naturally
- 🚫 **Hallucination prevention** — Strict "answer-only-from-context" prompt
- ⚡ **Fast inference** — Powered by Groq (GPT-OSS 120B)
- 💰 **Zero API costs** — Local HuggingFace embeddings bypass rate limits
- 🎨 **Modern UI** — Gradient dark theme with chat interface

---

## 🛠️ Tech Stack

| Layer | Technology |
|:---|:---|
| **Frontend** | Streamlit |
| **Orchestration** | LangChain |
| **LLM** | Groq (GPT-OSS 120B) |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) |
| **Vector Store** | FAISS |
| **PDF Parsing** | PyPDFLoader |
| **Language** | Python 3.11 |

---

## 🏗️ Architecture
User Uploads PDFs
↓
PyPDFLoader extracts text
↓
RecursiveCharacterTextSplitter chunks text (1000 chars, 200 overlap)
↓
HuggingFace embeddings convert chunks to vectors
↓
FAISS stores vectors for semantic search
↓
User asks question
↓
Retriever fetches top-6 relevant chunks
↓
Groq LLM generates answer (grounded in context only)
↓
Answer + filename + page citations shown in UI

---

## 🚀 Getting Started Locally

### 1. Clone the repository
```bash
git clone https://github.com/MysticVed7385/DocuMind.git
cd DocuMind

### 2. Create a Virtual Environment
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Set up your API Key
Create a .env file in the project root: 
GROQ_API_KEY=your_groq_api_key_here
Get your free Groq API key from console.groq.com/keys.

### 5. Run the app 
streamlit run app.py

Open your browser to http://localhost:8501

📖 How to Use
1. Upload one or more text-based PDFs using the sidebar

2. Wait for the "✅ N PDF(s) ready!" confirmation

3. Ask any question in natural language

4. View the answer along with source filenames and page numbers

Example questions: 
1. "What is the main topic of these documents?"

2. "Summarize the key findings across all PDFs."

3. "What does the paper say about attention mechanisms?"


📁 Project Structure:
DocuMind/
├── .streamlit/
│   └── config.toml          # Theme + server config
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                      # API key (not committed)
├── .gitignore                # Git ignore rules
└── README.md                 # This file

🎯 What I Learned Building This
--> Implementing Retrieval-Augmented Generation(RAG) from scratch

--> Document chunking strategies for optimal context retrieval

--> Prompt engineering to prevent LLM hallucinations

--> Working with LangChain v1.0's modular architecture

--> Deploying ML apps to Streamlit Community Cloud


🎯 Key Technical Decisions
Why local embeddings? 
--> Using HuggingFace embeddings (all-MiniLM-L6-v2) instead of API-based embeddings eliminates rate limits and reduces cost to zero.

Why Groq? 
--> Groq's inference speed (300+ tokens/sec) makes the chat interface feel instant.

Why FAISS?
--> FAISS is a lightweight, in-memory vector store that requires no server setup — perfect for a deployed demo.

Why a strict prompt? 
--> The "answer-only-from-context" rule prevents the LLM from hallucinating answers when the PDF doesn't contain the information.


👤 Author
Vedant
GitHub: @MysticVed7385