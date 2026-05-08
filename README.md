Multi-Source Stateful RAG Agent 🤖

A production-ready Retrieval-Augmented Generation (RAG) system that transforms static portfolio documents into an interactive, stateful AI assistant. The system follows a “search-before-answer” architecture using LangGraph for decision-making and FAISS for fast semantic retrieval.

🚀 Key Features
Stateful Orchestration
Implements LangGraph to manage conversational state and enforce structured decision flows, ensuring retrieval occurs before response generation.
Torch-Free Architecture (Windows Optimized)
Designed to run reliably on Windows by eliminating PyTorch dependencies. Uses ONNX-based embeddings via FastEmbed and optional Google Generative AI integration.
Semantic Memory with FAISS
Indexes unstructured .txt data using FAISS for high-speed similarity search with sub-second latency.
Smart Recursive Chunking
Custom text-splitting strategy (~500-character chunks with overlap) to preserve semantic context and improve retrieval quality.
High-Performance Inference
Integrated with Groq (LLaMA 3.1 / 3.3) for low-latency, real-time responses.
🛠️ Technical Stack
Frameworks: LangChain, LangGraph
LLM: Groq (LLaMA-3.1-8B, LLaMA-3.3-70B)
Embeddings: FastEmbed (ONNX, local) / Google Generative AI (optional)
Vector Database: FAISS
Frontend/UI: Streamlit
Environment: Python 3.12, uv package manager
🧠 System Architecture
Ingestion
Loads and parses text files from a portfolio/ directory.
Indexing
Applies custom chunking
Converts text into vector embeddings
Stores vectors in FAISS index
Agent Workflow
User Input: Captured via Streamlit UI
Decision Layer: Determines if retrieval is required
Tool Invocation: Calls search_portfolio to query FAISS
Response Generation: LLM synthesizes retrieved context with the query
🔧 Installation & Setup
git clone https://github.com/<your-username>/Multi-Source-Stateful-RAG-Agent
cd Multi-Source-Stateful-RAG-Agent
uv venv --python 3.12
.venv\Scripts\activate
uv pip install streamlit langchain-community langchain-groq fastembed faiss-cpu python-dotenv langgraph

Create a .env file:

GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key  # Optional

Run the app:

streamlit run app.py
🛡️ Challenges & Solutions
Windows DLL Errors (WinError 1114 / 126)
Resolved by removing PyTorch dependency and implementing ONNX-based embeddings with module-level patching.
Infinite Agent Loops
Prevented using recursion limits and stricter system prompts to control tool-calling behavior.
Context & Token Optimization
Designed sliding-window chunking to balance context richness with LLM token limits.
📌 Impact
Enabled interactive querying of static portfolio data
Achieved low-latency responses with high factual accuracy
Built a production-style RAG pipeline suitable for real-world deployment