import os
import sys
from dotenv import load_dotenv

# Prevent heavy lib crashes
sys.modules["torch"] = None
sys.modules["sentence_transformers"] = None

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# -------------------------------
# 📂 LOAD DOCUMENTS
# -------------------------------
def load_portfolio():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(base_dir, "portfolio")

    docs = []
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        return []

    for root, _, files in os.walk(docs_dir):
        category = os.path.basename(root)

        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    docs.append(Document(
                        page_content=f.read(),
                        metadata={"source": file, "category": category}
                    ))
    return docs

chunk_size=300
def chunk_docs(docs):
    chunks = []
    for doc in docs:
        text = doc.page_content
        for i in range(0, len(text),chunk_size-100):
            chunks.append(Document(
                page_content=text[i:i+450],
                metadata=doc.metadata
            ))
    return chunks


# -------------------------------
# 🧠 VECTOR DB
# -------------------------------

from langchain_community.embeddings import FastEmbedEmbeddings

# This is the ONLY local model that works without Torch/DLL errors
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")



if not os.path.exists("faiss_index"):
    print("🚀 No index found. Building vector store from portfolio...") # Add this
    raw_docs = load_portfolio()
    if raw_docs:
        chunks = chunk_docs(raw_docs)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local("faiss_index")
        print(f"✅ Index created with {len(chunks)} chunks.") # Add this
    else:
        print("❌ Error: No documents found in portfolio folder!")
        vector_store = None
else:
    print("📂 Loading existing FAISS index...") # Add this
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )



# -------------------------------
# 🤖 LLM (CHEAP + FAST)
# -------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=150
)
# -------------------------------
# 🧠 CORE RAG LOGIC (SIMPLE + STRONG)
# -------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage
from typing import Annotated    
from typing_extensions import TypedDict
# 1. Define the "Notebook" (State)
class AgentState(TypedDict):
    messages: Annotated[list,add_messages]
    context: str
    answer: str
    steps: List[str]
    count: int

# 2. Node: The Retriever
# 2. Node: The Retriever
def retrieve(state: AgentState):
    # ✅ Fixed: Correctly pulling from messages list
    query = state["messages"][-1].content
    
    docs = vector_store.similarity_search(query, k=6)
    context = "\n\n".join([d.page_content for d in docs])
    
    return {"context": context, "count": state.get("count", 0) + 1}






# 3. Node: The Grader (The "Logic Check")
# 3. Node: The Grader (The "Logic Check")
def grade_results(state: AgentState):
    context = state["context"].lower()
    
    # ❌ FIX: Instead of state["question"], we look at the last message
    latest_query = state["messages"][-1].content.lower()
    current_count = state.get("count", 0)

    # Check if we found the venue keywords
    has_venue = any(k in context for k in ["springer", "icdam", "conference", "journal", "2025"])

    # ❌ FIX: Use latest_query here instead of 'question'
    if ("published" in latest_query or "paper" in latest_query) and not has_venue:
        if current_count < 3: 
            print(f"❌ Venue missing (Attempt {current_count}). Retrying search...")
            return "rewrite_and_search"

    print("✅ Moving to generation.")
    return "generate"




# 4. Node: The Answer Generator
def generate(state: AgentState):
    # 1. Get the current size of the vector store for the prompt
    kb_size = vector_store.index.ntotal 
    
    # 2. Extract the latest question from the message list
    current_question = state["messages"][-1].content
    
    # 3. Create the System Instructions
    sys_msg = SystemMessage(content=(
        "You are Sarthak Sharma's professional Career Assistant.\n"
        f"FACT: Your knowledge base contains {kb_size} chunks of Sarthak's resume and projects.\n"
        "INSTRUCTIONS:\n"
        "- Answer based ONLY on the provided context and chat history.\n"
        "- If the answer isn't in the context, politely say you don't know.\n"
        "- Always include specific details like conference names (e.g., ICDAM) or metrics."
    ))
    
    # 4. Create the Context Message (The "RAG" part)
    # We pass this as a separate message to keep the history clean
    context_msg = SystemMessage(content=f"SUPPORTING CONTEXT FROM PORTFOLIO:\n{state['context']}")
    
    # 5. Build the full payload for the LLM
    # We combine: Instructions + Context + The Entire Conversation History
    full_history = [sys_msg, context_msg] + state["messages"]
    
    # 6. Call the LLM (Groq)
    response = llm.invoke(full_history)
    
    # 7. Return the update to the state
    return {
        "answer": response.content,
        # 'add_messages' in your AgentState will append this to the history automatically
        "messages": [AIMessage(content=response.content)],
        "steps": state.get("steps", []) + ["generate_answer"]
    }



from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

# 5. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")

# Add the conditional logic: If grade is bad, we could add a rewrite step.
# For now, let's connect the flow:
workflow.add_conditional_edges(
    "retrieve",
    grade_results,
    {
        "generate": "generate",
        "rewrite_and_search": "retrieve" # It will try to re-retrieve
    }
)

workflow.add_edge("generate", END)

# Compile

agent_app = workflow.compile(checkpointer = memory)

