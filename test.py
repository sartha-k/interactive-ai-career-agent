from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings # Import the translator
import os

# 1. Define the same embeddings you used in app.py
# If you used Google, use GoogleGenerativeAIEmbeddings here instead
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 2. Load the index
if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # 3. Get all document IDs
    # In newer FAISS, we access the docstore through the .docstore._dict
    all_ids = list(vector_store.docstore._dict.keys())

    print(f"📊 Total Chunks in DB: {len(all_ids)}\n")

    # 4. Print the first 3 chunks to "peek" inside
    for i in range(min(3, len(all_ids))):
        doc_id = all_ids[i]
        document = vector_store.docstore.search(doc_id)
        
        print(f"--- 📄 CHUNK {i+1} ---")
        print(f"SOURCE: {document.metadata.get('source', 'Unknown')}")
        print(f"CONTENT: {document.page_content}") 
        print("-" * 30 + "\n")
else:
    print("❌ Error: 'faiss_index' folder not found. Run app.py first!")
