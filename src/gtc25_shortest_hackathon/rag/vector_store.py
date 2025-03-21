import os
from uuid import uuid4

from langchain_chroma import Chroma

from gtc25_shortest_hackathon.rag import embeddings

persist_directory = "./chroma_langchain_db"
vector_store = Chroma(
    collection_name="docs",
    embedding_function=embeddings.client,
    persist_directory=persist_directory,
)

if not os.path.exists(persist_directory):
    from gtc25_shortest_hackathon.rag import ingestion
    
    uuids = [str(uuid4()) for _ in range(len(ingestion.docs))]
    vector_store.add_documents(documents=ingestion.docs, ids=uuids)

if __name__ == "__main__":
    results = vector_store.similarity_search(
        "What NIM can help me generate a 3d facial animation?",
        k=2,
    )
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")