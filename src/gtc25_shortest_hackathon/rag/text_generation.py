import os

from openai import OpenAI

from dotenv import find_dotenv, load_dotenv

from gtc25_shortest_hackathon.rag import vector_store

_ = load_dotenv(find_dotenv())

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("NGC_API_KEY"),
)

if __name__ == "__main__":
    question = "Que NIM me puede ayudar a generar animaciones faciales en 3D?"
    results = vector_store.vector_store.similarity_search(question, k=2)
    context = "\n\n".join([res.page_content.strip() for res in results])
    
    completion = client.chat.completions.create(
      model="nvidia/llama-3.3-nemotron-super-49b-v1",
      messages=[
          {"role":"system","content":"detailed thinking off. Answer user question in the source language of the question"}, 
          {"role":"user","content": f"question: {question}, context: {context}"},
      ],
      temperature=0.6,
      top_p=0.95,
      max_tokens=4096,
      frequency_penalty=0,
      presence_penalty=0,
    )
    print(completion.choices[0].message.content)

