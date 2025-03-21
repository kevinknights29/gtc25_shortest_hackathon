import os

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

client = NVIDIAEmbeddings(
  model="nvidia/llama-3.2-nv-embedqa-1b-v2", 
  api_key=os.getenv("NGC_API_KEY"), 
  truncate="NONE", 
)
