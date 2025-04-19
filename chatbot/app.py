from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from pathlib import Path
import streamlit as st
import os
from dotenv import load_dotenv
import getpass 
# from llama_cpp import Llama
from langchain.llms import LlamaCpp


# Explicitly load the .env file from the root directory
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

# retrieve tokens
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN") 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user questions"),
    ("user","Question: {question}")])

# streamlit framework

st.title('Langchain + Llama GGUF Chatbot (CPU)')
input_text = st.text_input("Write your query here")

# Huggingface llm
llm = LlamaCpp(model_path=r"C:\Users\91996\LANGCHAIN\chatbot\llama-2-7b-chat.Q4_K_M.gguf",
              n_ctx = 2048,
              n_threads = 4,
              temperature=0.7,
              max_tokens=512)

output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))