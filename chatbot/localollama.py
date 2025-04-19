from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from pathlib import Path

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Explicitly load the .env file from the root directory
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

# retrieve tokensos.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user questions"),
    ("user","Question: {question}")])

# streamlit framework

st.title('Langchain + Llama GGUF Chatbot (CPU)')
input_text = st.text_input("Write your query here")

llm = Ollama(model="qwen")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))