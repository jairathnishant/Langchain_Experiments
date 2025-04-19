from fastapi import FastAPI
from langchain.prompt import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from uvicorn
import os
from langchain_community.llms import Ollama

