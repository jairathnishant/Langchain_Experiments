from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
from pathlib import Path
# from llama_cpp import Llama
from langchain.llms import LlamaCpp
from pydantic import BaseModel
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

from pydantic import BaseModel

class TopicRequest(BaseModel):
    topic: str


load_dotenv()

# Explicitly load the .env file from the root directory
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

# retrieve tokens
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN") 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title = "Langchain server",
    version = "1.0",
    description = "API Server for chatbot"
)


llm_1 = LlamaCpp(model_path=r"C:\Users\91996\LANGCHAIN\chatbot\llama-2-7b-chat.Q4_K_M.gguf",
              n_ctx = 2048,
              n_threads = 4,
              temperature=0.7,
              max_tokens=512)

llm_2 = Ollama(model="tinyllama")

prompt = ChatPromptTemplate.from_template("Write an essay about {prompt} in 100 words")
# prompt_2 = ChatPromptTemplate.from_template("Write an poem about {topic}")

# chain = prompt_1 | llm_2


class PromptInput(BaseModel):
    prompt: str

@app.post("/tinyllama")
def tinyllama_chat(input: PromptInput):
    llm = llm_2
    return {"output": (prompt | llm).invoke({"prompt": input.prompt})}

@app.post("/llama2")
def llama2_chat(input: PromptInput):
    llm = llm_1
    return {"output": (prompt | llm).invoke({"prompt": input.prompt})}

# @app.post("/essay")
# def generate_essay(request: TopicRequest):
#     return {"output": chain.invoke({"topic": request.topic})}

# add_routes(
#     app,
#     prompt_1|llm_1,
#     path = "/essay",
#     # rebuild_model=True,
#     # include_schema=False
#     # force_rebuild = True
# )

# add_routes(
#     app,
#     prompt_2|llm_2,
#     path = "/poem",
#     # rebuild_model=True
#     # include_schema=False
#     # force_rebuild = True
# )

if __name__ == "__main__":
    
    # PromptInput.model_rebuild()

    uvicorn.run(app, host = "localhost", port = 8000)