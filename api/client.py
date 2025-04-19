import requests
import streamlit as st

# Dropdown for model selection
model_choice = st.selectbox(
    "Choose a model",
    ["tinyllama", "llama2"],
    index=0
)

# def get_llama_response(input_text):
#     response = requests.post("http://localhost:8000/essay/invoke",
#     json = {'input':{'topic' :input_text}}
#     )
#     return response.json()['output']
# def get_ollama_response(input_text):
#     try:
#         response = requests.post("http://localhost:8000/poem/invoke",
#             json={'input': {'topic': input_text}}
#         )
        
#         # Print response details for debugging
#         st.write(f"Status Code: {response.status_code}")
#         st.write(f"Response Text: {response.text[:500]}")  # Show first 500 chars
        
#         if response.status_code == 200:
#             return response.json()['output']
#         else:
#             return f"Error: Server returned status code {response.status_code}"
#     except requests.exceptions.RequestException as e:
#         return f"Connection error: {str(e)}"
#     except ValueError as e:  # JSON decode error
#         return f"Error parsing response: {str(e)}\nResponse text: {response.text[:200]}..."


st.title('Langchain demo with 2 models')
input_text = st.text_input("Enter your query")


# Function to call FastAPI endpoint dynamically based on model
def get_model_response(input_text, model):
    try:
        response = requests.post(
            f"http://localhost:8000/{model}",
            json={"prompt": input_text})

        if response.status_code == 200:
            return response.json()["output"]
        else:
            return f"Error: {response.status_code}\n{response.text[:300]}"
    except requests.exceptions.RequestException as e:
        return f"Connection error: {str(e)}"

# Display output
if input_text:
    st.write(get_model_response(input_text, model_choice))

