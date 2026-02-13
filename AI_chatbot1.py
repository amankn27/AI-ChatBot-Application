import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

st.title("Falcon AI")

if 'model' not in st.session_state:
    st.session_state['model'] = OpenAI(
        api_key=HF_TOKEN,
        base_url="https://router.huggingface.co/v1"
    )

if 'message' not in st.session_state:
    st.session_state['message'] = []

st.sidebar.title('Model parameters')
temperature = st.sidebar.slider("temperature",min_value=0.0,max_value=2.0,value=0.7,step=0.1)
max_tokens = st.sidebar.slider("max_tokens",min_value=1,max_value=1500,value=127)

for message in st.session_state['message']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


if prompt := st.chat_input("Enter your query"):
    st.session_state['message'].append({"role": "user", "content": prompt})     
    with st.chat_message('user'):
        st.markdown(prompt)   

    with st.chat_message('assistant'):
        client = st.session_state['model']
        stream = client.chat.completions.create(
             model="meta-llama/Llama-3.1-8B-Instruct",
            messages = [
                {"role": message["role"], "content": message["content"]} for message in st.session_state['message']
            ],
            temperature = temperature,
            max_tokens = max_tokens,
            stream=True 
        )  
    response = st.write_stream(stream)
    st.session_state['message'].append({"role": "assistant","content": response})
    