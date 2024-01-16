from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
from Main import get_qa_chain  # Assuming Main.py contains the get_qa_chain function
import os

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]

st.title("MoraSparks - AI Driven Hack")

client = OpenAI(api_key=api_key)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call get_qa_chain function here
    chain = get_qa_chain()
    qa_chain_response = chain(prompt)
    # Adjust parameters as needed

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add the response from the get_qa_chain function to messages
    st.session_state.messages.append({"role": "assistant", "content": qa_chain_response})


# this only for testing 123