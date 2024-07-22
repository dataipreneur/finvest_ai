import streamlit as st
from agent import financial_react_agent

AGENT_NAME = "Finvest AI"

# Streamlit UI configuration
st.set_page_config(
    page_title=AGENT_NAME,
    page_icon="💸💰",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "Report a bug": "https://github.com/adiagarwalrock/finvest-ai/issues",
        "About": "Finvest AI leverages cutting-edge AI technology to provide instant and comprehensive financial insights.",
    },
)

st.title(f"{AGENT_NAME} 💸💰")
st.caption(f"🤖 Chat with {AGENT_NAME} 🔌 by Y-Finance and Llama3")


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": f"Hi! I am your friendly neighborhood financial analyst",
        }
    ]


if "react_agent" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.react_agent = financial_react_agent

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.react_agent.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)

        message = {"role": "assistant", "content": response_stream.response}

        st.session_state.messages.append(message)
