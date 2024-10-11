from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import os

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

client = OpenAI(api_key=api_key, base_url=base_url)

username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")

if "login" not in st.session_state:
    st.session_state.login = False

if "username" not in st.session_state:
    st.session_state.username = ""
if "password" not in st.session_state:
    st.session_state.password = ""

if "msg" not in st.session_state:
    st.session_state.msg = []

if "mem" not in st.session_state:
    st.session_state.mem = []

if "cache" not in st.session_state:
    st.session_state.cache = True

if st.session_state.mem and st.session_state.mem[-1]["role"] == "assistant":
    st.session_state.cache = False
elif not st.session_state.mem or st.session_state.mem[-1]["role"] != "assistant":
    st.session_state.cache = True

if "sys" not in st.session_state:
    st.session_state.sys = "You are a smart and helpful assistant."

if "token" not in st.session_state:
    st.session_state.token = 4096
if "temp" not in st.session_state:
    st.session_state.temp = 0.70
if "topp" not in st.session_state:
    st.session_state.topp = 0.70
if "freq" not in st.session_state:
    st.session_state.freq = 0.00
if "pres" not in st.session_state:
    st.session_state.pres = 0.00

if "state" not in st.session_state:
    st.session_state.state = False

def main():
    st.subheader(":robot_face: General Chat", anchor=False)

    with st.sidebar:
        exit_btn = st.button("Exit", "exit_btn", disabled=st.session_state.msg!=[])
        if exit_btn:
            st.session_state.login = False
            st.session_state.username = ""
            st.session_state.password = ""
            st.rerun()

        model_list = [
            "deepseek-ai/DeepSeek-V2.5",
            "Qwen/Qwen2.5-72B-Instruct-128K",
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "deepseek-ai/DeepSeek-Coder-V2-Instruct"]
        model = st.selectbox("Model", model_list, 0, key="model", disabled=st.session_state.msg!=[])

        clear_btn = st.button("Clear", "clear_btn", type="primary", use_container_width=True, disabled=st.session_state.msg==[])
        retry_btn = st.button("Retry", "retry_btn", use_container_width=True, disabled=st.session_state.cache)
        undo_btn = st.button("Undo", "undo_btn", use_container_width=True, disabled=st.session_state.msg==[])

        system_prompt = st.text_area("System Prompt", st.session_state.sys, key="system_prompt", disabled=st.session_state.msg!=[])
        st.session_state.sys = system_prompt

        with st.expander("Parameter Settings"):
            max_tokens = st.slider("Max Tokens", 1, 4096, st.session_state.token, 1, key="max_tokens", disabled=st.session_state.msg!=[])
            st.session_state.token = max_tokens
            temperature = st.slider("Temperature", 0.0, 2.0, st.session_state.temp, 0.01, key="temperature", disabled=st.session_state.msg!=[])
            st.session_state.temp = temperature
            top_p = st.slider("Top P", 0.0, 1.0, st.session_state.topp, 0.01, key="top_p", disabled=st.session_state.msg!=[])
            st.session_state.topp = top_p
            frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, st.session_state.freq, 0.01, key="frequency_penalty", disabled=st.session_state.msg!=[])
            st.session_state.freq = frequency_penalty
            presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, st.session_state.pres, 0.01, key="presence_penalty", disabled=st.session_state.msg!=[])
            st.session_state.pres = presence_penalty
    
    for i in st.session_state.mem:
        with st.chat_message(i["role"]):
            st.markdown(i["content"])
    
    if query := st.chat_input("Say something...", key="query"):
        st.session_state.msg.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        messages = [{"role": "system", "content": system_prompt}] + st.session_state.msg

        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=True)
            result = st.write_stream(chunk.choices[0].delta.content for chunk in response if chunk.choices[0].delta.content is not None)

        st.session_state.msg.append({"role": "assistant", "content": result})
        st.session_state.mem = st.session_state.msg
        st.rerun()
    
    if clear_btn:
        st.session_state.msg = []
        st.session_state.mem = []
        st.session_state.sys = "You are a smart and helpful assistant."
        st.session_state.token = 4096
        st.session_state.temp = 0.70
        st.session_state.topp = 0.70
        st.session_state.freq = 0.00
        st.session_state.pres = 0.00
        st.rerun()
    
    if retry_btn:
        st.session_state.msg.pop()
        st.session_state.mem = []
        st.session_state.state = True
        st.rerun()
    if st.session_state.state:
        for i in st.session_state.msg:
            with st.chat_message(i["role"]):
                st.markdown(i["content"])
        
        messages = [{"role": "system", "content": system_prompt}] + st.session_state.msg

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=True)
        result = st.write_stream(chunk.choices[0].delta.content for chunk in response if chunk.choices[0].delta.content is not None)

        st.session_state.msg.append({"role": "assistant", "content": result})
        st.session_state.mem = st.session_state.msg
        st.session_state.state = False
        st.rerun()
    
    if undo_btn:
        del st.session_state.msg[-1]
        del st.session_state.mem[-1]
        st.rerun()

if not st.session_state.login:
    user_name = st.text_input("Username", "", key="user_name", type="default")
    pass_word = st.text_input("Password", "", key="pass_word", type="password")
    login_btn = st.button("Login", "login_btn", type="primary")

    if login_btn:
        st.session_state.username = user_name
        st.session_state.password = pass_word
        
        if st.session_state.username == username and st.session_state.password == password:
            st.session_state.login = True
        elif st.session_state.username != username or st.session_state.password != password:
            st.warning("Invalid username or password.")

        st.rerun()
else:
    main()