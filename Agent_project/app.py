import time
import uuid

import streamlit as st
from agent.langgraph_agent import MainAgent

# 标题
st.title("智扫通机器人智能客服")
st.divider()

# 生成或获取session_id
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "agent" not in st.session_state:
    st.session_state["agent"] = MainAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt, st.session_state["session_id"])

        def capture(generator, cache_list):
            full_response = []
            for chunk in generator:
                full_response.append(chunk)
                yield chunk
            # 保存完整响应
            cache_list.extend(full_response)

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        full_response = "".join(response_messages)
        st.session_state["message"].append({"role": "assistant", "content": full_response})
        st.rerun()