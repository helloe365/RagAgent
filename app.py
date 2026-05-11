from time import sleep

import streamlit as st
from agent.react_agent import ReactAgent

# 标题
st.title("小小智扫机器人智能客服")
st.divider()

if "agent" not in st.session_state:
    try:
        st.session_state["agent"] = ReactAgent()
        st.session_state["agent_error"] = ""
    except Exception as exc:
        st.session_state["agent"] = None
        st.session_state["agent_error"] = str(exc)

if "message" not in st.session_state:
    st.session_state["message"] = []

if st.session_state.get("agent") is None:
    st.error(f"智能客服初始化失败：{st.session_state.get('agent_error', '未知错误')}")
    st.stop()


for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})
    response_messages = []

    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    sleep(0.01)  # 模拟打字效果
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        assistant_response = "".join(response_messages).strip()
        st.session_state["message"].append({"role": "assistant", "content": assistant_response})
        st.rerun()






















