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

    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)
        thought_text = ""
        result_text = ""

        with st.chat_message("assistant"):
            thought_expander = st.expander("思考中。。。（点击查看详情）", expanded=False)
            thought_placeholder = thought_expander.empty()
            result_placeholder = st.empty()

            for stage, chunk in res_stream:
                if stage == "thought":
                    thought_text += chunk
                    thought_placeholder.markdown(f"思考中：{thought_text}")
                elif stage == "result":
                    result_text += chunk
                    result_placeholder.write(result_text)

        assistant_response = result_text.strip()
        st.session_state["message"].append({"role": "assistant", "content": assistant_response})
        st.rerun()






















