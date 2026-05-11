from langchain.agents import create_agent

from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import log_before_model, monitor_tool, report_prompt_switch
from model.factory import get_chat_model
from utils.prompt_loader import load_system_prompts


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                blocks.append(str(item.get("text", "")))
            elif isinstance(item, str):
                blocks.append(item)
        return "".join(blocks)
    return str(content) if content is not None else ""


def _next_assistant_delta(current_text: str, previous_text: str) -> str:
    if not current_text:
        return ""
    if current_text == previous_text:
        return ""
    if current_text.startswith(previous_text):
        return current_text[len(previous_text) :]
    return current_text


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=get_chat_model(),
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_weather,
                get_user_location,
                get_user_id,
                get_current_month,
                fetch_external_data,
                fill_context_for_report,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        assistant_text = ""
        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            messages = chunk.get("messages", [])
            if not messages:
                continue
            latest_message = messages[-1]
            if getattr(latest_message, "type", "") != "ai":
                continue
            current_text = _content_to_text(getattr(latest_message, "content", "")).strip()
            delta = _next_assistant_delta(current_text=current_text, previous_text=assistant_text)
            if not delta:
                continue
            assistant_text = current_text
            yield delta


if __name__ == "__main__":
    agent = ReactAgent()

    # for chunk in agent.execute_stream("给我生成我的使用报告"):
    for chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
        print(chunk, end="", flush=True)
