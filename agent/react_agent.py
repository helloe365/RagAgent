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


def normalize_history(
    history: list[dict] | None, max_messages: int = 20, max_chars: int = 8000
) -> list[dict[str, str]]:
    valid_messages: list[dict[str, str]] = []
    for message in history or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue
        valid_messages.append({"role": role, "content": content})

    if max_messages <= 0 or max_chars <= 0:
        return []

    recent_messages = valid_messages[-max_messages:]
    retained_reversed: list[dict[str, str]] = []
    total_chars = 0
    for message in reversed(recent_messages):
        message_chars = len(message["content"])
        if total_chars + message_chars > max_chars:
            break
        retained_reversed.append(message)
        total_chars += message_chars
    return list(reversed(retained_reversed))


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


def _message_stage(latest_message: object) -> str | None:
    if getattr(latest_message, "type", "") != "ai":
        return None
    tool_calls = getattr(latest_message, "tool_calls", None)
    if tool_calls:
        return "thought"
    return "result"


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

    def execute_stream(self, query: str, history: list[dict] | None = None):
        input_dict = {
            "messages": normalize_history(history) + [{"role": "user", "content": query}]
        }

        thought_text = ""
        result_text = ""
        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            messages = chunk.get("messages", [])
            if not messages:
                continue
            latest_message = messages[-1]
            stage = _message_stage(latest_message)
            if stage is None:
                continue
            current_text = _content_to_text(getattr(latest_message, "content", "")).strip()
            previous_text = thought_text if stage == "thought" else result_text
            delta = _next_assistant_delta(current_text=current_text, previous_text=previous_text)
            if not delta:
                continue
            if stage == "thought":
                thought_text = current_text
            else:
                result_text = current_text
            yield stage, delta


if __name__ == "__main__":
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
    # for _, chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
        print(chunk, end="", flush=True)
