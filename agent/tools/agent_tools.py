import json
import os
import random
import urllib.error
import urllib.request
from threading import Lock
from urllib.parse import quote

from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from langchain_core.tools import tool

rag: RagSummarizeService | None = None
rag_init_lock = Lock()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]
cities = ["深圳", "合肥", "杭州", "北京", "上海", "广州", "成都", "西安", "武汉", "南京"]

external_data = {}
WEATHER_TIMEOUT = float(os.getenv("WEATHER_TIMEOUT", "8"))
WEATHER_SOURCE_URL = "https://wttr.in"


def _first_text(value: object) -> str:
    if isinstance(value, list) and value:
        first_item = value[0]
        if isinstance(first_item, dict):
            return str(first_item.get("value", "")).strip()
        return str(first_item).strip()
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _fetch_weather_payload(city: str) -> dict:
    encoded_city = quote(city.strip())
    url = f"{WEATHER_SOURCE_URL}/{encoded_city}?format=j1&lang=zh-cn"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(request, timeout=WEATHER_TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("天气接口返回的数据格式不正确")
    return payload


def _format_weather_result(city: str, payload: dict) -> str:
    current_condition = payload.get("current_condition") or []
    weather_days = payload.get("weather") or []

    current = current_condition[0] if current_condition else {}
    today = weather_days[0] if weather_days else {}

    desc = _first_text(current.get("weatherDesc")) or "天气信息"
    temp_c = _first_text(current.get("temp_C")) or "未知"
    feels_like_c = _first_text(current.get("FeelsLikeC")) or "未知"
    humidity = _first_text(current.get("humidity")) or "未知"
    wind_dir = _first_text(current.get("winddir16Point")) or "未知"
    wind_speed = _first_text(current.get("windspeedKmph")) or "未知"
    max_temp = _first_text(today.get("maxtempC")) or "未知"
    min_temp = _first_text(today.get("mintempC")) or "未知"

    rain_chances = []
    for item in today.get("hourly", []) if isinstance(today, dict) else []:
        if isinstance(item, dict):
            chance = _safe_int(item.get("chanceofrain"))
            if chance is not None:
                rain_chances.append(chance)
    rain_desc = f"，今日最大降雨概率{max(rain_chances)}%" if rain_chances else ""

    return (
        f"城市{city}实时天气（来源：wttr.in）：{desc}，当前气温{temp_c}℃，"
        f"体感{feels_like_c}℃，湿度{humidity}% ，风向{wind_dir}，风速{wind_speed}km/h，"
        f"今日最高{max_temp}℃，最低{min_temp}℃{rain_desc}"
    )


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    city = (city or "").strip()
    if not city:
        return "请输入有效的城市名称后再查询天气。"

    try:
        payload = _fetch_weather_payload(city)
        return _format_weather_result(city, payload)
    except urllib.error.HTTPError as exc:
        logger.warning(f"[get_weather]天气接口HTTP错误：city={city}, status={exc.code}, reason={exc.reason}")
    except urllib.error.URLError as exc:
        logger.warning(f"[get_weather]天气接口网络错误：city={city}, error={exc}")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(f"[get_weather]天气接口解析失败：city={city}, error={exc}")
    except Exception as exc:
        logger.warning(f"[get_weather]查询天气时发生未知错误：city={city}, error={exc}")

    return f"暂时无法查询到城市{city}的实时天气，请稍后再试。"


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    global rag
    if rag is None:
        with rag_init_lock:
            if rag is None:
                rag = RagSummarizeService()
    return rag.rag_summarize(query)

@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    return random.choice(cities)


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f'外部数据文件不存在，请确保路径正确: {external_data_path}')

        with open(external_data_path,'r',encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(',')
                user_id: str = arr[0].replace('"', '') #把 " 替换为空字符串
                feature: str = arr[1].replace('"', '')
                efficiency: str = arr[2].replace('"', '')
                consumables: str = arr[3].replace('"', '')
                comparision: str = arr[4].replace('"', '')
                time: str = arr[5].replace('"', '')

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time]={
                    "特征": feature,
                    "效率": efficiency,
                    "消耗品": consumables,
                    "对比": comparision,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        return json.dumps(external_data[user_id][month], ensure_ascii=False)
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"


# if __name__ == '__main__':
    # res = fetch_external_data("1001", "2025-01")
    # print(res)
    # print(get_weather("成都"))
