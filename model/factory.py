from abc import ABC,abstractmethod
from typing import Optional
import os
from pydantic import SecretStr
from langchain_core.embeddings import Embeddings
from langchain_openai.chat_models.base import ChatOpenAI, BaseChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod #强制子类实现指定的方法（常用于接口或工厂类）。如果子类未实现所有抽象方法，则不能实例化该子类。
    def generator(self) -> Optional[Embeddings | BaseChatOpenAI]:
        pass


def _resolve_api_key(api_key:str) -> SecretStr:
    api_key = os.path.expandvars(api_key)
    if api_key.startswith("${") and api_key.endswith("}"):
        env_name = api_key[2:-1]
        api_key = os.environ.get(env_name, "")

    if not api_key:
        raise ValueError("请先配置有效的 API Key，例如设置 GITHUB_API_KEY 环境变量")

    return SecretStr(api_key)


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatOpenAI]:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            base_url=rag_conf["base_url_chat"],
            api_key=_resolve_api_key(rag_conf["api_key_chat"]),
        )

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatOpenAI]:
        return OpenAIEmbeddings(
            model=rag_conf["embedding_model_name"],
            base_url=rag_conf["base_url_embed"],
            api_key=_resolve_api_key(rag_conf["api_key_embed"]),
        )

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()