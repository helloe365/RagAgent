import unittest

from langchain_core.documents import Document

from agent.react_agent import _content_to_text, _next_assistant_delta
from rag.rag_service import build_context_text


class StreamAndContextTests(unittest.TestCase):
    def test_build_context_text_keeps_all_documents(self):
        docs = [
            Document(page_content="内容A", metadata={"source": "a.txt"}),
            Document(page_content="内容B", metadata={"source": "b.txt"}),
        ]

        context = build_context_text(docs)

        self.assertIn("【参考资料1】", context)
        self.assertIn("内容A", context)
        self.assertIn("【参考资料2】", context)
        self.assertIn("内容B", context)
        self.assertIn("\n", context)

    def test_next_assistant_delta_returns_increment(self):
        previous = "你好"
        current = "你好，我是客服助手"
        self.assertEqual("，我是客服助手", _next_assistant_delta(current, previous))

    def test_next_assistant_delta_handles_non_prefix_updates(self):
        previous = "旧内容"
        current = "新内容"
        self.assertEqual("新内容", _next_assistant_delta(current, previous))

    def test_content_to_text_handles_text_blocks(self):
        content = [{"type": "text", "text": "你好"}, {"type": "text", "text": "世界"}]
        self.assertEqual("你好世界", _content_to_text(content))


if __name__ == "__main__":
    unittest.main()
