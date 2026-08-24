# RagAgent Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 RagAgent 的伪多轮、非幂等索引、脆弱 CSV 解析及配置依赖不一致问题。

**Architecture:** 保留 Streamlit、LangChain Agent、Chroma 和 SQLite；只在现有入口增加明确消息裁剪、确定性分块 ID 与 SQLite 索引清单、标准 CSV 解析。一个实现任务内按四个独立红—绿循环推进，避免新增框架和无关重构。

**Tech Stack:** Python 3.10+、unittest、Streamlit、LangChain/LangGraph、Chroma、SQLite、标准库 csv。

**Spec:** `docs/superpowers/specs/2026-08-24-ragagent-reliability-design.md`

## Global Constraints

- 历史只接受 `user`/`assistant`，最多 20 条和 8,000 字符；系统提示词仍只由 `create_agent` 管理。
- 稳定 `source_id` 来自知识库根目录下的规范化相对路径；内容版本来自文件哈希；分块 ID 必须确定性生成。
- 新分块全部写入后才能切换 SQLite 当前版本；失败必须保留旧版本并允许相同操作安全重试。
- CSV 使用标准库并以 `utf-8-sig` 读取，支持引号逗号、空字段、BOM 和多行字段。
- 聊天密钥统一为 `GLM_API_KEY`，Embedding 密钥保持 `GITHUB_API_KEY`；不得写入真实密钥。
- 日志不得记录 API key、文档正文或完整模型输入；不得删除或迁移现有数据、模型、向量库和输出。
- 只修改本任务相关文件；不全仓格式化，不推送、不合并、不强推。

---

### Task 1: 沿现有调用链完成四项可靠性修复

**Files:**
- Modify: `app.py`
- Modify: `agent/react_agent.py`
- Modify: `rag/vector_store.py`
- Modify: `agent/tools/agent_tools.py`
- Modify: `model/factory.py`
- Modify: `README.MD`
- Create: `.env.example`
- Create: `pyproject.toml`
- Create: `requirements.lock`
- Create: `tests/test_conversation_history.py`
- Create: `tests/test_vector_store_indexing.py`
- Create: `tests/test_external_csv.py`
- Modify: `tests/test_stream_and_context.py` only if an existing assertion needs to use the new public signature.

**Interfaces:**
- `normalize_history(history: list[dict] | None, max_messages: int = 20, max_chars: int = 8000) -> list[dict[str, str]]` returns valid messages in original order.
- `ReactAgent.execute_stream(query: str, history: list[dict] | None = None)` sends normalized history followed by exactly one current user message.
- `VectorStoreService.load_document()` remains the public indexing entrypoint and operates per source through deterministic IDs.
- `parse_external_records(path: str) -> dict[str, dict[str, dict[str, str]]]` parses all rows before returning; `generate_external_data()` replaces the cache only after success.

- [ ] **Step 1: Write failing conversation-history tests**

Create a small fake agent whose `stream()` records `input_dict`. Construct `ReactAgent` with `__new__`, assign the fake to `.agent`, exhaust `execute_stream`, and assert the literal submitted message list:

```python
history = [
    {"role": "user", "content": "我叫小李"},
    {"role": "assistant", "content": "记住了"},
]
list(agent.execute_stream("我叫什么？", history=history))
self.assertEqual(
    history + [{"role": "user", "content": "我叫什么？"}],
    fake_agent.last_input["messages"],
)
```

Add separate tests for empty history, unsupported roles/empty content, 21 messages retaining the latest 20, and character overflow retaining the newest complete messages without exceeding 8,000 characters.

- [ ] **Step 2: Run the conversation tests and verify RED**

Run: `python -m unittest tests.test_conversation_history -v`

Expected: FAIL because `execute_stream` has no `history` parameter and `normalize_history` does not exist.

- [ ] **Step 3: Implement minimal conversation history support**

In `agent/react_agent.py`, normalize a copied history list and build:

```python
input_dict = {
    "messages": normalize_history(history) + [
        {"role": "user", "content": query},
    ]
}
```

In `app.py`, take the history snapshot before appending the current prompt, pass it to `execute_stream`, and add only a small “清空对话” button that empties `st.session_state["message"]` and reruns. Do not introduce a conversation store class or persistence layer.

- [ ] **Step 4: Run conversation and existing stream tests and verify GREEN**

Run: `python -m unittest tests.test_conversation_history tests.test_stream_and_context -v`

Expected: PASS.

- [ ] **Step 5: Write failing indexing tests**

Use `TemporaryDirectory`, real SQLite, simple `Document` objects and an in-memory fake vector store implementing `add_documents(documents, ids)`, `delete(ids)` and metadata filtering. Inject the fake dependencies without creating a real embedding client. Cover these observable cases:

```python
service.load_document()
first_ids = set(fake_store.documents)
service.load_document()
self.assertEqual(first_ids, set(fake_store.documents))

source.write_text("new content", encoding="utf-8")
service.load_document()
self.assertNotEqual(first_ids, set(fake_store.documents))
self.assertTrue(all(d.metadata["content_version"] == current_hash
                    for d in fake_store.documents.values()))
```

Add one test where `add_documents` fails after a partial write and retry does not increase the final block count, plus one where old-ID deletion fails after the SQLite switch and the next run removes the stale IDs.

- [ ] **Step 6: Run indexing tests and verify RED**

Run: `python -m unittest tests.test_vector_store_indexing -v`

Expected: FAIL because the current MD5-only implementation has no stable source/version manifest or recovery state.

- [ ] **Step 7: Implement the minimal idempotent indexing protocol**

Move the nested SQLite helpers to private module-level functions. Create a `source_index` table with `source_id`, `source_path`, `content_version`, `chunk_ids_json`, `stale_chunk_ids_json`, and `updated_at`. Use SHA-256 of the normalized relative path for `source_id`, the existing file hash as `content_version`, and IDs shaped as `source_id:content_version:index`.

For each file: reconcile recorded stale IDs first; skip only when the current version matches and no stale IDs remain; add the new deterministic IDs; transactionally upsert the current IDs and previous IDs as stale; delete stale IDs; then clear the stale list. On add failure, leave SQLite unchanged and log only the safe relative path and stage. Preserve `load_document()` and `get_retriever()` APIs.

- [ ] **Step 8: Run indexing tests and verify GREEN**

Run: `python -m unittest tests.test_vector_store_indexing -v`

Expected: PASS.

- [ ] **Step 9: Write failing CSV tests**

Create temporary `utf-8-sig` CSV fixtures with the existing six-column order. Assert literal parsed values for a quoted comma, an empty field, and a quoted multiline field. Add a five-column row and assert the exception message contains the source filename and `line 2`.

- [ ] **Step 10: Run CSV tests and verify RED**

Run: `python -m unittest tests.test_external_csv -v`

Expected: FAIL because `parse_external_records` does not exist and `split(',')` corrupts complex rows.

- [ ] **Step 11: Implement standard-library CSV parsing**

Add `parse_external_records(path)` using `open(..., newline="", encoding="utf-8-sig")` and `csv.reader`. Skip the header, require exactly six columns, and use `reader.line_num` in errors. Build a local dictionary and return it. In `generate_external_data`, call the parser and update the global cache only after the full parse succeeds. Preserve the existing Chinese result keys and missing-record behavior.

- [ ] **Step 12: Run CSV tests and verify GREEN**

Run: `python -m unittest tests.test_external_csv -v`

Expected: PASS.

- [ ] **Step 13: Align configuration, dependency files and README**

Change the API-key example in `model/factory.py` to mention the resolved environment variable generically or `GLM_API_KEY` for chat. Add `.env.example` containing only:

```dotenv
GITHUB_API_KEY=
GLM_API_KEY=
```

Add `pyproject.toml` for Python `>=3.10` with only imports used by the shipped application, plus a `test` optional group if needed. Produce `requirements.lock` from that dependency declaration using an available resolver; if dependency resolution is unavailable, pin direct dependencies to the currently installed compatible versions and explicitly report that transitive locking was not verified. Update README installation, variables, indexing behavior and test command to match the shipped files.

- [ ] **Step 14: Run full verification**

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q agent rag model utils app.py
git diff --check
```

Expected: all tests pass, compile command exits 0, and `git diff --check` prints no errors.

- [ ] **Step 15: Inspect scope and commit exact files**

Run `git status --short`, inspect `git diff --stat` and `git diff`. Confirm no data, Chroma, SQLite, logs, secrets, caches or unrelated files are present. Stage only the files listed in this task; do not use `git add .`.

Commit message: `fix: make RagAgent state and indexing reliable`.
