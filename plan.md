# RagAgent 升级最终执行计划

> 整合自 `plan_gpt.md` 与 `plan_gemini.md`，基于当前代码库实际情况批判性裁决后产出。
> 本文件为**规划文档**，不包含任何代码改动。所有实施步骤在对应阶段再落地。

---

## 1. 现状诊断

### 1.1 当前核心技术栈
- **前端**：`Streamlit`（`app.py`，54 行，单页、单会话）
- **Agent 层**：`langchain.agents.create_agent`（已基于 LangGraph）+ 自定义 middleware（`wrap_tool_call` / `before_model` / `dynamic_prompt`），通过 `runtime.context["report"]` 做提示词切换
- **RAG 层**：`langchain_chroma.Chroma` + `RecursiveCharacterTextSplitter`，单一稠密向量召回，`k=3`
- **模型层**：`ChatOpenAI`（`newapi.nki.pw` 代理的 `deepseek-v4-flash`）+ `OpenAIEmbeddings`（`models.github.ai` 渠道的 `text-embedding-3-small`），使用模块级单例缓存
- **工具集**：`rag_summarize` / `get_weather`（wttr.in 真实调用）/ `get_user_location|id|current_month`（**随机桩**）/ `fetch_external_data`（CSV）/ `fill_context_for_report`（空工具触发报告分支）
- **存储**：Chroma 向量库（`chroma_db/`）+ SQLite 去重库（`md5_db/md5.sqlite3`）
- **配置**：`config/*.yml` + `utils/config_handler.py`，已支持 `${ENV_VAR}` 展开
- **测试**：仅 1 个 `unittest` 文件（4 个用例，只覆盖纯函数）
- **日志**：`logging` + 文件按日期命名（**无轮转**）

### 1.2 主要技术债务 / 升级阻碍

| 序号 | 位置 | 问题 | 风险等级 |
|---|---|---|---|
| D1 | 项目根 | **无 `requirements.txt` / `pyproject.toml`**，依赖版本完全隐式 | 🔴 严重 |
| D2 | `agent/tools/agent_tools.py:174-191` | `records.csv` 用 `line.split(',')` 解析，**字段内含逗号/引号就炸** | 🔴 严重 |
| D3 | `rag/vector_store.py:121-154` | MD5 去重只做**新增不做更新**：源文件内容变更后旧 chunk 永远留在向量库，新旧并存污染召回 | 🔴 严重 |
| D4 | `utils/logger_handler.py:15` | 参数名拼写错误 `console_leverr`；无 `RotatingFileHandler` | 🟡 中 |
| D5 | `agent/tools/agent_tools.py:23, 143-191` | `external_data` 模块级字典 + `rag` 单例 + `user_ids/month_arr/cities` 随机桩 → 多会话串扰、报告生成无意义 | 🟡 中 |
| D6 | `rag/rag_service.py:19-25` | 每次 `RagSummarizeService()` 会重建 `vector_store` + `chain`；`agent_tools.py` 已加锁单例但 lazy，**冷启动延迟集中在首个用户** | 🟡 中 |
| D7 | `rag/vector_store.py:25-30` | `RecursiveCharacterTextSplitter` 固定 `chunk_size=200`，对 PDF 和 QA 类文档**语义切断严重**；PDF 没有按页/章节元数据 | 🟡 中 |
| D8 | `agent/react_agent.py:77` | 通过 `runtime.context["report"]` 隐式传状态；`monitor_tool` 中直接改 `context`（副作用藏在日志工具里），**状态与可观测性耦合** | 🟡 中 |
| D9 | `app.py` | 直接持有 `ReactAgent` 对象，**前后端强耦合**，无法被外部前端 / 企业 IM 复用 | 🟢 低 |
| D10 | `config/rag.yml:2` | `chat_model_name: "[渠道2]deepseek-ai/deepseek-v4-flash"` 形似手填渠道前缀，**渠道号变更即全链路挂**，依赖单一第三方代理 | 🟡 中 |
| D11 | 全局 | 无 CI、无 lint / format / type check / coverage；测试不覆盖 Agent / 工具 / RAG | 🟡 中 |
| D12 | `agent/tools/agent_tools.py:107-116` | `get_weather` 的外网调用无缓存、无失败重试 | 🟢 低 |

---

## 2. 方案决策

### 2.1 采纳 GPT 方案的部分（主干）

- ✅ **P0/P1 先行的增量节奏**（先可观测 → 再算法 → 最后服务化）
- ✅ **P1.1 CSV 健壮化（`csv.DictReader`）**：对应 D2 的 bug 修复
- ✅ **P1.2 日志轮转 + trace_id + 结构化工具日志**：对应 D4
- ✅ **P2.1 Hybrid + 轻量重排 + 查询改写**（先 BM25 并集召回，再轻量 re-rank）
- ✅ **P2.2 切片元数据标准化（`source`/`doc_id`/`chunk_id`/`updated_at`）**：是 D3 的根治
- ✅ **P2.3 回答引用输出**
- ✅ **P3.1 显式流程节点 + P3.2 工具契约标准化**：对应 D8
- ✅ **P4 FastAPI 抽离 + session_id/user_id 贯穿**：对应 D5/D9
- ✅ **P5.1 离线评测集**
- ✅ **feature flag 灰度 + 可回退**

### 2.2 采纳 Gemini 方案的部分（补强）

- ✅ **语义切分替代固定 200 字符**：对应 D7。使用 `langchain_experimental.SemanticChunker`（复用现有 embeddings，不引入新模型依赖）
- ✅ **`watchdog` 监控 `data/` 目录**：搭配 P2.2 的 `doc_id` 元数据做增量 upsert。仅作开发期可选项
- ✅ **会话持久化（SQLite）**：对应 Streamlit `session_state` 刷新丢失；**不用 Redis**（YAGNI）
- ✅ **Router Node**：用 `@before_model` middleware 做轻量意图分类，**不做 LangGraph 级别重写**

### 2.3 完全摒弃 / 推迟的部分

| 摒弃项 | 出处 | 原因 |
|---|---|---|
| 🚫 Next.js / Vue / 微信小程序前端重写 | Gemini Phase 3 | 当前是内部 demo，YAGNI |
| 🚫 Redis 跨会话记忆 | Gemini Phase 1 | 复杂度溢出；SQLite 够用 |
| 🚫 LangGraph 级别的多 Agent 彻底重写 | Gemini Phase 2 | `create_agent` 已基于 LangGraph |
| 🚫 `WebSearchTool`（Tavily / DuckDuckGo） | Gemini Phase 1 | 引入外部 API Key、边界模糊 |
| 🚫 `DataAnalyticsTool`（pandas 动态分析 + 图表） | Gemini Phase 1 | 偏离核心域，安全面放大 |
| 🚫 `Langfuse` / `Arize Phoenix` 接入 | Gemini Phase 3 | 本地 demo 过重；结构化日志 + trace_id 足够 |
| 🚫 SFT 反馈闭环 | Gemini Phase 3 | 无模型训练能力支撑 |
| 🚫 `bge-reranker` cross-encoder（首期） | GPT P2.1 | 需 `sentence-transformers` + 本地权重；先用基于现有 LLM 的轻量 re-rank |

### 2.4 两个方案冲突点判决

| 冲突点 | Gemini | GPT | 采纳 | 理由 |
|---|---|---|---|---|
| 多 Agent 架构 | LangGraph 多 Agent 重写 | 显式节点化现有 Agent | **GPT** | 现有 `create_agent` 已是 LangGraph，重写成本>收益 |
| 前端 | 换 Next.js | 保留 Streamlit，后端抽 FastAPI | **GPT** | 项目阶段决定 |
| 切分策略 | 语义切分 | 元数据标准化 | **两者互补** | 共存 |
| 记忆 | Redis / SQLite | 未涉及 | **Gemini 的 SQLite** | 简单够用 |
| Reranker | Cross-Encoder (bge) | Cross-Encoder 或轻量打分 | **轻量 LLM 打分优先** | 最小依赖面 |

---

## 3. 最终执行计划

> 全程遵循「小步提交、可回退、可灰度」。任何算法/架构改动必须同时更新 `config/features.yml` 开关与评测对比。

### 阶段零：地基稳固（P0，必做，先于一切）

- [ ] **任务 0.1**：建立依赖与环境基线
  - *涉及文件*：新增 `pyproject.toml`（推荐）或 `requirements.txt`、`.env.example`
  - *动作*：盘点实际 import 并锁版本（`langchain`, `langchain-chroma`, `langchain-openai`, `langchain-experimental`, `streamlit`, `pyyaml`, `pypdf`, `chromadb` 等）；加入 `python-dotenv`；`pyproject` 用 `>=,<` 次版本区间锁
  - *预期结果*：`pip install -e .` 一键就绪

- [ ] **任务 0.2**：引入开发工具链
  - *动作*：`ruff`（lint+format）、`mypy`（宽松模式）、`pytest` + `pytest-cov`，配置于 `pyproject.toml`
  - *预期结果*：`ruff check .` / `pytest --cov` 可跑

- [ ] **任务 0.3**：最小 CI
  - *动作*：GitHub Actions 或本地 `scripts/ci.sh`，跑 lint + 测试
  - *预期结果*：PR 前有自动闸门

- [ ] **任务 0.4**：修复两个明显 bug
  - *涉及文件*：`utils/logger_handler.py:15`（`console_leverr` → `console_level`）、`config/rag.yml:2`（验证 `chat_model_name` 实际可用）
  - *预期结果*：冷启动日志干净；模型调用不 404

### 阶段一：稳定性 & 可观测性（P1，小改动）

- [ ] **任务 1.1**：CSV 读取健壮化（对应 D2）
  - *涉及文件*：`agent/tools/agent_tools.py:143-191`
  - *逻辑*：`csv.DictReader`，字段缺失走 `logger.warning` 并跳过
  - *预期结果*：含引号/逗号字段正确解析

- [ ] **任务 1.2**：日志升级（对应 D4）
  - *涉及文件*：`utils/logger_handler.py`、`agent/tools/middleware.py`、`app.py`
  - *逻辑*：`RotatingFileHandler(maxBytes=10MB, backupCount=5)`；`trace_id` 通过 `contextvars` + Filter 注入；工具日志结构化字段 `tool_name / args / status / error / duration_ms`
  - *预期结果*：一次请求所有日志可按 `trace_id` 串联

- [ ] **任务 1.3**：会话持久化
  - *涉及文件*：新建 `utils/session_store.py`（SQLite）、修改 `app.py`
  - *逻辑*：`session_id` 存入 `st.session_state`；消息写入 `sessions.sqlite3` 的 `messages(session_id, role, content, created_at, trace_id)` 表
  - *预期结果*：刷新页面消息不丢；为 P4 FastAPI 铺路

- [ ] **任务 1.4**：UI 小工
  - *涉及文件*：`app.py`
  - *逻辑*：清空会话 / 重试上次问题 / 明确错误提示
  - *预期结果*：交互中断后用户可自助恢复

### 阶段二：RAG 质量升级（P2，大改动，靠评测集守门）

- [ ] **任务 2.0**：先建评测集
  - *涉及文件*：新建 `eval/golden_set.yaml`（≥30 条：问答 20 + 报告 5 + 边界 5）、`eval/run_eval.py`
  - *指标*：hit@k、关键词覆盖、耗时、工具调用成功率

- [ ] **任务 2.1**：切片元数据 + 去重一致性（对应 D3）
  - *涉及文件*：`rag/vector_store.py`、`utils/file_handler.py`
  - *逻辑*：`metadata` 统一 `source/doc_id/chunk_id/updated_at`；替换为**基于 `doc_id` 先删后加的 upsert 语义**

- [ ] **任务 2.2**：语义切分（对应 D7）
  - *涉及文件*：`rag/vector_store.py`、`config/chroma.yml`
  - *逻辑*：新增 `chunk_strategy: semantic | recursive`；semantic 走 `SemanticChunker`；默认保留 recursive 做 fallback

- [ ] **任务 2.3**：Hybrid 检索 + 轻量重排
  - *涉及文件*：`rag/vector_store.py`、`rag/rag_service.py`、`config/chroma.yml`
  - *逻辑*：`BM25Retriever` + `Chroma` 的 `EnsembleRetriever`；取 Top-10 交给 chat_model 简短 prompt 打 0-10 分；`features.hybrid_search` 默认 false 灰度开启

- [ ] **任务 2.4**：查询改写
  - *涉及文件*：`rag/rag_service.py`、新增 `prompts/query_rewrite.txt`
  - *逻辑*：用户口语化 → 2-3 条检索 query，合并召回去重

- [ ] **任务 2.5**：回答引用输出
  - *涉及文件*：`rag/rag_service.py`、`prompts/rag_summarize.txt`、`app.py`
  - *逻辑*：每条结论附 `[source#chunk]`；`app.py` 底部渲染参考资料

- [ ] **任务 2.6**（可选）：`watchdog` 监听 `data/`
  - *触发条件*：仅开发期开启；生产关闭

### 阶段三：Agent 流程状态机化 + 工具契约（P3）

- [ ] **任务 3.1**：Router Node
  - *涉及文件*：`agent/tools/middleware.py` 新增 `@before_model` 的 `intent_router`
  - *逻辑*：小成本 prompt 分类 `intent ∈ {chitchat, rag_qa, report, weather, other}`

- [ ] **任务 3.2**：显式报告流程替代 `context["report"]`（对应 D8）
  - *涉及文件*：`agent/react_agent.py`、`agent/tools/middleware.py`
  - *逻辑*：副作用从 `monitor_tool` 中移除，由独立 middleware / `intent_router` 切换 prompt

- [ ] **任务 3.3**：工具契约标准化
  - *涉及文件*：`agent/tools/agent_tools.py`
  - *逻辑*：返回 `{"ok", "data", "error_type", "error_msg"}`；错误分类 `NETWORK_ERROR / PARAM_ERROR / DATA_NOT_FOUND / UPSTREAM_LIMIT`；`get_weather` 加 1 次重试 + 60s 内存缓存

- [ ] **任务 3.4**：桩工具透明化（对应 D5）
  - *涉及文件*：`agent/tools/agent_tools.py`、`config/agent.yml`
  - *逻辑*：明确标注 `stub`；提供 demo 模式 vs 真实模式切换

### 阶段四：服务化（P4，可独立分支）

- [ ] **任务 4.1**：新建 `api/` 目录（FastAPI）
  - *接口*：`POST /chat`（SSE）、`POST /chat/session/clear`、`GET /health`
  - *逻辑*：`ReactAgent` 依赖注入；接收 `session_id / user_id`

- [ ] **任务 4.2**：`app.py` 改为 FastAPI 客户端
  - *逻辑*：调用 `/chat` SSE 流式渲染

### 阶段五：持续评测 & 追踪（P5）

- [ ] **任务 5.1**：扩展 2.0 的评测集并版本化记录（`eval/runs/*.json`）
- [ ] **任务 5.2**：关键指标结构化日志（延迟、失败率、召回耗时）；暂不引入 Langfuse

### 阶段六：验证与测试（贯穿所有阶段）

每阶段结束必须：
- `ruff check .` 通过
- `pytest --cov` 新代码覆盖 ≥ 80%
- `eval/run_eval.py` 对比上个版本无明显回退
- `git tag` 打里程碑便于回滚

---

## 4. 潜在风险与应对

| 风险点 | 描述 | 应对策略 |
|---|---|---|
| **R1 模型代理不稳定** | `chat_model_name: "[渠道2]deepseek-ai/deepseek-v4-flash"` 强依赖 `newapi.nki.pw` 单一代理 | 任务 0.4 验证可用性；`ChatModelFactory` 增加备用模型配置 |
| **R2 向量库重建代价** | 任务 2.1/2.2 调整元数据 + 语义切分意味着**一次性重建** `chroma_db/` | 提供 `scripts/rebuild_index.py`；保留 `chroma_db.bak/` 快照便于回滚 |
| **R3 SemanticChunker 的 embedding 调用成本** | 对每句打 embedding，大 PDF 调用量可能翻 10× | 离线流水线执行；`.txt` 先用标题分割；`SemanticChunker` 限定 PDF |
| **R4 Hybrid + 重排带来延迟** | 多一次 LLM 调用，p95 可能从 2s → 4s | 评测集卡 p95 ≤ 3.5s；re-rank 默认 off；不达标降级为 BM25 + 向量 RRF |
| **R5 Feature Flag 失控** | 多个开关同时在途 | `config/features.yml` 统一管理 + 启动日志打印 flag + 评测矩阵 |
| **R6 工具契约变更破坏老 prompt** | P3.3 改返回结构 `main_prompt.txt` 会偏离 | 改工具同步更新 prompt；加 `to_text()` 兼容层 |
| **R7 Session 持久化含用户内容 → 合规** | SQLite 存储问答可能含隐私 | 默认保留 N 天自动清理；`SESSION_RETENTION_DAYS` 可配 |
| **R8 没有真正的用户数据** | `get_user_id/location/month` 随机桩 | 任务 3.4 明示 demo 模式 |
| **R9 Windows 路径与 `watchdog`** | Win 下偶有事件丢失 | 开发期才启用；`python scripts/ingest.py --force` 后备 |
| **R10 依赖锁定过严** | 0.1 一锁可能影响未来升级 | `>=,<` 次版本区间；`langchain` 系列放宽 |

---

## 5. 自我反思

**这个计划在实际执行中可能遇到的报错**：
- `SemanticChunker` 需要 `langchain-experimental`——任务 0.1 必须补进依赖
- `BM25Retriever` 依赖 `rank_bm25`——同上
- Chroma 的 `delete(where=...)` 行为因版本而异——任务 2.1 做 API 兼容检查
- FastAPI + Streamlit 并存时端口/跨域要显式配（任务 4.1/4.2）

**容易遗漏的环境步骤**：
- `GITHUB_API_KEY` / `CAT_API_KEY` 在 README 未提及，需补 `.env.example`
- `chroma_db/` 和 `md5_db/` 已 `.gitignore`，重建索引需明确本地清理步骤
- Windows `bash` shell 下路径分隔符差异

**最可能失败的阶段**：阶段二的 2.3（Hybrid + 重排）——延迟和效果需反复回归；按 `false → 灰度 5% → 全量` 三挡推进。

---

**交付边界**：执行时严格按 P0 → P5 顺序推进，每阶段前跑评测集基线，结束后对比报告再合入主干。
