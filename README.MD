# RagAgentProject

一个基于 **LangChain + RAG + Streamlit** 的智能客服学习项目，聚焦扫地/扫拖机器人场景，支持知识库检索问答、天气辅助建议、以及用户使用报告生成。

## 项目亮点

- **RAG 检索增强**：将本地知识文档向量化后检索，回答更贴合领域知识
- **ReAct 工具调用**：Agent 可按需调用天气、用户信息、外部记录等工具
- **动态提示词切换**：普通咨询与“报告生成”场景自动切换系统提示词
- **流式输出体验**：前端展示“思考中 + 最终回答”的流式交互
- **知识库去重**：通过文件 MD5 + SQLite 记录，避免重复入库

## 技术栈

- Python
- Streamlit
- LangChain / LangGraph
- Chroma（向量数据库）
- OpenAI 兼容接口模型（Embedding + Chat）
- SQLite（MD5 去重记录）

## 目录结构

```text
RagAgentProject/
├─ app.py                      # Streamlit 入口
├─ agent/                      # Agent 与工具逻辑
├─ rag/                        # RAG 检索与向量库服务
├─ model/                      # 模型工厂（Chat/Embedding）
├─ config/                     # yml 配置（模型、向量库、提示词、外部数据）
├─ prompts/                    # 系统提示词 / RAG提示词 / 报告提示词
├─ data/                       # 知识文档与外部记录
├─ chroma_db/                  # Chroma 持久化目录
├─ md5_db/                     # 文件MD5去重数据库目录
├─ logs/                       # 日志目录
└─ tests/                      # 单元测试
```

## 快速开始

### 1. 环境准备

- 建议 Python 3.10+

### 2. 安装依赖

项目当前未提供 `requirements.txt`，可先安装以下核心依赖：

```bash
pip install streamlit pyyaml pydantic langchain langgraph langchain-openai langchain-chroma chromadb langchain-community langchain-text-splitters pypdf
```

### 3. 配置环境变量（Windows PowerShell）

`config/rag.yml` 中默认使用了环境变量：

- `GITHUB_API_KEY`（Embedding）
- `CAT_API_KEY`（Chat）

在 PowerShell 中设置：

```powershell
$env:GITHUB_API_KEY="你的Key"
$env:CAT_API_KEY="你的Key"
```

### 4. 准备知识库（首次或文档更新后执行）

```bash
python rag\vector_store.py
```

### 5. 启动应用

```bash
streamlit run app.py
```

### 6. 运行测试

```bash
python -m unittest tests\test_stream_and_context.py
```

## 功能说明

1. **智能问答**：通过 `rag_summarize` 从向量库检索扫地机器人相关知识并回答用户问题。
2. **天气辅助建议**：通过 `get_weather` 获取实时天气，辅助保养建议。
3. **用户报告生成**：按固定流程调用工具（用户ID、月份、外部记录、上下文注入）生成个性化使用报告。
