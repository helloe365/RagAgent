# RagAgentProject

一个基于 **LangChain + RAG + Streamlit** 的智能客服学习项目，聚焦扫地/扫拖机器人场景，支持知识库检索问答、天气辅助建议、以及用户使用报告生成。

## 项目亮点

- **RAG 检索增强**：将本地知识文档向量化后检索，回答更贴合领域知识
- **ReAct 工具调用**：Agent 可按需调用天气、用户信息、外部记录等工具
- **动态提示词切换**：普通咨询与“报告生成”场景自动切换系统提示词
- **流式输出体验**：前端展示“思考中 + 最终回答”的流式交互
- **可靠知识库索引**：通过每个源文件的内容版本和 SQLite 清单实现幂等更新与失败恢复

## 技术栈

- Python
- Streamlit
- LangChain / LangGraph
- Chroma（向量数据库）
- OpenAI 兼容接口模型（Embedding + Chat）
- SQLite（源文件索引清单）

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
├─ md5_db/                     # 知识库源文件索引清单
├─ logs/                       # 日志目录
└─ tests/                      # 单元测试
```

## 快速开始

### 1. 环境准备

- 建议 Python 3.10+

### 2. 安装依赖

使用已解析的依赖锁文件安装：

```bash
python -m pip install -r requirements.lock
```

也可用 [uv](https://docs.astral.sh/uv/) 创建项目环境：

```bash
uv sync --extra test
```

### 3. 配置环境变量（Windows PowerShell）

`config/rag.yml` 中默认使用了环境变量：

- `GITHUB_API_KEY`（Embedding）
- `GLM_API_KEY`（Chat）

在 PowerShell 中设置：

```powershell
$env:GITHUB_API_KEY="你的Key"
$env:GLM_API_KEY="你的Key"
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
python -m unittest discover -s tests -v
```

## 功能说明

1. **智能问答**：通过 `rag_summarize` 从向量库检索扫地机器人相关知识并回答用户问题。
2. **天气辅助建议**：通过 `get_weather` 获取实时天气，辅助保养建议。
3. **用户报告生成**：按固定流程调用工具（用户ID、月份、外部记录、上下文注入）生成个性化使用报告。
4. **会话上下文**：界面在每轮调用时传入此前完成的用户与助手消息；可使用“清空对话”开始新会话。
5. **索引更新**：同一文件内容不会重复入库；内容更新会先写入新分块，再清理旧分块。删除源文件不会自动删除既有索引。
