# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

这是一个基于 LangGraph 的文档导入知识库流水线（dataset-rag）。核心功能是将 PDF/Markdown 文档经过解析、图片处理、文本切分、商品名称识别、向量化后写入 Milvus 向量数据库。

## 技术栈与工具

- **包管理**: `uv` — 所有依赖在 `pyproject.toml` 中
- **Python**: >= 3.12
- **流水线引擎**: LangGraph
- **向量数据库**: Milvus (pymilvus)
- **对象存储**: MinIO
- **PDF 解析**: MinerU API（远程服务）
- **嵌入模型**: BGE-M3（稠密 + 稀疏双向量）
- **LLM**: OpenAI 兼容 API（千问等国产模型）
- **日志**: loguru，配置文件输出到 `logs/app_YYYYMMDD.log`

## 命令

```bash
# 运行整个导入流水线（需要先配置 .env）
uv run python main.py

# 查看图结构（打印 ASCII 流程图）
uv run python -m app.import_process.agent.main_graph

# 单独调试某个节点（各节点的 `if __name__ == "__main__"` 块均已写好）
uv run python -m app.import_process.agent.nodes.node_document_split
uv run python -m app.import_process.agent.nodes.node_item_name_recognition
uv run python -m app.import_process.agent.nodes.node_md_img

# 运行集成测试
uv run python -m app.test.test_import_main_graph

# 安装新依赖
uv add <package>
```

## 核心架构

### 流水线 DAG

```
START → node_entry → [PDF→node_pdf_to_md | MD→node_md_img] → node_md_img → node_document_split → node_item_name_recognition → node_bge_embedding → node_import_milvus → END
```

- `node_entry` (`app/import_process/agent/nodes/node_entry.py:9`): 入口节点，根据文件后缀判断走 PDF 还是 Markdown 路径
- `node_pdf_to_md`: 调用 MinerU API 将 PDF 转为 Markdown
- `node_md_img`: 提取 Markdown 中的图片，上传 MinIO，通过 VL 模型生成图片摘要替换原图
- `node_document_split`: 按标题切分 + RecursiveCharacterTextSplitter 递归切割，最小块合并保护
- `node_item_name_recognition`: 利用 chunks 上下文调用 LLM 识别文档所属的商品名称，写入 Milvus 的 item_name 集合
- `node_bge_embedding`: **未实现（Stub）**，计划用 BGE-M3 对每个 chunk 生成稠密+稀疏向量
- `node_import_milvus`: **未实现（Stub）**，计划将向量数据写入 Milvus

`node_entry` 处的条件路由 (`route_entry`): PDF 文件 → `node_pdf_to_md`，Markdown 文件 → 直接跳到 `node_md_img`。

### 状态管理

全局状态定义在 `app/import_process/agent/state.py`，使用 `TypedDict`（`ImportGraphState`）。所有节点通过修改同一字典传递数据。创建状态时使用 `create_default_state(**overrides)` 而非直接构造。

### 模块分层

| 层 | 路径 | 职责 |
|---|---|---|
| 流水线节点 | `app/import_process/agent/nodes/` | 7 个 LangGraph 节点 |
| 流水线编排 | `app/import_process/agent/main_graph.py` | 图定义、节点注册、边连接 |
| 状态定义 | `app/import_process/agent/state.py` | `ImportGraphState` TypedDict |
| 外部客户端 | `app/clients/` | Milvus、MinIO、MongoDB、Neo4j 连接（均为单例模式） |
| 配置 | `app/conf/` | dataclass 风格配置，全部从 `.env` 读取 |
| LM 工具 | `app/lm/` | LLM 客户端、BGE-M3 嵌入、Reranker 工具 |
| 通用工具 | `app/utils/` | 路径、格式化、限流、SSE 推送、任务追踪 |
| Prompt 模板 | `prompts/` | `.prompt` 文件，通过 `load_prompt(name, **kwargs)` 渲染 |

### 配置模式

所有配置类使用 `@dataclass` + `os.getenv()` 模式（见 `app/conf/`）。`load_dotenv()` 在每个配置模块顶部调用。需要新增配置项时的流程：`.env` 添加变量 → 对应 `conf/*.py` 添加字段 → 业务代码引用。

### 关键约定

- **日志**: 使用 `from app.core.logger import logger`，不要用标准库 logging。logger 已全局 patch 过，能自动显示正确的调用位置。
- **LLM 客户端**: 通过 `lm_utils.get_llm_client(model, json_mode)` 获取，带全局缓存，不要直接 new `ChatOpenAI`。
- **Milvus 客户端**: 通过 `get_milvus_client()` 获取，单例模式，需判空。
- **MinIO 客户端**: 通过 `get_minio_client()` 获取，模块导入时自动初始化。
- **BGE-M3**: 通过 `embedding_utils.generate_embeddings(texts)` 生成向量，返回 `{"dense": [...], "sparse": [...]}`。
- **Prompt**: 使用 `load_prompt('prompt_name', var=value)` 加载并渲染 `prompts/` 目录下的模板文件。