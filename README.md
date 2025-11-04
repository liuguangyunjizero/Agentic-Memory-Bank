# Agentic Memory Bank

> **层次化图结构的多智能体记忆管理系统**
> 专门解决单次任务长上下文问题的学术研究项目

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

复制 `.env.example` 为 `.env` 并填入你的 API 密钥：

### 3. 运行

**直接提问（推荐）**：
```bash
python main.py "Among CS conferences, in 2025, which conference has exactly the same full paper submission deadline and the same CCF rank as IJCAI?"
```

**交互式模式**：
```bash
python main.py --interactive
```

**从文件读取**：
```bash
python main.py --file input.txt --output memory.json
```

**调试模式（显示详细日志）**：
```bash
python main.py "你的问题" --debug
```

---

## 核心特性

- **三层存储架构**：Insight Doc (任务状态) + Query Graph (语义记忆) + Interaction Tree (交互历史)
- **多智能体协作**：Classification, Structure, Analysis, Integration, Planning 五个 Agent
- **ReAct 框架**：Think-Act-Observe 循环，支持 search, visit, deep_retrieval 工具
- **混合检索**：BM25 + Embedding，自动检测冲突并解决
- **增量式规划**：动态任务调整，只维护 0-1 个待办任务

---

## 项目结构

```
src/
├── memory_bank.py           # 主类
├── storage/                 # 三层存储
│   ├── insight_doc.py
│   ├── query_graph.py
│   └── interaction_tree.py
├── agents/                  # 5个LLM Agent
│   ├── classification_agent.py
│   ├── structure_agent.py
│   ├── analysis_agent.py
│   ├── integration_agent.py
│   └── planning_agent.py
├── tools/                   # 工具集
│   ├── search_tool.py
│   ├── visit_tool.py
│   └── react_agent.py
├── modules/                 # 硬编码模块
│   ├── embedding.py
│   ├── retrieval.py
│   └── graph_ops.py
└── interface/
    └── adapter.py           # 核心协调器
```

---

## Python API

```python
from src.memory_bank import MemoryBank
from src.config import Config

# 初始化
config = Config()
memory_bank = MemoryBank(config)

# 运行查询
result = memory_bank.run("你的问题")

# 导出记忆
memory_bank.export_memory("memory.json")
```

---

## 技术文档

- **[REQUIREMENTS_FINAL.md](REQUIREMENTS_FINAL.md)** - 完整的技术规范和实现细节
- **[tests/test_end_to_end.py](tests/test_end_to_end.py)** - 端到端测试示例

---

## 注意事项

- 本项目专注于**单次任务**的长上下文管理，任务完成后记忆会被清理
- 需要 Serper API 密钥（网络搜索必需）
- Jina Reader API 可选，但推荐使用以获得更好的网页解析质量

---

**⭐ 如果这个项目对您有帮助，欢迎 Star！**
