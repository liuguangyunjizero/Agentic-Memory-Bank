# Testing Guide

本文档说明如何安装依赖、运行测试和使用示例。

## 1. 安装依赖

### 步骤 1.1: 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 步骤 1.2: 安装依赖包

```bash
cd "d:\Material\NYUsh RA\Agentic-Memory-Bank"
pip install -r requirements.txt
```

**依赖包说明**：
- `sentence-transformers`: 用于计算文本的embedding向量
- `numpy`: 用于向量计算
- `networkx`: 用于图结构管理

**首次安装注意事项**：
- `sentence-transformers` 会自动下载模型 `all-MiniLM-L6-v2`（约90MB）
- 首次运行时会需要一些时间下载模型
- 模型会缓存在本地，后续运行会很快

---

## 2. 运行测试

### 2.1 集成测试（推荐先运行）

集成测试会验证完整的工作流程：

```bash
cd "d:\Material\NYUsh RA\Agentic-Memory-Bank"
python tests/test_integration.py
```

**测试内容**：
1. 创建 MemoryManager
2. 创建任务
3. 添加待办任务
4. 添加不同类型的记忆（文本、网页、代码）
5. 检索记忆（文本查询、关键词、标签）
6. 获取完整记忆上下文
7. 查看统计信息
8. 测试 InsightDoc 上下文生成
9. 保存和加载 JSON 文件

**预期输出**：
```
Testing complete workflow...

1. Creating Memory Manager...
   ✓ Created: MemoryManager(nodes=0, edges=0, trees=0)

2. Creating Task...
   ✓ Task created: task_20240115_xxxxx

...

✅ All tests passed!
```

### 2.2 基础 CRUD Demo

演示基本的增删改查操作：

```bash
python examples/demo_basic_crud.py
```

**演示内容**：
- CREATE: 添加记忆
- READ: 检索记忆（多种方式）
- UPDATE: 修改记忆
- 任务管理
- 统计信息
- 保存/加载

---

## 3. 验证各层功能

### 3.1 验证 InsightDoc 层

创建一个简单的测试脚本：

```python
from core import InsightDoc

# 创建任务
doc = InsightDoc(
    user_question="研究Python异步编程",
    understood_goal="学习asyncio库和async/await语法"
)

# 添加待办任务
doc.add_pending_tasks([
    "搜索asyncio文档",
    "学习async/await语法",
    "对比asyncio和threading"
])

# 设置当前任务
doc.set_current_task("搜索asyncio文档")

# 完成任务
doc.complete_current_task(
    result="找到了官方文档和基础示例",
    impact="准备深入学习语法"
)

# 查看上下文
print(doc.get_current_task_context())
```

### 3.2 验证 QueryGraph 层

```python
from core import QueryGraph
from utils import EmbeddingManager
import numpy as np

# 创建图
embedding_mgr = EmbeddingManager()
graph = QueryGraph(embedding_manager=embedding_mgr)

# 添加节点
embedding = embedding_mgr.compute_embedding("Python异步编程基础")
node_id = graph.add_node(
    summary="Python异步编程基础",
    keywords=["python", "async"],
    tags=["programming"],
    embedding=embedding,
    interaction_refs={"text": []}
)

# 检索
results = graph.query_by_keywords(["python"])
print(f"找到 {len(results)} 个节点")

# 添加边
node_id2 = graph.add_node(
    summary="asyncio事件循环",
    keywords=["python", "asyncio", "event_loop"],
    tags=["programming"],
    embedding=embedding_mgr.compute_embedding("asyncio事件循环"),
    interaction_refs={"text": []}
)

graph.add_edge(node_id, node_id2, edge_type="related", note="深入关系")

# 查看邻居
neighbors = graph.get_neighbors(node_id)
print(f"节点 {node_id} 有 {len(neighbors)} 个邻居")
```

### 3.3 验证 InteractionTree 层

```python
from core import InteractionTree

# 创建树
tree = InteractionTree()

# 创建根节点
root_id = tree.create_tree_root(query_node_id="node_001")

# 添加文本分支
text_id = tree.add_text_branch(
    parent_id=root_id,
    text="这是一段详细的交互历史..."
)

# 添加网页分支
webpage_id = tree.add_webpage_branch(
    parent_id=root_id,
    url="https://example.com",
    title="示例网页",
    parsed_text="网页内容..."
)

# 获取完整树
full_tree = tree.get_full_tree(root_id)
print(f"树结构: {full_tree}")
```

---

## 4. 常见问题排查

### 4.1 导入错误

**错误**: `ModuleNotFoundError: No module named 'sentence_transformers'`

**解决**:
```bash
pip install sentence-transformers
```

### 4.2 Embedding 模型下载失败

**错误**: 网络问题导致模型下载失败

**解决方案 1** - 使用镜像源：
```bash
pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**解决方案 2** - 手动下载模型：
1. 访问 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
2. 下载模型文件
3. 放置在本地目录
4. 修改代码指定本地路径

### 4.3 JSON 序列化错误

**错误**: `TypeError: Object of type ndarray is not JSON serializable`

**原因**: numpy 数组无法直接序列化

**检查**: 确保使用了 `utils.serialization` 模块的序列化函数

---

## 5. 性能测试

### 5.1 测试大规模节点

```python
from core import MemoryManager
import time

manager = MemoryManager()
manager.create_task("性能测试")

# 添加100个节点
start = time.time()
for i in range(100):
    manager.add_memory(
        summary=f"测试节点 {i}",
        keywords=[f"keyword_{i}", "test"],
        tags=["performance"],
        text_content=f"这是测试节点 {i} 的内容"
    )
end = time.time()

print(f"添加100个节点耗时: {end - start:.2f} 秒")

# 检索性能
start = time.time()
results = manager.retrieve_memories(query="测试", top_k=10)
end = time.time()

print(f"检索耗时: {end - start:.2f} 秒")
print(f"找到 {len(results)} 个结果")
```

### 5.2 测试 Embedding 计算速度

```python
from utils import EmbeddingManager
import time

embedding_mgr = EmbeddingManager()

# 单个文本
start = time.time()
embedding = embedding_mgr.compute_embedding("测试文本")
end = time.time()
print(f"单个embedding计算耗时: {end - start:.4f} 秒")

# 批量计算
texts = [f"测试文本 {i}" for i in range(100)]
start = time.time()
embeddings = embedding_mgr.batch_compute_embeddings(texts)
end = time.time()
print(f"批量计算100个embedding耗时: {end - start:.2f} 秒")
print(f"平均每个: {(end - start) / 100:.4f} 秒")
```

---

## 6. 调试技巧

### 6.1 启用详细日志

在测试脚本开头添加：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 6.2 检查数据结构

```python
# 检查节点详情
node = manager.query_graph.get_node(node_id)
print(f"节点ID: {node.id}")
print(f"摘要: {node.summary}")
print(f"关键词: {node.keywords}")
print(f"标签: {node.tags}")
print(f"状态: {node.metadata.status}")
print(f"访问次数: {node.metadata.access_count}")
print(f"Embedding维度: {node.embedding.shape}")

# 检查边
edges = manager.query_graph.get_edges(node_id)
for edge in edges:
    print(f"边: {edge.from_id} -> {edge.to_id} ({edge.edge_type})")

# 检查统计
stats = manager.get_statistics()
for key, value in stats.items():
    print(f"{key}: {value}")
```

### 6.3 检查保存的 JSON 文件

```python
import json

# 读取保存的文件
with open("memory_bank.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"节点数量: {len(data['query_graph']['nodes'])}")
print(f"边数量: {len(data['query_graph']['edges'])}")
print(f"树节点数量: {len(data['interaction_tree']['nodes'])}")
```

---

## 7. 下一步

系统基础功能已完成，后续可以：

1. **实现 Agent 组件**：
   - 分类/聚类Agent
   - 结构化Agent
   - 计划Agent
   - 记忆分析Agent

2. **实现适配器**：
   - 拦截和增强Prompt
   - 集成外部框架（ReAct, ReWOO等）
   - 深入检索工具

3. **优化和扩展**：
   - 添加更多检索策略
   - 实现记忆整理机制
   - 添加可视化功能

---

## 8. 获取帮助

如果遇到问题：

1. 检查 Python 版本（需要 3.10+）
2. 确认依赖包已正确安装
3. 查看错误堆栈信息
4. 参考本文档的常见问题部分

Happy testing! 🚀
