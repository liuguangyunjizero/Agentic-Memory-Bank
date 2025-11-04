# Agentic Memory Bank - 最终技术规范

> **版本**: 最终版（综合 v3.0, v4.0 及实现代码）+ 2025-01 优化版
> **更新日期**: 2025-01
> **面向**: 程序员和 Claude Code
> **重点**: 核心逻辑实现 + 实现优化记录

---

## 1. 系统概述

**目的**：层次化图结构的多智能体记忆管理系统，解决**单次任务中的长上下文问题**。

**核心特征**：
- 任务完成后清理记忆（非长期记忆系统）
- 会话制：一个任务 = 一个会话
- 学术验证导向（最小可行实现）
- 可集成到外部 Agent 框架

**应用场景**：
1. **深度研究**：多源信息检索、验证、综合分析
2. **长文档问答**：理解和回答长文档相关问题
3. **长对话推理**：多轮交互中的上下文管理

---

## 2. 系统架构

### 2.1 三层存储架构

**第一层：Insight Doc（任务状态层）**
- `task_goal: str` - 任务总目标
- `completed_tasks: List[CompletedTask]` - 已完成子任务
- `pending_tasks: List[str]` - 待办任务（通常 0-1 个元素）

**第二层：Query Graph（语义记忆图）**
- `nodes_dict: Dict[str, QueryGraphNode]` - 节点字典
- 实现方式：邻接表（`node.links = [id1, id2, ...]`）
- 边类型：`related`（无向边）

**第三层：Interaction Tree（交互历史层）**
- `entries: Dict[str, InteractionEntry]` - 所有交互条目
- `node_to_entries: Dict[str, List[str]]` - 节点到条目的映射
- `merge_events: List[MergeEvent]` - 节点合并事件
- 特性：只读，不修改历史

### 2.2 核心组件

**硬编码模块**（无需 LLM）：
- `EmbeddingModule`：SentenceTransformer（all-MiniLM-L6-v2）
- `RetrievalModule`：BM25 + Embedding 混合检索（α=0.5）
- `GraphOperations`：Query Graph 的 CRUD 操作
- `ContextUpdateModule`：更新节点元数据 + 重新计算 embedding

**LLM 驱动的 Agent**：
- `ClassificationAgent`：长上下文的主题聚类
- `StructureAgent`：内容压缩（30-50% 压缩率）
- `AnalysisAgent`：判断关系（冲突/相关/无关）
- `IntegrationAgent`：合并冲突节点
- `PlanningAgent`：增量式任务规划（0-1 个待办任务）

**工具集**：
- `SearchTool`：Serper API（必需）
- `VisitTool`：Jina Reader API（可选）+ BeautifulSoup（备选）
- `DeepRetrievalTool`：读取节点的完整 Interaction Tree

**执行框架**：
- `MultiTurnReactAgent`：Think-Act-Observe 循环
- 支持 `<tool_call>` 和 `<answer>` 标签
- Token 限制处理 + 强制终止

### 2.3 模块组织

```
src/
├── storage/           # 三层存储：insight_doc, query_graph, interaction_tree
├── modules/           # 硬编码：embedding, retrieval, graph_ops, context_update
├── agents/            # LLM 驱动：5 个 Agent
├── interface/         # Adapter（协调层）
├── tools/             # search, visit, deep_retrieval, react_agent
├── prompts/           # Agent 提示词模板
├── utils/             # llm_client, file_utils
└── config.py          # 配置管理
```

---

## 3. 数据结构

### 3.1 Insight Doc（任务状态层）

```python
@dataclass
class CompletedTask:
    type: TaskType          # NORMAL | CROSS_VALIDATE
    description: str        # 任务详细描述
    status: str            # "成功" | "失败"
    context: str           # 1-2 句话的总结

@dataclass
class InsightDoc:
    doc_id: str                          # 唯一标识符
    task_goal: str                       # 用户原始问题
    completed_tasks: List[CompletedTask]
    pending_tasks: List[str]             # 通常 0-1 个元素
```

**设计原则**：增量式规划
- `pending_tasks` 通常只有 **0-1 个元素**
- 不维护优先级队列，只决定**下一步**
- Planning Agent 动态调整

### 3.2 Query Graph（语义记忆图）

**节点定义**：
```python
@dataclass
class QueryGraphNode:
    id: str                    # UUID
    summary: str               # 结构化详细摘要
    context: str               # 一句话主题描述
    keywords: List[str]        # 关键词列表
    embedding: np.ndarray      # 语义向量（384 维，all-MiniLM-L6-v2）
    timestamp: float           # 创建时间
    links: List[str]           # 邻居节点 ID（related 边）
```

**图结构**：
- 实现：**邻接表**（dict + list，不使用 NetworkX）
- 存储：`{id: QueryGraphNode}`
- 边类型：`related`（无向边）
- 关系优先级：`conflict > related > unrelated`

**图操作**：
```python
class QueryGraph:
    nodes_dict: Dict[str, QueryGraphNode]  # {id: Node}

    def add_node(node: QueryGraphNode)
    def add_edge(node_id1: str, node_id2: str)
    def delete_node(node_id: str)
    def get_neighbors(node_id: str) -> List[QueryGraphNode]
```

### 3.3 Interaction Tree（历史层）

```python
@dataclass
class Attachment:
    id: str              # 附件唯一 ID
    type: AttachmentType # IMAGE | DOCUMENT | CODE
    content: str         # 文件路径（非文件内容）

@dataclass
class InteractionEntry:
    entry_id: str                     # UUID
    text: str                         # 完整文本
    timestamp: float
    metadata: Dict[str, Any]          # source, tool_calls 等
    attachments: List[Attachment]

@dataclass
class MergeEvent:
    event_id: str
    merged_node_ids: List[str]        # 旧节点 ID
    new_node_id: str                  # 新合并节点 ID
    timestamp: float
    description: str

class InteractionTree:
    entries: Dict[str, InteractionEntry]       # {entry_id: entry}
    node_to_entries: Dict[str, List[str]]      # {node_id: [entry_ids]}
    merge_events: List[MergeEvent]
```

**只读特性**：
- 永不修改历史条目
- 永不删除条目
- 节点合并时记录 `MergeEvent`，但保持原条目不变

---

## 4. 核心逻辑流程

### 4.1 初始化阶段

**输入**：`user_input`（文本上下文 + 问题）

**执行步骤**：

**步骤 1：解析用户输入**
- 提取：`text_context: str`
- 提取：`question: str`
- 生成：`doc_id: UUID`

**步骤 2：判断文本上下文**
- 如果 `text_context` 为空：跳转到步骤 7（仅规划）

**步骤 3：Classification Agent**
- 输入：`ClassificationInput(context=text_context, task_goal=question)`
- 如果 `token_count > CLASSIFICATION_AGENT_WINDOW`：
  - 分块处理（chunk_ratio=0.9）
  - 分别处理每个块
- 输出：`ClassificationOutput`
  - `should_cluster: bool`
  - `clusters: List[Cluster]`，每个 Cluster 包含：
    - `cluster_id: str`
    - `context: str`（一句话主题）
    - `content: str`（原始文本）
    - `keywords: List[str]`

**步骤 4：处理每个 cluster**

**4.1 Structure Agent**
- 输入：`StructureInput(content=cluster.content, context=cluster.context, keywords=cluster.keywords)`
- 输出：`StructureOutput(summary=compressed_summary)`
- 压缩率：30-50%

**4.2 创建节点**
- `node_id = UUID()`
- `embedding_text = f"{summary} {context} {' '.join(keywords)}"`
- `embedding = EmbeddingModule.compute_embedding(embedding_text)`
- 创建 `QueryGraphNode` 实例
- 调用 `GraphOperations.add_node(node)`

**4.3 检索**
- 调用 `RetrievalModule.hybrid_retrieval(query_embedding=node.embedding, query_keywords=node.keywords, graph=query_graph)`
- 返回：top-k 节点 + 1 跳邻居，按 timestamp 降序

**4.4 Analysis Agent**
- 输入：`AnalysisInput(new_node=NodeInfo(...), candidate_nodes=List[NodeInfo])`
- 对每个候选节点：
  - 判断关系优先级：conflict > related > unrelated
- 输出：`AnalysisOutput`
  - `relationships: List[Relationship]`，每个包含：
    - `existing_node_id: str`
    - `relationship: str`（"conflict" | "related" | "unrelated"）
    - `reasoning: str`
    - 如果是 conflict：`conflict_description: str`
    - 如果是 related：`context_update_new, context_update_existing, keywords_update_new, keywords_update_existing`

**4.5 处理关系**
- 遍历 `analysis_output.relationships`：
  - 如果 `relationship == "conflict"`：
    - 记录冲突：`conflicts.append({...})`
  - 如果 `relationship == "related"`：
    - 调用 `GraphOperations.add_edge(node.id, existing_node_id)`（双向更新 links）
    - 调用 `ContextUpdateModule.update_node_context(node.id, context_update_new, keywords_update_new)`（重新计算 embedding）
    - 调用 `ContextUpdateModule.update_node_context(existing_node_id, context_update_existing, keywords_update_existing)`

**4.6 创建 Interaction Tree 条目**
- 创建 `InteractionEntry` 实例
- 调用 `InteractionTree.add_entry(node.id, entry)`

**步骤 5：结束 cluster 循环**

**步骤 6：Planning Agent**
- 输入：`PlanningInput`，包含：
  - `insight_doc`（初始状态）
  - `new_memory_nodes`（新创建的节点）
  - `conflict_notification`（如果有冲突）
- 规划原则：
  - 优先级 1：处理冲突（Cross Validation 任务）
  - 优先级 2：普通任务执行
  - 增量式：只决定下一步（pending_tasks 只有 0-1 个元素）
- 输出：`PlanningOutput`
  - `task_goal: str`
  - `completed_tasks: List[CompletedTask]`
  - `pending_tasks: List[str]`（0-1 个元素）

**步骤 7：创建 Insight Doc**
- 使用 `PlanningOutput` 创建 `InsightDoc` 实例

**步骤 8：Adapter.enhance_prompt**
- 构建 `<task>` 部分：
  - task_goal
  - completed_tasks（带上下文）
  - pending_tasks
- 构建 `<memory>` 部分：
  - 如果 `pending_tasks` 为空："暂无相关记忆"
  - 否则：
    - `query_text = pending_tasks[0]`
    - 计算 query_embedding
    - 提取 query_keywords
    - 调用 `RetrievalModule.hybrid_retrieval(...)`
    - 格式化为"记忆 N：主题 + 关键词 + 摘要"
- 组合：`"<task>\n{task}\n</task>\n\n<memory>\n{memory}\n</memory>"`
- 返回：`enhanced_prompt`

**步骤 9：返回 enhanced_prompt 给 ReAct Agent**

### 4.2 执行循环

**输入**：`enhanced_prompt`（来自初始化或上一次迭代）

**执行步骤**：

**步骤 1：ReAct Agent 执行**

**1.1 调用 LLM**
- `MultiTurnReactAgent.run(enhanced_prompt)`
- 循环（最多 max_iterations 次）：
  - 用消息历史调用 LLM
  - 解析响应

**1.2 解析响应**
- 如果包含 `<tool_call>`：
  - 提取 JSON：`{"name": "tool_name", "arguments": {...}}`
  - 执行工具：
    - `search`：`SearchTool.call(arguments)`
      - Serper API：POST `https://google.serper.dev/search`
      - 返回：top-10 结果（title, url, snippet）
    - `visit`：`VisitTool.call(arguments)`
      - 如果有 JINA_API_KEY：Jina Reader API
      - 否则：BeautifulSoup 解析 HTML
      - 使用 LLM 提取相关内容（基于 goal）
    - `deep_retrieval`：`DeepRetrievalTool.call(arguments)`
      - `node_id = arguments["node_id"]`
      - `entries = InteractionTree.get_entries(node_id)`
      - 读取每个 entry 的文本
      - 返回：JSON 格式的所有条目
  - 格式化响应：`"<tool_response>{result}</tool_response>"`
  - 添加到消息历史
- 如果包含 `<answer>`：
  - 提取答案文本
  - 返回结果并终止
- 检查 token 数量：
  - 如果 > MAX_CONTEXT_TOKENS：强制生成答案

**1.3 返回结果**
- 返回：`{"question": str, "prediction": str, "messages": List[dict], "termination": str}`

**步骤 2：Adapter.intercept_context**

**2.1 判断任务类型**
- 从 `react_result["messages"]` 提取工具响应
- 确定 `task_type`：
  - 如果 pending_tasks[0] 包含"验证"或"Cross Validation"：`task_type = "CROSS_VALIDATE"`
  - 否则：`task_type = "NORMAL"`

**2.2 处理 CROSS_VALIDATE**
- 执行 `_handle_conflict_resolution()`：
  - 从 Insight Doc 提取 `conflicting_node_ids`
  - `validation_result = 合并的工具响应`
  - 从 Query Graph 加载冲突节点
  - 加载每个节点的邻居
  - 调用 Integration Agent：
    - 输入：`IntegrationInput(nodes_to_merge=[NodeWithNeighbors(...)], validation_result=validation_result)`
    - 输出：`IntegrationOutput`
      - `merged_node: {summary, context, keywords}`
      - `neighbor_updates: {neighbor_id: {context, keywords}}`
      - `interaction_tree_description: str`
  - 创建新合并节点：
    - 生成 `new_node_id`
    - 计算 `embedding`
    - 创建 `QueryGraphNode` 实例
  - 调用 `GraphOperations.merge_nodes(old_node_ids, new_node)`：
    - 添加 new_node
    - 继承所有旧节点的边（去重）
    - 删除旧节点
  - 更新邻居：
    - 遍历 `neighbor_updates`
    - 调用 `ContextUpdateModule.update_node_context(...)`
  - 记录合并事件：
    - 创建 `MergeEvent` 实例
    - 调用 `InteractionTree.record_merge(merge_event)`

**2.3 处理 NORMAL**
- 执行 `_handle_normal_task()`：
  - `combined_context = 连接所有工具响应`
  - 调用 `ClassificationAgent.run(combined_context)`
  - 对每个 cluster：
    - 调用 `StructureAgent.run(...)`
    - 创建节点
    - 调用 `RetrievalModule.hybrid_retrieval(...)`
    - 调用 `AnalysisAgent.run(...)`
    - 处理关系（添加边、更新上下文）
    - 创建 `InteractionEntry`
  - 如果检测到新冲突：记录 `conflict_notification`

**步骤 3：Planning Agent 更新 Insight Doc**
- 输入：`PlanningInput`（包含当前 insight_doc、新节点、新冲突等）
- 输出：`PlanningOutput`（更新的 completed_tasks 和 pending_tasks）
- 更新：`insight_doc = InsightDoc(...)`

**步骤 4：检查终止条件**
- 如果 `pending_tasks` 为空且所有 `completed_tasks.status == "成功"`：
  - 终止：返回最终结果
- 否则：继续下一次迭代

**步骤 5：为下一次迭代增强 Prompt**
- 调用 `Adapter.enhance_prompt(insight_doc)`
- 基于 `pending_tasks[0]` 检索相关记忆
- 构建增强 prompt
- 返回步骤 1（ReAct Agent 执行）

### 4.3 记忆转换过程（详细）

**触发时机**：执行循环中每次工具响应后

**输入**：工具响应文本（搜索结果、访问的页面内容等）

**处理流程**：

**1. 分类**
- 按主题拆分内容
- 对于长内容：分块处理（ratio=0.9）
- 输出：`List[Cluster]`（包含 context, content, keywords）

**2. 结构化（对每个 cluster）**
- 压缩内容至 30-50%
- 保留关键信息、证据、逻辑关系
- 输出：结构化摘要

**3. 节点创建**
- 组合 summary + context + keywords
- 计算 embedding
- 创建带 UUID 的 `QueryGraphNode`
- 添加到 Query Graph

**4. 检索**
- 混合检索：BM25 + Embedding
- 公式：`final_score = α * bm25_score + (1-α) * semantic_score`
- α = 0.5（默认）
- 返回：top-k 节点 + 1 跳邻居

**5. 关系分析**
- 优先级：conflict > related > unrelated
- 对每个候选节点：
  - 如果检测到冲突：
    - 记录 `conflict_description`
    - 停止分析（冲突优先级最高）
  - 如果相关：
    - 为两个节点生成上下文更新
    - 为两个节点生成关键词更新
  - 否则（无关）：跳过

**6. 边创建和上下文更新**
- 如果相关：
  - 在节点间添加无向边：
    - `node1.links.append(node2.id)`
    - `node2.links.append(node1.id)`
  - 更新 node1 的 context 和 keywords：
    - 重新计算 `node1.embedding`
  - 更新 node2 的 context 和 keywords：
    - 重新计算 `node2.embedding`

**7. Interaction Tree 记录**
- 创建带完整工具响应文本的 `InteractionEntry`
- 链接条目到节点：`node_to_entries[node.id].append(entry.entry_id)`
- 永不修改或删除现有条目

### 4.4 冲突解决（详细）

**触发时机**：Planning Agent 检测到冲突并创建 CROSS_VALIDATE 任务

**处理流程**：

**1. ReAct Agent 执行交叉验证**
- 使用 search/visit 工具验证冲突信息
- 返回 `validation_result`（哪个信息更准确/更新）

**2. Adapter 拦截 CROSS_VALIDATE 任务**
- 从 Insight Doc 提取 `conflicting_node_ids`
- 加载节点：`[node1, node2, ...]`
- 加载每个节点的邻居

**3. Integration Agent**
- 输入：
  - `nodes_to_merge: List[NodeWithNeighbors]`（包含 id, summary, context, keywords, neighbors）
  - `validation_result: str`（来自 ReAct Agent）
- 处理：
  - 分析 validation_result
  - 确定哪个信息更准确
  - 综合所有冲突节点的信息
  - 生成合并节点内容
  - 对每个继承的邻居：
    - 更新 context 以反映与合并节点的关系
    - 相应更新 keywords
- 输出：
  - `merged_node: {summary, context, keywords}`
  - `neighbor_updates: {neighbor_id: {context, keywords}}`
  - `interaction_tree_description: str`

**4. 执行合并**
- 创建带合并内容的 `new_node`
- 计算 `new_node.embedding`
- 调用 `GraphOperations.merge_nodes(old_node_ids, new_node)`：
  - 将 new_node 添加到图
  - 收集所有旧节点的邻居（去重，排除被合并的节点）
  - 在 new_node 和所有邻居之间创建边
  - 删除旧节点

**5. 更新邻居**
- 遍历 `neighbor_updates`
- 对每个 neighbor_id：
  - 调用 `ContextUpdateModule.update_node_context(neighbor_id, new_context, new_keywords)`
  - 重新计算 `neighbor.embedding`

**6. 记录合并事件**
- 创建 `MergeEvent` 实例
- 调用 `InteractionTree.record_merge(merge_event)`
- 重新链接条目：
  - 将旧节点的条目关联到新节点
  - 保持旧条目不变（只读）

**7. Planning Agent 更新 Insight Doc**
- 标记 CROSS_VALIDATE 任务为已完成
- 规划下一个任务（通常回到普通任务）

---

## 5. API 和配置

### 5.1 必需 API

**LLM API**（必需）：
- DeepSeek API：`https://api.deepseek.com/v1`
- OpenAI 兼容端点
- 模型：`deepseek-chat` 或兼容模型

**Serper API**（必需）：
- 端点：`https://google.serper.dev/search`
- 方法：POST
- 用途：网络搜索
- **无备选方案** - 必需

**Jina Reader API**（可选）：
- 端点：`https://r.jina.ai/{url}`
- 方法：GET
- 用途：高质量网页解析
- 备选：BeautifulSoup HTML 解析

**Embedding**（本地，无需 API）：
- 模型：`all-MiniLM-L6-v2`（SentenceTransformer）
- 本地运行，无外部 API 调用

### 5.2 配置参数

```python
# config.py

class Config:
    # === LLM 配置 ===
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = "deepseek-chat"
    LLM_TEMPERATURE: float = 0.6
    LLM_MAX_TOKENS: int = 4096

    # === Embedding 配置 ===
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # === 检索配置 ===
    RETRIEVAL_K: int = 5           # top-k 节点数
    RETRIEVAL_ALPHA: float = 0.5   # BM25 权重（1-α 为 embedding 权重）

    # === API 密钥 ===
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")  # 必需
    JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")      # 可选

    # === ReAct 配置 ===
    MAX_LLM_CALL_PER_RUN: int = 60
    MAX_CONTEXT_TOKENS: int = 32000

    # === Agent 窗口配置 ===
    CLASSIFICATION_AGENT_WINDOW: int = 8000
    STRUCTURE_AGENT_WINDOW: int = 8000
    ANALYSIS_AGENT_WINDOW: int = 8000
    INTEGRATION_AGENT_WINDOW: int = 8000
    PLANNING_AGENT_WINDOW: int = 8000

    # === 长上下文处理 ===
    CHUNK_RATIO: float = 0.9  # 使用 90% 窗口，留 10% 缓冲

    # === 文件路径 ===
    TEMP_DIR: str = "data/temp"
    STORAGE_DIR: str = "data/storage"
```

### 5.3 环境变量

```bash
# .env 文件

# LLM API（必需）
LLM_API_KEY=sk-your-deepseek-api-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 搜索 API（必需）
SERPER_API_KEY=your-serper-api-key

# Jina Reader API（可选）
JINA_API_KEY=your-jina-api-key
```

---

## 6. 实现细节

### 6.1 混合检索算法

```python
def hybrid_retrieval(
    query_embedding: np.ndarray,
    query_keywords: List[str],
    graph: QueryGraph,
    k: int = 5,
    alpha: float = 0.5
) -> List[QueryGraphNode]:
    """
    混合检索：BM25 + Embedding

    返回：top-k 节点 + 1 跳邻居，按 timestamp 降序
    """

    # 1. 计算所有节点的分数
    all_nodes = list(graph.nodes_dict.values())
    scored_nodes = []

    for i, node in enumerate(all_nodes):
        # BM25 分数（关键词匹配）
        bm25_score = bm25.get_scores(query_keywords)[i]
        bm25_score_normalized = bm25_score / max(bm25.get_scores(query_keywords))

        # Embedding 分数（语义相似度）
        semantic_score = np.dot(query_embedding, node.embedding)

        # 混合分数
        final_score = alpha * bm25_score_normalized + (1 - alpha) * semantic_score
        scored_nodes.append((node.id, final_score))

    # 2. 选择 top-k
    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    top_k_ids = [node_id for node_id, _ in scored_nodes[:k]]

    # 3. 扩展 1 跳邻居
    result_ids = set(top_k_ids)
    for node_id in top_k_ids:
        neighbors = graph.get_neighbors(node_id)
        for neighbor in neighbors:
            result_ids.add(neighbor.id)

    # 4. 按 timestamp 降序排序
    result_nodes = [graph.nodes_dict[nid] for nid in result_ids]
    result_nodes.sort(key=lambda n: n.timestamp, reverse=True)

    return result_nodes
```

### 6.2 增量式规划策略

**设计原则**：只决定**下一步**，不制定完整计划

**实现**：
```python
def planning_agent_logic(insight_doc, new_memory_nodes, conflict_notification, multimodal_info):
    """
    Planning Agent 逻辑：增量式规划，0-1 个待办任务
    """

    pending_tasks = []

    # 优先级 1：处理冲突
    if conflict_notification:
        task = f"交叉验证：验证冲突节点 {conflict_notification['conflicting_node_ids']}"
        pending_tasks.append(task)
        return pending_tasks  # 立即返回，只有 1 个任务

    # 优先级 2：普通任务
    # 分析 task_goal 和 completed_tasks 确定是否需要更多工作
    if is_task_complete(insight_doc):
        return []  # 无待办任务，准备终止
    else:
        next_task = determine_next_normal_task(insight_doc, new_memory_nodes)
        pending_tasks.append(next_task)
        return pending_tasks
```

**关键特性**：
- `pending_tasks` 大多数时候只有 **0-1 个元素**
- 永不维护长任务队列
- 根据执行结果动态调整
- 冲突任务优先级更高

### 6.3 Prompt 增强结构

**格式**：
```xml
<task>
任务目标: {task_goal}

已完成的子任务:
1. [NORMAL] 搜索量子计算信息 - 成功
   知识上下文: 找到 10 篇关于 2024 年量子计算突破的文章
2. [NORMAL] 访问量子纠错文章 - 成功
   知识上下文: 关于 Google 2024 年量子纠错成就的详细信息

待办任务:
1. 总结所有信息并提供最终答案

</task>

<memory>
记忆 1:
主题: 2024 年量子计算突破
关键词: quantum, error correction, Google, breakthrough
摘要: 2024 年，Google 在量子纠错方面取得重大突破...

记忆 2:
主题: IBM 量子处理器开发
关键词: IBM, quantum processor, 1000 qubits
摘要: IBM 宣布推出 1000+ 量子比特处理器...

</memory>

请根据以上任务和记忆，执行下一步操作。
```

**记忆检索逻辑**：
- 基于 `pending_tasks[0]`（当前任务）
- 从任务描述计算 query embedding
- 从任务描述提取关键词
- 混合检索（k=5）
- 将检索到的节点格式化为结构化记忆

### 6.4 工具调用格式

**ReAct Agent 工具调用**：
```xml
<think>
我需要搜索量子计算信息。
</think>

<tool_call>
{"name": "search", "arguments": {"query": ["quantum computing 2024", "quantum breakthrough"]}}
</tool_call>
```

**工具响应格式**：
```xml
<tool_response>
Google 搜索 'quantum computing 2024' 找到 10 个结果:
1. [Google 的量子突破](https://example.com)
   来源: Nature
   摘要: Google 宣布在量子纠错方面取得突破...
...
</tool_response>
```

**答案格式**：
```xml
<think>
基于收集到的所有信息，我现在可以提供全面的答案。
</think>

<answer>
量子计算在 2024 年取得了几项重大突破：
1. Google 的量子纠错成就...
2. IBM 发布 1000+ 量子比特处理器...
...
</answer>
```

### 6.5 长上下文处理

**问题**：输入超过 agent 窗口大小

**解决方案**：分块处理（ratio=0.9）

```python
def handle_long_context(context: str, window_size: int, chunk_ratio: float = 0.9):
    """
    将长上下文拆分成块进行处理
    """

    token_count = count_tokens(context)

    if token_count <= window_size:
        return [context]  # 无需拆分

    # 计算块大小（使用 90% 窗口，留 10% 缓冲）
    chunk_size = int(window_size * chunk_ratio)

    # 按段落边界拆分（而非字符数）
    paragraphs = context.split('\n\n')
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if current_tokens + para_tokens <= chunk_size:
            current_chunk += para + "\n\n"
            current_tokens += para_tokens
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para + "\n\n"
            current_tokens = para_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
```

**应用**：
- Classification Agent：拆分输入上下文
- Structure Agent：处理长 cluster 内容
- 独立处理每个块
- 聚合结果

---

## 7. Agent 规范

### 7.1 Classification Agent

**输入**：
```python
@dataclass
class ClassificationInput:
    context: str                    # 待分类的长上下文
    task_goal: Optional[str] = None # 用户问题（参考）
```

**输出**：
```python
@dataclass
class ClassificationOutput:
    should_cluster: bool           # 是否需要聚类
    clusters: List[Cluster]

@dataclass
class Cluster:
    cluster_id: str
    context: str                   # 一句话主题
    content: str                   # 原始文本
    keywords: List[str]
```

**逻辑**：
1. 检查上下文是否超过窗口大小
2. 如果是：拆分成块（ratio=0.9）
3. 对每个块：识别主题并按主题聚类
4. 输出：带 context、content、keywords 的 cluster 列表

### 7.2 Structure Agent

**输入**：
```python
@dataclass
class StructureInput:
    content: str              # 原始内容
    context: str              # 主题描述（参考）
    keywords: List[str]       # 关键词（参考）
```

**输出**：
```python
@dataclass
class StructureOutput:
    summary: str              # 结构化摘要（30-50% 压缩）
```

**逻辑**：
1. 压缩内容同时保持关键信息
2. 保留证据、逻辑关系
3. 使用结构化格式（要点列表、层次结构）
4. 目标：原始长度的 30-50%

### 7.3 Analysis Agent

**输入**：
```python
@dataclass
class AnalysisInput:
    new_node: NodeInfo                 # 新节点
    candidate_nodes: List[NodeInfo]    # 检索到的候选节点

@dataclass
class NodeInfo:
    id: str = None  # 新节点为 None
    summary: str
    context: str
    keywords: List[str]
```

**输出**：
```python
@dataclass
class AnalysisOutput:
    relationships: List[Relationship]

@dataclass
class Relationship:
    existing_node_id: str
    relationship: str  # "conflict" | "related" | "unrelated"
    reasoning: str

    # 冲突特有字段
    conflict_description: Optional[str] = None

    # 相关特有字段
    context_update_new: Optional[str] = None
    context_update_existing: Optional[str] = None
    keywords_update_new: Optional[List[str]] = None
    keywords_update_existing: Optional[List[str]] = None
```

**逻辑**：
1. 对每个候选节点：
   - 检查冲突：是否矛盾？
     - 如果是：返回冲突及描述，停止
   - 检查相关：语义/主题是否相关？
     - 如果是：返回相关，及两者的 context/keyword 更新
   - 否则：返回无关
2. 优先级：**conflict > related > unrelated**

### 7.4 Integration Agent

**输入**：
```python
@dataclass
class IntegrationInput:
    nodes_to_merge: List[NodeWithNeighbors]
    validation_result: str

@dataclass
class NodeWithNeighbors:
    id: str
    summary: str
    context: str
    keywords: List[str]
    neighbors: List[Dict[str, Any]]  # [{"id": ..., "context": ..., "keywords": ...}]
```

**输出**：
```python
@dataclass
class IntegrationOutput:
    merged_node: Dict[str, Any]        # {"summary": str, "context": str, "keywords": List[str]}
    neighbor_updates: Dict[str, Dict]  # {neighbor_id: {"context": str, "keywords": List[str]}}
    interaction_tree_description: str  # 合并描述
```

**逻辑**：
1. 分析 validation_result 确定哪个信息正确/更新
2. 综合所有冲突节点的信息
3. 生成合并节点，包含：
   - 组合的 summary（优先验证过的信息）
   - 更新的 context 反映合并状态
   - 合并的 keywords（去重）
4. 对每个继承的邻居：
   - 更新邻居的 context 以反映与合并节点的关系
   - 相应更新邻居的 keywords
5. 生成 Interaction Tree 合并事件的描述

### 7.5 Planning Agent

**输入**：
```python
@dataclass
class PlanningInput:
    insight_doc: InsightDoc
    new_memory_nodes: Optional[List[QueryGraphNode]] = None
    conflict_notification: Optional[ConflictNotification] = None

@dataclass
class ConflictNotification:
    conflicting_node_ids: List[str]
    conflict_description: str
```

**输出**：
```python
@dataclass
class PlanningOutput:
    task_goal: str
    completed_tasks: List[CompletedTask]
    pending_tasks: List[str]  # 0-1 个元素
```

**逻辑**：
1. 分析当前状态（已完成任务、新记忆、冲突）
2. 应用优先级规则：
   - **优先级 1**：如果有冲突 → 创建 CROSS_VALIDATE 任务
   - **优先级 2**：基于 task_goal 确定下一个普通任务
3. 终止检查：
   - 如果 task_goal 已实现 → pending_tasks = []
   - 否则 → pending_tasks = [next_task]
4. 如适用，更新 completed_tasks
5. 返回：只有 **0-1 个**待办任务（增量式规划）

---

## 8. 文件和目录结构

```
Agentic-Memory-Bank/
├── src/
│   ├── storage/
│   │   ├── insight_doc.py           # InsightDoc, CompletedTask, TaskType
│   │   ├── query_graph.py           # QueryGraph, QueryGraphNode
│   │   └── interaction_tree.py      # InteractionTree, InteractionEntry, MergeEvent
│   ├── modules/
│   │   ├── embedding.py             # EmbeddingModule
│   │   ├── retrieval.py             # RetrievalModule（BM25 + Embedding）
│   │   ├── graph_ops.py             # GraphOperations
│   │   └── context_update.py        # ContextUpdateModule
│   ├── agents/
│   │   ├── base_agent.py            # BaseAgent（共享功能）
│   │   ├── classification_agent.py  # ClassificationAgent
│   │   ├── structure_agent.py       # StructureAgent
│   │   ├── analysis_agent.py        # AnalysisAgent
│   │   ├── integration_agent.py     # IntegrationAgent
│   │   └── planning_agent.py        # PlanningAgent
│   ├── interface/
│   │   └── adapter.py               # MemoryBankAdapter
│   ├── tools/
│   │   ├── search_tool.py           # SearchTool（Serper）
│   │   ├── visit_tool.py            # VisitTool（Jina/BeautifulSoup）
│   │   ├── deep_retrieval_tool.py   # DeepRetrievalTool
│   │   └── react_agent.py           # MultiTurnReactAgent
│   ├── prompts/
│   │   └── agent_prompts.py         # 所有 agent 提示词
│   ├── utils/
│   │   ├── llm_client.py            # LLMClient（带重试机制）
│   │   └── file_utils.py            # FileUtils
│   ├── config.py                    # Config 类
│   └── memory_bank.py               # MemoryBank（主类）
├── tests/
│   ├── test_storage.py
│   ├── test_modules.py
│   ├── test_agents.py
│   ├── test_interface_tools.py
│   └── test_memory_bank.py
├── data/
│   └── storage/                     # 存储目录
├── main.py                          # CLI 入口
├── requirements.txt                 # Python 依赖
├── .env.example                     # 环境变量模板
├── .env                             # 实际环境变量（不在 git 中）
└── README.md                        # 用户文档
```

---

## 9. 依赖

```
# requirements.txt

# 核心依赖
sentence-transformers>=2.2.2
openai>=1.12.0
numpy>=1.24.3
requests>=2.31.0
beautifulsoup4>=4.12.0
rank-bm25>=0.2.2

# 测试
pytest>=7.4.0

# 工具
python-dotenv>=1.0.0
```

**注意**：不使用 NetworkX。Query Graph 使用邻接表（dict + list）。

---

## 10. 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **图存储** | 邻接表（dict + list） | 比 NetworkX 更轻量，与 A-mem 设计兼容 |
| **检索** | BM25 + Embedding（α=0.5） | 平衡关键词和语义匹配 |
| **Agent 架构** | 5 个独立 LLM agent | 单一职责，易于调试和测试 |
| **Prompt 增强** | `<task>` + `<memory>` 结构 | 为 ReAct Agent 提供结构化上下文 |
| **增量式规划** | 只有 0-1 个待办任务 | 避免过早规划，动态适应 |
| **工具集成** | search, visit, deep_retrieval | 覆盖常见研究场景 |
| **API 选择** | Serper（必需），Jina（可选） | 质量和稳定性优于免费替代品 |
| **Embedding** | SentenceTransformer（本地） | 无 API 成本，推理快速 |
| **无 Mock 模式** | 仅生产 API | 避免测试-生产差异 |
| **关系优先级** | conflict > related > unrelated | 立即处理冲突 |

---

## 11. 执行入口

```python
# main.py

from src.memory_bank import MemoryBank
from src.config import Config

def main():
    # 初始化
    config = Config()
    memory_bank = MemoryBank(config)

    # 用户输入
    user_input = """
    上下文: 量子计算利用量子叠加和量子纠缠。

    问题: 量子计算有哪些实际应用？
    """

    # 运行
    result = memory_bank.run(user_input)

    # 输出
    print(f"任务: {result['insight_doc']['task_goal']}")
    print(f"已完成: {len(result['insight_doc']['completed_tasks'])}")
    print(f"节点数: {len(result['query_graph']['nodes'])}")

    # 导出
    memory_bank.export_memory("output.json")

if __name__ == "__main__":
    main()
```

**MemoryBank.run() 伪代码**：
```python
def run(self, user_input: str) -> Dict[str, Any]:
    # 1. 初始化阶段
    enhanced_prompt = self._initialize(user_input)

    # 2. 执行循环
    while not self._should_terminate():
        # 2.1 ReAct 执行
        react_result = self.react_agent.run(enhanced_prompt)

        # 2.2 上下文拦截（将工具响应转换为记忆）
        tool_responses = self._extract_tool_responses(react_result["messages"])
        if tool_responses:
            self.adapter.intercept_context(
                context=combined_context,
                task_type=self._infer_task_type(),
                insight_doc=self.insight_doc
            )

        # 2.3 更新 Insight Doc（Planning Agent）
        self._update_insight_doc()

        # 2.4 检查终止条件
        if not self.insight_doc.pending_tasks:
            break

        # 2.5 为下一次迭代增强 prompt
        enhanced_prompt = self.adapter.enhance_prompt(self.insight_doc)

    # 3. 清理并返回
    self.adapter.cleanup_temp_storage()
    return {
        "insight_doc": self.insight_doc.to_dict(),
        "query_graph": self.query_graph.to_dict(),
        "interaction_tree": self.interaction_tree.to_dict()
    }
```

---

## 12. 终止条件

**主要条件**：
```python
def should_terminate() -> bool:
    return (
        len(insight_doc.pending_tasks) == 0 and
        all(task.status == "成功" for task in insight_doc.completed_tasks)
    )
```

**备选条件**（在 ReAct Agent 内）：
- Token 限制超出：`token_count > MAX_CONTEXT_TOKENS`
- 达到最大迭代次数：`iterations >= MAX_LLM_CALL_PER_RUN`
- Agent 返回 `<answer>` 标签

**终止时返回**：
- 返回包含 Insight Doc、Query Graph、Interaction Tree 的最终结果

---

## 13. 错误处理

**LLM 客户端重试机制**：
```python
def call_llm_with_retry(messages, max_retries=10):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(...)
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                return f"错误: {max_retries} 次尝试后失败"
            time.sleep(min(2 ** attempt, 30))  # 指数退避
```

**工具调用错误处理**：
```python
def handle_tool_call(tool_call_json):
    try:
        tool_name = tool_call_json["name"]
        tool_args = tool_call_json["arguments"]
        result = tools[tool_name].call(tool_args)
        return f"<tool_response>{result}</tool_response>"
    except Exception as e:
        return f"<tool_response>错误: {str(e)}</tool_response>"
```

---

## 14. 参考实现映射

### 来自 A-mem（Query Graph）

| 我们的实现 | A-mem 对应 |
|-----------|-----------|
| `QueryGraphNode` | `MemoryNote` |
| `QueryGraphNode.links` | `MemoryNote.links` |
| `QueryGraph.nodes_dict` | `AgenticMemorySystem.memories` |
| `RetrievalModule` | `HybridRetriever` |
| `EmbeddingModule` | `SimpleEmbeddingRetriever` |

### 来自 WebResummer（ReAct 框架）

| 我们的实现 | WebResummer 对应 |
|-----------|-----------------|
| `MultiTurnReactAgent` | `MultiTurnReactAgent` |
| `MultiTurnReactAgent.run()` | `MultiTurnReactAgent._run()` |
| `LLMClient.call()` | `call_server()` |
| `SearchTool` | `tool_search.py` |
| `VisitTool` | `tool_visit.py` |

---

**规范结束**

本文档专注于核心逻辑实现。所有架构决策、数据流和算法均以程序员可读格式描述，无冗余解释或视觉图表。
